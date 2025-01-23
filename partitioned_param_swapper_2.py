# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping tensors to/from (NVMe) storage devices.
"""

import os
import shutil
from enum import Enum
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import AsyncIOBuilder
from .constants import *
from .utils import swap_in_tensors, swap_out_tensors, MIN_AIO_BYTES, AIO_ALIGNED_BYTES, print_object, SwapBufferPool

from .utils import swap_in_tensors_sync, swap_out_tensors_sync, swap_in_tensors_sync_2, swap_out_tensors_sync_2

import time

def print_rank_0(message, debug=False, force=False):
    if dist.get_rank() == 0 and (debug or force):
        print(message)


class PartitionedParamStatus(Enum):
    # Partitioned parameters are present and ready for use
    AVAILABLE = 1

    # partitioned params are in some non-memory device
    NOT_AVAILABLE = 2

    # partitioned params are being read from some non-memory device.
    INFLIGHT = 3


class AsyncPartitionedParameterSwapper(object):

    def __init__(self, ds_config, pinned_buffer_information, nvme_layers, model_dtype, layer_id_to_rank, rank):
        # 创建了一个 AsyncIOBuilder 类的实例。这是一个 OpBuilder 类，用于构建异步 I/O 相关的操作。
        # 对刚创建的 AsyncIOBuilder 实例调用了 load 方法。这个方法用于加载异步 I/O 操作的模块
        # 实质上就是 Loads a PyTorch C++ extension just-in-time (JIT).
        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle
        self.dtype = model_dtype
        self.rank = rank
        self.layer_id_to_rank = layer_id_to_rank

        #set swap buffers, create aio handles
        # 1.构建nvme路径并创建新的文件夹
        # 2.根据配置信息计算每个buffer对齐后的元素数量
        # 3.buffers: 创建一个一维的空tensor，锁定在内存中。大小为buffer对齐后的元素数量 × buffer数量
        # 4.aio handles：创建两个aio_handle(C++)类，用于读和写
        self._configure_aio_2(ds_config, pinned_buffer_information, nvme_layers)

        #mapping from param id to path
        self.id_to_path = {}

        #mapping from pram_id to buffer id
        self.param_id_to_buffer_id = {}

        # mapping from param_id to swap buffer
        # 将当前param_id直接映射到真正的buffer上（本质上是buffer的narrow）
        self.param_id_to_swap_buffer = {}

        #number of elements in the param
        # 📌参数量的映射是 划分后的参数量
        self.param_id_to_numel = {}

        self.pending_writes = 0
        self.pending_reads = 0

        #keep track of async swap in params and buffers
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

        #keep track of available params
        self.available_params = set()
        self.available_numel = 0

        # for swapping out from partitioned fp32 params
        self.partitioned_swap_buffer = None
        self.partitioned_swap_pool = None

        # 创建了一个包含单个元素值 1 的 PyTorch 张量，类型 torch.float16
        self.invalid_buffer = torch.tensor(1).half()

        if self.rank == 0:
            exclude_list = ['aio_read_handle', 'aio_write_handle', 'buffers']
            print_object(obj=self, name='AsyncPartitionedParameterSwapper', exclude_list=exclude_list)

    def available_swap_in_buffers(self):
        return len(self.available_buffer_ids)

    # 这个应该不用动，只要提前把参数设置好就行，最重要的应该是我们只用1个buffer，不用创建多个

    # 1.构建nvme路径并创建新的文件夹
    # 2.根据配置信息计算每个buffer对齐后的元素数量
    # 3.为所有Buffer创建一个连续的存储空间，即一个一维空tensor，位于pinned memory。大小为buffer对齐后的元素数量 × buffer数量
    # 4.创建两个aio_handle(C++)类，用于读和写
    # 📌备注：什么是buffer：buffer pool中有多个buffer，里面装的param准备卸载到NVMe.
    # self.buffer就是一个固定在内存的大tensor，大小为buffer pool中所有buffer的大小之和
    def _configure_aio(self, ds_config):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split(".")[1] # torch.float32
        # 构建nvme路径
        self.swap_folder = os.path.join(self.swap_config.nvme_path, 'zero_stage_3', f'{torch_dtype_string}params',
                                        f'rank{dist.get_rank()}')
        
        # 清理已存在的临时文件夹，并创建新的文件夹
        shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        # .element_size(): 获取张量中(一个)元素的字节大小
        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        # 若没在 "ds_config" 中设置的话，aio_config中存的都是默认值
        self.aio_config = ds_config.aio_config

        # Read/Write alignment for each thread during Intra-request parallelism
        # 
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE]) # 1024**2, 1048576 (都是1MB，相等)
        # 异步I/O的对齐字节数
        # 对齐字节数 × I/O线程数量
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_THREAD_COUNT] # 1024 × 1
        # 对齐的元素数量 = 对齐的字节数/tensor中一个元素的字节数
        # 即一个对齐的块能装多少元素
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # Size of buffers in buffer pool for parameter offloading to NVMe. Default 1e8
        self.elements_per_buffer = self.swap_config.buffer_size
        # 若buffer中元素数量能被 对齐元素数量 整模，返回元素数量；不然返回对齐后的元素数量（向上对齐）
        self.aligned_elements_per_buffer = self._io_aligned_numel(self.elements_per_buffer)
        # Number of buffers in buffer pool for parameter offloading to NVMe. Default 5
        self.param_buffer_count = self.swap_config.buffer_count

        self.available_buffer_ids = [i for i in range(self.param_buffer_count)]
        self.reserved_buffer_ids = []
        # 创建一个一维的空tensor，锁定在内存中
        # 为所有Buffer创建一个连续的存储空间，位于pinned memory：buffer对齐后的元素数量 × buffer个数
        self.buffers = get_accelerator().pin_memory(torch.empty(int(self.aligned_elements_per_buffer *
                                                                    self.param_buffer_count),
                                                                dtype=self.dtype,
                                                                requires_grad=False),
                                                    align_bytes=0)

        # 创建两个aio_handle(C++)类，用于读和写
        self.aio_read_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                               self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS],
                                               self.aio_config[AIO_THREAD_COUNT])

        self.aio_write_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                                self.aio_config[AIO_SINGLE_SUBMIT],
                                                self.aio_config[AIO_OVERLAP_EVENTS], self.aio_config[AIO_THREAD_COUNT])

        self.swap_out_params = []

    # 📍
    def _configure_aio_2(self, ds_config, pinned_buffer_information, nvme_layers):
        self.swap_config = ds_config.zero_config.offload_param
        torch_dtype_string = str(self.dtype).split(".")[1] # torch.float32
        # 构建nvme路径
        self.swap_folder = os.path.join(self.swap_config.nvme_path, 'zero_stage_3', f'{torch_dtype_string}params',
                                        f'rank{self.rank}')
        print(f"rank:{self.rank}, 当前rank的卸载路径为:{self.swap_folder}")
        # 清理已存在的临时文件夹，并创建新的文件夹
        shutil.rmtree(self.swap_folder, ignore_errors=True)
        os.makedirs(self.swap_folder, exist_ok=True)

        # .element_size(): 获取张量中(一个)元素的字节大小
        self.swap_element_size = torch.tensor([], dtype=self.dtype).element_size()

        # 若没在 "ds_config" 中设置的话，aio_config中存的都是默认值
        self.aio_config = ds_config.aio_config
        print(f"rank:{self.rank}, aioconfig:{self.aio_config}")

        # Read/Write alignment for each thread during Intra-request parallelism
        # 
        self.min_aio_bytes = max(MIN_AIO_BYTES, self.aio_config[AIO_BLOCK_SIZE]) # 1024**2, 1048576 (都是1MB，相等)
        # 异步I/O的对齐字节数
        # 对齐字节数 × I/O线程数量
        self.aligned_bytes = AIO_ALIGNED_BYTES * self.aio_config[AIO_THREAD_COUNT] # 1024 × 1
        # 对齐的元素数量 = 对齐的字节数/tensor中一个元素的字节数
        # 即一个对齐的块能装多少元素
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # 📍直接拿到计算出来的buffers的大小，替代手动指定buffer大小
        self.elements_per_buffer = pinned_buffer_information.get_buffers_size()

        # 📍计算出来的buffer大小已经包含了补齐的部分，这里不需要再次补齐了
        self.aligned_elements_per_buffer = self.elements_per_buffer

        # Number of buffers in buffer pool for parameter offloading to NVMe. Default 5
        self.param_buffer_count = self.swap_config.buffer_count

        # 创建一个一维的空tensor，锁定在内存中
        # 📍上面已经返回算好的值了，这里不用再 × buffer的数量了
        self.buffers = get_accelerator().pin_memory(torch.empty(int(self.aligned_elements_per_buffer),
                                                                dtype=self.dtype,
                                                                requires_grad=False),
                                                    align_bytes=0)

        self.pinned_buffer_information = pinned_buffer_information

        self.layer_count = self.pinned_buffer_information.layer_count

        self.nvme_layers = nvme_layers
        self.layer_id_param_idx_to_path = {}
        for layer_id in self.nvme_layers:
            self.layer_id_param_idx_to_path[layer_id] = {}

        # 📍2
        self.layer_id_to_path = {}

        # 创建两个aio_handle(C++)类，用于读和写
        self.aio_read_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                               self.aio_config[AIO_SINGLE_SUBMIT], self.aio_config[AIO_OVERLAP_EVENTS],
                                               self.aio_config[AIO_THREAD_COUNT], self.rank)

        self.aio_write_handle = self.aio_handle(self.aio_config[AIO_BLOCK_SIZE], self.aio_config[AIO_QUEUE_DEPTH],
                                                self.aio_config[AIO_SINGLE_SUBMIT],
                                                self.aio_config[AIO_OVERLAP_EVENTS], self.aio_config[AIO_THREAD_COUNT], self.rank)

        print(f"当前nvme配置为: block_size:{self.aio_config[AIO_BLOCK_SIZE]}, queue_depth：{self.aio_config[AIO_QUEUE_DEPTH]}, single_submit:{self.aio_config[AIO_SINGLE_SUBMIT]}, overlap:{self.aio_config[AIO_OVERLAP_EVENTS]}, thread:{self.aio_config[AIO_THREAD_COUNT]}")
        # exit(0)
        self.swap_out_params = []

    #Check if partitioned param or numel in a tensor is swappable or not
    # 若传入的参数是numel，计算总的字节数，若总字节数大于等于初始化时写死的最少I/O字节数(默认1MB)，返回true
    def swappable_tensor(self, param=None, numel=None):
        if param is not None:
            assert numel is None, "Both parma and numel cannot be provided"
            numel = param.ds_tensor.ds_numel
        # 检查规定的最小字节数量是否小于或等于总的字节数
        # 意思就是tensor必须大到一定程度（默认1MB）才是可换出的
        if numel is not None:
            # 初始化时写死的最少I/O字节数 <= 元素数量×一个元素占几个字节
            return self.min_aio_bytes <= numel * self.swap_element_size
        assert False, "Either param or numel must be provided"

    def get_path(self, param, must_exist=False):
        paths = self._get_swap_paths([param], must_exist=must_exist)
        return paths[0]

    # 得到params对应的path并返回。若当前param的路径不存在，创建一个
    def _get_swap_paths(self, params, must_exist=False):
        paths = []
        for param in params:
            param_id = param.ds_id
            if param_id in self.id_to_path.keys():
                param_path = self.id_to_path[param_id]
            else:
                assert not must_exist, f"Path for param id {param_id} does not exist"
                # 若当前param的路径不存在，创建一个
                # os.path.join: 接受任意数量的字符串参数，将它们连接成一个有效的文件路径
                param_path = os.path.join(self.swap_folder, f'{param_id}_param.tensor.swp')

                self.id_to_path[param_id] = param_path
            paths.append(param_path)

        return paths

    # 📍通过layer_id, layer idx，param_idx来获取path
    def _get_swap_paths_2(self, layer_id, param_idx, layer_idx=None, must_exist=False):
        owner_rank = self.layer_id_to_rank[layer_id]

        if param_idx in self.layer_id_param_idx_to_path[layer_id].keys():
            param_path = self.layer_id_param_idx_to_path[layer_id][param_idx]
        else:
            assert not must_exist, f"Path for layer id {layer_id} param idx {param_idx} does not exist"
            # param_path = os.path.join(self.swap_folder, f'{layer_id}_{param_idx}_param.tensor.swp')
            param_path = os.path.join(
                self.swap_config.nvme_path,
                'zero_stage_3',
                f'{str(self.dtype).split(".")[1]}params',
                f'rank{owner_rank}',  # 使用拥有这个layer的rank的目录
                f'{layer_id}_{param_idx}_param.tensor.swp'
            )

            self.layer_id_param_idx_to_path[layer_id][param_idx] = param_path
        return param_path

    # 📍2
    def _get_transformer_layer_swap_paths(self, layer_id, layer_idx=None, must_exist=False):
        owner_rank = self.layer_id_to_rank[layer_id]

        if layer_id in self.layer_id_to_path.keys():
            param_path = self.layer_id_to_path[layer_id]
        else:
            assert not must_exist, f"Path for layer id {layer_id} does not exist"
            # param_path = os.path.join(self.swap_folder, f'{layer_id}_{param_idx}_param.tensor.swp')
            param_path = os.path.join(
                self.swap_config.nvme_path,
                'zero_stage_3',
                f'{str(self.dtype).split(".")[1]}params',
                f'rank{owner_rank}',  # 使用拥有这个layer的rank的目录
                f'layer_{layer_id}.tensor.swp'
            )

            self.layer_id_to_path[layer_id] = param_path
        return param_path

    # 得到params对应的buffer(一个大tensor(就是所谓的buffer)的narrow后的结果)并返回
    def _get_swap_buffers(self, params):
        buffers = []
        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_swap_buffer.keys(), \
            f'param {param_id} has not been assigned a swap buffer'
            buffers.append(self.param_id_to_swap_buffer[param_id])

        return buffers

    # 对param list中的每个param，建立param_id到其partition后参数量的映射
    def _track_numel(self, params):
        for param in params:
            assert param.ds_tensor is not None, "Partitioned tensor is None"
            self.param_id_to_numel[param.ds_id] = param.ds_tensor.ds_numel

    # 1.从可用的buffer_id中弹出一个id，建立当前param id到该buffer_id的映射关系
    # 2.使用narrow方法返回buffer中对应该param的部分(长度为对齐后的元素数量)，即swap_buffer，并建立当前param id到该buffer的映射关系
    #   将该buffer加入到swap_buffers列表中
    # 3.进一步从swap_buffer中narrow出当前param参数量的部分，得到compute_buffer（swap_buffer可能含有补齐的部分）
    #   并将该buffer加入到compute_buffers列表中
    # 返回两个列表：
    # 第一个列表装着每个param对应的compute_buffer(长度仅为当前param的参数量)，
    # 第二个列表装着每一个对应的swap_buffer（可能包含补齐的部分）
    def _allocate_and_return_buffers_for_swap_in(self, params):
        compute_buffers = []
        swap_buffers = []

        for param in params:
            param_id = param.ds_id
            assert param_id in self.param_id_to_numel.keys(), f" Number of elements in param {param_id} is unknown"
            assert param_id not in self.param_id_to_buffer_id.keys(
            ), f"param {param_id} already assigned swap buffer id {self.param_id_to_buffer_id[param_id]}"
            assert param_id not in self.param_id_to_swap_buffer.keys(
            ), f"param {param_id} has already been assigned a swap buffer"

            # 1.从可用的buffer_id中弹出一个id，建立当前param id到该buffer_id的映射关系
            buffer_id = self.available_buffer_ids.pop()
            print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}  ")
            self.param_id_to_buffer_id[param_id] = buffer_id
            # 得到参数对齐后的元素数量
            aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
            # 2.返回buffer中对应该param的部分，即swap_buffer，并建立当前param到该buffer的映射关系
            #   将该buffer加入到swap_buffers列表中
            swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

            self.param_id_to_swap_buffer[param_id] = swap_buffer
            # 3.进一步从swap_buffer中narrow出当前param参数量的部分，得到compute_buffer（swap_buffer可能含有补齐的部分）
            #   并将该buffer加入到compute_buffers列表中
            compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            compute_buffers.append(compute_buffer)
            swap_buffers.append(swap_buffer)

        return compute_buffers, swap_buffers

    # 📍直接根据transformer layer和内部param的下标获取在buffers中的范围，作为接收tensor的空间
    def _allocate_and_return_buffers_for_swap_in_2(self, layer_idx, param_idx):

        # 不需要弹出什么buffer id，直接拿到param在buffer中的偏移量
        layer_start = self.pinned_buffer_information.get_layer_start_pos_in_buffers(layer_idx)
        layer_size = self.pinned_buffer_information.get_buffer_size()
        param_start = self.pinned_buffer_information.get_param_start_pos(param_idx)
        param_size = self.pinned_buffer_information.get_param_size(param_idx)
        buffer = self.buffers.narrow(0, layer_start, layer_size)
        compute_buffer = buffer.narrow(0, param_start, param_size)

        aligned_size = self.pinned_buffer_information.get_param_aligned_size(param_idx)
        swap_buffer = buffer.narrow(0, param_start, aligned_size)

        return compute_buffer, swap_buffer
    
    # 📍因为直接卸载读取整个transformer layer，所以不需要补齐param，直接按param原始
    #   的大小获取就好了
    def _allocate_and_return_buffers_for_param(self, layer_idx, param_idx):
        layer_start = self.pinned_buffer_information.get_layer_start_pos_in_buffers(layer_idx)
        layer_size = self.pinned_buffer_information.get_buffer_size()
        param_start = self.pinned_buffer_information.get_param_start_pos(param_idx)
        param_size = self.pinned_buffer_information.get_param_size(param_idx)
        buffer = self.buffers.narrow(0, layer_start, layer_size)
        compute_buffer = buffer.narrow(0, param_start, param_size)

        return compute_buffer


    # 📍3.跟上面一个版本的
    def _return_buffer_for_transformer_layer(self, layer_idx):
        layer_start = self.pinned_buffer_information.get_layer_start_pos_in_buffers(layer_idx)
        aligned_layer_size = self.pinned_buffer_information.get_buffer_size()
        layer_size = self.pinned_buffer_information.get_layer_size()

        swap_buffer = self.buffers.narrow(0, layer_start, aligned_layer_size)

        compute_buffer = swap_buffer.narrow(0, 0, layer_size)

        return compute_buffer, swap_buffer

    # 📍因为直接卸载读取整个transformer layer，所以不需要补齐param，直接按param原始
    #   的大小获取就好了
    def _allocate_and_return_buffers_for_param_double_buffer(self, buffer_id, layer_idx, param_idx):
        layer_start = self.pinned_buffer_information.get_layer_start_pos_in_buffers(buffer_id, layer_idx)
        layer_size = self.pinned_buffer_information.get_layer_size()
        param_start = self.pinned_buffer_information.get_param_start_pos(param_idx)
        param_size = self.pinned_buffer_information.get_param_size(param_idx)
        buffer = self.buffers.narrow(0, layer_start, layer_size)
        compute_buffer = buffer.narrow(0, param_start, param_size)

        return compute_buffer
    
    # 📍4.双buffer版本
    def _return_buffer_for_transformer_layer_double_buffer(self, buffer_id, layer_idx):
        layer_start = self.pinned_buffer_information.get_layer_start_pos_in_buffers(buffer_id, layer_idx)
        aligned_layer_size = self.pinned_buffer_information.get_aligned_layer_size()
        layer_size = self.pinned_buffer_information.get_layer_size()
        swap_buffer = self.buffers.narrow(0, layer_start, aligned_layer_size)
        compute_buffer = swap_buffer.narrow(0, 0, layer_size)

        return compute_buffer, swap_buffer

    #waits for inflight nvme write to complete
    # 1.等在正在进行的写操作全部完成
    # 2.
    # 2.1.将当前参数所在的buffer的 id 放回可用buffer列表
    # 2.2.删除该param_id的映射关系(即字典)
    # 2.3.若当前参数在可用参数集合中，将其移除；同时更新参数量(减去移除的tensor的参数量)
    # 2.4.将param.ds_tensor.data指向一个无效的tensor的数据(该tensor只有一个1)，同时更新param.ds_tensor的状态为 NOT_AVAILABLE
    def synchronize_writes(self):
        if self.pending_writes == 0:
            return
        # 等在正在进行的写操作全部完成
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        # 1.将当前参数所在的buffer的 id 放回可用buffer列表
        # 2.删除该param_id的映射关系(即字典)
        # 3.若当前参数在可用参数集合中，将其移除；同时更新参数量(减去移除的tensor的参数量)
        # 4.将param.ds_tensor.data指向一个无效的tensor的数据(该tensor只有一个1)，同时更新param.ds_tensor的状态为 NOT_AVAILABLE
        self.remove_partition_and_release_buffers(self.swap_out_params)
        self.swap_out_params = []

    # 📍因为是从shared memory直接卸载的，因此根本无需回收pinned buffer
    #    释放参数则交给上级类处理，不在本类中处理
    def synchronize_writes_without_release(self):
        if self.pending_writes == 0:
            return
        # 等在正在进行的写操作全部完成
        assert self.pending_writes == self.aio_write_handle.wait()
        self.pending_writes = 0
        # self.swap_out_params = []

    #waits for inflight nvme reads to complete
    # 1.等待正在从nvme读数据的线程全部完成
    # 2.将上面刚刚完成读取的那些param的 param.ds_tensor.data 指向对应的 buffer(刚刚读到buffer里了)。
    #   并将 param.ds_tensor.status 设置为 AVAILABLE
    # 3.将上面刚执行完的参数的ds_id添加到可用参数集合中(📌可用说的是ds_tensor可用了)。清空追踪的变量（追踪正在读取的参数的相关信息）
    def synchronize_reads(self):
        # 若没有正在读取的参数，直接返回
        if self.pending_reads == 0:
            return

        assert self.pending_reads == self.aio_read_handle.wait()

        # 清空正在执行的读取的参数数量
        self.pending_reads = 0

        # 将上面刚刚完成读取的那些param的 param.ds_tensor.data 指向对应的 buffer （刚刚读到buffer里了）
        # 状态设置为 AVAILABLE
        for param, swap_in_buffer in zip(self.inflight_params, self.inflight_swap_in_buffers):
            param_id = param.ds_id
            compute_buffer = swap_in_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
            param.ds_tensor.data = compute_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE

        # 将上面刚执行完的参数的ds_id添加到可用参数集合中
        self.available_params.update([param.ds_id for param in self.inflight_params])
        self.available_numel += self.inflight_numel

        # 清空正在追踪的要读取的param的信息
        self.inflight_params = []
        self.inflight_swap_in_buffers = []
        self.inflight_numel = 0

    def synchronize_reads_2(self):
        if self.pending_reads == 0:
            return
        assert self.pending_reads == self.aio_read_handle.wait()
        self.pending_reads = 0

    #Removes the memory assignment and releases the buffers
    #Should only be executed after swapping out the tensors
    # 1.若当前参数在param_id_to_buffer_id的映射中，则：
    #   1.1.将当前参数所在的buffer的 id 放回可用buffer列表
    #   1.2.删除该param_id到buffer_id的映射关系，删除param_id到swap_buffer的映射关系
    #   1.3.若当前参数在可用参数集合中(说明在cpu buffer上存着)，将其移除；同时更新可用参数的参数量(减去移除的tensor的参数量)
    # 2.将param.ds_tensor.data指向一个无效的tensor的数据(该tensor只有一个1)，同时更新param.ds_tensor的状态为 NOT_AVAILABLE
    def remove_partition_and_release_buffers(self, params):
        for param in params:
            param_id = param.ds_id

            if param_id in self.param_id_to_buffer_id.keys():

                buffer_id = self.param_id_to_buffer_id[param_id]

                assert buffer_id is not None, "Missing buffer id for releasing"

                # 将释放的缓冲区 id 放回可用列表
                # ❓这是啥逻辑
                self.available_buffer_ids.append(buffer_id)
                # 删除该param_id的映射关系(即字典)
                del self.param_id_to_buffer_id[param_id]
                del self.param_id_to_swap_buffer[param_id]
                print_rank_0(f"param {param.ds_id} releases buffer id {buffer_id}  ")

                # 若当前参数在可用参数集合中，将其移除
                # 同时更新参数量(减去移除的tensor的参数量)
                if param_id in self.available_params:
                    self.available_params.remove(param_id)
                    self.available_numel -= self.param_id_to_numel[param_id]

            # 将参数的数据指向一个无效的tensor的数据(该tensor只有一个1)，同时更新状态
            param.ds_tensor.data = self.invalid_buffer.data
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE

    #writes from in memory to nvme. Does not release the buffers
    # 
    def _swap_out(self, params, async_op=True):

        swap_out_paths = self._get_swap_paths(params) # 得到参数对应的path并返回。若当前param的路径不存在，创建一个
        # ❓param的底层数据何时装到这个buffer里的？
        # 答:见partitioned_parameter.py中的_partition_param的第一个📌，param.ds_tensor已经和buffer共享底层存储了
        swap_out_params = self._get_swap_buffers(params) # 得到params对应的buffer(一个tensor narrow后的结果)并返回
        self._track_numel(params) # 对param list中的每个param，建立param_id到其partition后参数量的映射

        # 异步写入，给线程分配完任务就直接返回继续往下运行了
        swap_out_tensors(self.aio_write_handle, swap_out_params, swap_out_paths)

        self.pending_writes += len(swap_out_params)
        self.swap_out_params += params

        # partition_param中未传入async_op参数，默认使用swap_out_and_release传入的false，这里执行同步写入
        if not async_op:
            # 1.等待正在进行的写操作全部完成
            # 2.
            # 2.1.将当前参数所在的buffer的 id 放回可用buffer列表
            # 2.2.删除该param_id的映射关系(即字典)
            # 2.3.若当前参数在可用参数集合中，将其移除；同时更新参数量(减去移除的tensor的参数量)
            # 2.4.将param.ds_tensor.data指向一个无效的tensor的数据(该tensor只有一个1)，同时更新param.ds_tensor的状态为
            #     NOT_AVAILABLE
            self.synchronize_writes()

    # 📍新的swap_out方法，直接卸载给定的tensor，不需要从pinned buffer卸载
    def swap_out_2(self, param, layer_id, param_idx, async_op=False):
        swap_out_path = self._get_swap_paths_2(layer_id, param_idx)
        print(f"rank:{self.rank}, 准备开始卸载layer{layer_id}", flush=True)
        swap_out_tensors(self.aio_write_handle, [param], [swap_out_path])
        self.pending_writes += 1
        # self.swap_out_params += param
        if not async_op:
            print(f"rank:{self.rank}, layer{layer_id} 开始sync写入nvme", flush=True)
            self.synchronize_writes_without_release()
            print(f"rank:{self.rank}, layer{layer_id}-{param_idx}卸载完成", flush=True)

    def swap_out_2_sync(self, param, layer_id, param_idx):
        swap_out_path = self._get_swap_paths_2(layer_id, param_idx)
        swap_out_tensors_sync(self.aio_write_handle, [param], [swap_out_path])
        print(f"rank:{self.rank}, layer{layer_id}-{param_idx}卸载完成", flush=True)
        
    def swap_out_2_sync_2(self, param, layer_id, param_idx):
        swap_out_path = self._get_swap_paths_2(layer_id, param_idx)
        swap_out_tensors_sync_2(self.aio_write_handle, [param], [swap_out_path])
        print(f"rank:{self.rank}, layer{layer_id}-{param_idx}卸载完成", flush=True)

    # 📍
    def swap_out_transformer_layer(self, layer_id, layer_idx, async_op=False):
        # start_time = time.perf_counter()
        swap_out_path = self._get_transformer_layer_swap_paths(layer_id)
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}, 返回地址时间: {(end_time - start_time):.4f} 秒", flush=True)
        # start_time = time.perf_counter()
        _, swap_buffer = self._return_buffer_for_transformer_layer(layer_idx)
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}, 分配buffer时间: {(end_time - start_time):.4f} 秒", flush=True)
        # print(f"rank:{self.rank}, 准备开始卸载layer{layer_id}", flush=True)
        # start_time = time.perf_counter()
        swap_out_tensors(self.aio_write_handle, [swap_buffer], [swap_out_path])
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}, 调用swap_out_tensors函数时间: {(end_time - start_time):.4f} 秒", flush=True)
        self.pending_writes += 1
        # 📌瓶颈在这，尼玛的占用时间巨长，应该是+=触发了tensor的深度拷贝
        # start_time = time.perf_counter()
        # self.swap_out_params += swap_buffer
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}, \"self.swap_out_params += swap_buffer\"时间: {(end_time - start_time):.4f} 秒", flush=True)
        if not async_op:
            # start_time = time.perf_counter()
            print(f"rank:{self.rank}, layer{layer_id} 开始sync写入nvme", flush=True)
            self.synchronize_writes_without_release()
            # end_time = time.perf_counter()
            # print(f"rank:{self.rank}，等待卸载的时间: {(end_time - start_time):.4f} 秒", flush=True)
            print(f"rank:{self.rank}, layer{layer_id}卸载完成", flush=True)

    # 📍
    def swap_out_transformer_layer_sync(self, layer_id, layer_idx, async_op=False):
        # start_time = time.perf_counter()
        swap_out_path = self._get_transformer_layer_swap_paths(layer_id)
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}, 返回地址时间: {(end_time - start_time):.4f} 秒", flush=True)
        # start_time = time.perf_counter()
        _, swap_buffer = self._return_buffer_for_transformer_layer(layer_idx)
        swap_in_tensors_sync_2(self.aio_write_handle, [swap_buffer], [swap_out_path])
        print(f"rank:{self.rank}, layer{layer_id}卸载完成", flush=True)
        

    # 📍双buffer版本
    def swap_out_transformer_layer_double_buffer(self, buffer_id, layer_id, layer_idx, async_op=False):
        swap_out_path = self._get_transformer_layer_swap_paths(layer_id)
        _, swap_buffer = self._return_buffer_for_transformer_layer_double_buffer(buffer_id, layer_idx)
        swap_out_tensors(self.aio_write_handle, [swap_buffer], [swap_out_path])
        self.pending_writes += 1
        if not async_op:
            self.synchronize_writes_without_release()
            print(f"rank:{self.rank}, layer{layer_id}卸载完成", flush=True)

    # 📍双buffer版本
    def swap_out_transformer_layer_double_buffer_sync(self, buffer_id, layer_id, layer_idx):
        swap_out_path = self._get_transformer_layer_swap_paths(layer_id)
        _, swap_buffer = self._return_buffer_for_transformer_layer_double_buffer(buffer_id, layer_idx)
        swap_out_tensors_sync_2(self.aio_write_handle, [swap_buffer], [swap_out_path])

    #blocking swap out followed by releasing the memory buffers
    def swap_out_and_release(self, params, async_op=False, force_buffer_release=False):
        if async_op:
            assert force_buffer_release, "Should not release preallocated buffers without completing the swap out. Set force_buffer_release to True to do it anyways"
        self._swap_out(params, async_op=async_op)


    # book keeping function for inflight swap in
    # 将参数列表加到inflight_params中，表示这些参数正在读取
    # 将buffers加入到inflight_swap_in_buffers列表，同理
    # 将正在读取的参数量赋给self.inflight_numel
    # 
    # 将所有正在读取的划分后的param的状态标记为 INFLIGHT
    # 记录正在读取的参数数量
    def _update_inflight_swap_in(self, params, swap_in_buffers, inflight_numel):
        self.inflight_params.extend(params)
        self.inflight_swap_in_buffers.extend(swap_in_buffers)
        self.inflight_numel += inflight_numel

        for param in params:
            param.ds_tensor.status = PartitionedParamStatus.INFLIGHT

        self.pending_reads += len(params)

    #assigns an in memory buffer and swaps in from nvme
    # 1.首先要确保所有的划分后的param都处于不可用状态
    # 2.若没有给定的buffer，则建立缓冲区用于接收nvme的参数
    #   -为每一个param分配一个buffer，用于接受nvme读上来的参数
    # 3.调用C++函数把根据把路径把param都拿到buffer中
    # 4.对正在读取的param和buffer进行追踪（就是赋给一些self成员变量）,并将所有正在读取的param的状态标记为 INFLIGHT
    #
    # 若并非异步执行，即必须等nvme读操作完成：
    # 5.
    #   5.1.等待正在从nvme读数据的线程全部完成
    #   5.2.将上面刚刚完成读取的那些param的 param.ds_tensor.data 指向对应的 buffer(刚刚读到buffer里了)。
    #       并将 param.ds_tensor.status 设置为 AVAILABLE
    #   5.3.将上面刚执行完的参数的ds_id添加到可用参数集合中(📌可用说的是ds_tensor可用了)。
    #       清空追踪的变量（追踪正在读取的参数的相关信息）
    def swap_in(self, params, async_op=True, swap_in_buffers=None):

        # 1.首先要确保所有的param都处于不可用状态
        assert all([param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE
                    for param in params]), "Some params are already available or in flight"
        swap_in_paths = self._get_swap_paths(params)

        # 2.若没有给定的buffer，则建立缓冲区用于接收nvme的参数
        # 必须保证可用buffer数量，比要读的参数多，不然直接中断
        if swap_in_buffers is None:
            if len(self.available_buffer_ids) < len(swap_in_paths):
                ids = [p.ds_id for p in params]
                print_rank_0(
                    f'Not enough swap in buffers {len(self.available_buffer_ids)} for {len(swap_in_paths)} params, ids = {ids}',
                    force=True)
                print_rank_0(
                    f'Num inflight: params {len(self.inflight_params)}, buffers {len(self.inflight_swap_in_buffers)}, numel = {self.inflight_numel}',
                    force=True)
                print_rank_0(
                    f'Num available params: count = {len(self.available_params)}, ids = {self.available_params}, numel = {self.available_numel}',
                    force=True)

            assert len(swap_in_paths) <= len(
                self.available_buffer_ids
            ), f"Not enough buffers {len(self.available_buffer_ids)} for swapping {len(swap_in_paths)}"
            # 3.为每一个param分配一个buffer，用于接受nvme读上来的参数
            # 3.1.从可用的buffer_id中弹出一个id，建立当前param id到该buffer_id的映射关系
            # 3.2.使用narrow方法返回buffer中对应该param的部分(长度为对齐后的元素数量)，即swap_buffer，并建立当前param id到该buffer的映射关系
            #     将该buffer加入到swap_buffers列表中
            # 3.3.进一步从swap_buffer中narrow出当前param参数量的部分，得到compute_buffer（swap_buffer可能含有补齐的部分）
            #     并将该buffer加入到compute_buffers列表中
            # 返回两个列表：
            # 第一个列表装着每个param对应的compute_buffer(长度仅为当前param的参数量)，
            # 第二个列表装着每一个对应的swap_buffer（可能包含补齐的部分）
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in(params)
            # 计算返回的buffer能装的参数量
            inflight_numel = sum([t.numel() for t in compute_buffers])
        else:
            inflight_numel = sum([t.numel() for t in swap_in_buffers])

        # 3.调用C++函数把根据把路径把param都拿到buffer中
        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)

        # 4.对正在读取的param和buffer进行追踪（就是赋给一些self成员变量）,并将所有正在读取的param的状态标记为 INFLIGHT
        # 将参数列表加到inflight_params中，表示这些参数正在读取
        # 将buffers加入到inflight_swap_in_buffers列表，同理
        # 将正在读取的参数量赋给self.inflight_numel
        # 
        # 将所有正在读取的划分后的param的状态标记为 INFLIGHT
        # 记录正在读取的参数数量
        self._update_inflight_swap_in(params, swap_in_buffers, inflight_numel)

        if not async_op:
            # 1.等待正在从nvme读数据的线程全部完成
            # 2.将上面刚刚完成读取的那些param的 param.ds_tensor.data 指向对应的 buffer(刚刚读到buffer里了)。
            #   并将 param.ds_tensor.status 设置为 AVAILABLE
            # 3.将上面刚执行完的参数的ds_id添加到可用参数集合中(可用说的是ds_tensor可用了)。
            #   清空追踪的变量（追踪正在读取的参数的相关信息）
            self.synchronize_reads()

    # 📍将nvme的数据读取到pinned buffer上，准备送给GPU
    def swap_in_2(self, layer_id, layer_idx, param_idx, async_op=True):
        print(f"rank:{self.rank} 准备获取layer{layer_id}的文件路径", flush=True)
        swap_in_path = self._get_swap_paths_2(layer_id, param_idx)
        print(f"rank:{self.rank} 想要读取的layer{layer_id}的文件路径为:{swap_in_path}", flush=True)

        compute_buffer, swap_in_buffer = self._allocate_and_return_buffers_for_swap_in_2(layer_idx, param_idx)
        print(f"rank:{self.rank}, pinned buffer分配完成", flush=True)
        # 计算返回的buffer能装的参数量
        inflight_numel = compute_buffer.numel()

        swap_in_tensors(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        self.pending_reads += 1

        #
        if not async_op:
            print(f"rank:{self.rank}, 开始sync读取", flush=True)
            self.synchronize_reads_2()
            print(f"rank:{self.rank}, sync结束", flush=True)

    # 同步读取，使用sync_pread方法，有线程
    def swap_in_2_sync(self, layer_id, layer_idx, param_idx):
        swap_in_path = self._get_swap_paths_2(layer_id, param_idx)
        compute_buffer, swap_in_buffer = self._allocate_and_return_buffers_for_swap_in_2(layer_idx, param_idx)
        swap_in_tensors_sync(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        print(f"rank:{self.rank}, 同步读取{layer_id}-{param_idx}完成", flush=True)

    # 真正的同步读取，使用read方法，没有线程
    def swap_in_2_sync_2(self, layer_id, layer_idx, param_idx):
        swap_in_path = self._get_swap_paths_2(layer_id, param_idx)
        compute_buffer, swap_in_buffer = self._allocate_and_return_buffers_for_swap_in_2(layer_idx, param_idx)
        swap_in_tensors_sync_2(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        print(f"rank:{self.rank}, 同步读取{layer_id}-{param_idx}完成", flush=True)

    # 📍2
    def swap_in_transformer_layer(self, layer_id, layer_idx, async_op=True):
        print(f"rank:{self.rank} 准备获取layer{layer_id}的文件路径", flush=True)
        swap_in_path = self._get_transformer_layer_swap_paths(layer_id)
        print(f"rank:{self.rank} 想要读取的layer{layer_id}的文件路径为:{swap_in_path}", flush=True)

        compute_buffer, swap_in_buffer = self._return_buffer_for_transformer_layer(layer_idx)
        print(f"rank:{self.rank}, pinned buffer分配完成", flush=True)
        # 计算返回的buffer能装的参数量
        inflight_numel = compute_buffer.numel()

        swap_in_tensors(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        self.pending_reads += 1

        #
        if not async_op:
            print(f"rank:{self.rank}, 开始sync读取", flush=True)
            self.synchronize_reads_2()
            print(f"rank:{self.rank}, sync结束", flush=True)
    
    # 📍2.同步读取transformer layer，底层C++库不会放到线程中，而是直接执行
    def swap_in_transformer_layer_sync(self, layer_id, layer_idx):
        swap_in_path = self._get_transformer_layer_swap_paths(layer_id)
        compute_buffer, swap_in_buffer = self._return_buffer_for_transformer_layer(layer_idx)
        swap_in_tensors_sync_2(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        print(f"rank:{self.rank}, 同步读取{layer_id}完成", flush=True)

    # 📍3.双buffer版本
    def swap_in_transformer_layer_double_buffer(self, buffer_id, layer_id, layer_idx, async_op=True):
        swap_in_path = self._get_transformer_layer_swap_paths(layer_id)
        compute_buffer, swap_in_buffer = self._return_buffer_for_transformer_layer_double_buffer(buffer_id, layer_idx)
        swap_in_tensors(self.aio_read_handle, [swap_in_buffer], [swap_in_path])
        self.pending_reads += 1

        if not async_op:
            self.synchronize_reads_2()

    # 📍3.双buffer版本
    def swap_in_transformer_layer_double_buffer_sync(self, buffer_id, layer_id, layer_idx):
        swap_in_path = self._get_transformer_layer_swap_paths(layer_id)
        compute_buffer, swap_in_buffer = self._return_buffer_for_transformer_layer_double_buffer(buffer_id, layer_idx)
        swap_in_tensors_sync_2(self.aio_read_handle, [swap_in_buffer], [swap_in_path])

    # Enables swapping into buffer that is out the control of swapper. This is always synchronous
    def swap_into_buffer(self, param, dest_buffer):
        assert param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE, f"param {param.ds_id} is already available or inflight"

        require_swap_buffer = not (get_accelerator().is_pinned(dest_buffer)
                                   and self._is_io_aligned(dest_buffer.numel()))

        if require_swap_buffer:
            assert len(self.available_buffer_ids) > 0, f"No buffer available to swap param {param.ds_id}."
            compute_buffers, swap_in_buffers = self._allocate_and_return_buffers_for_swap_in([param])
            inflight_numel = compute_buffers[0].numel()
        else:
            swap_in_buffers = [dest_buffer]
            inflight_numel = dest_buffer.numel()

        swap_in_paths = self._get_swap_paths([param])

        swap_in_tensors(self.aio_read_handle, swap_in_buffers, swap_in_paths)
        self._update_inflight_swap_in([param], swap_in_buffers, inflight_numel)
        self.synchronize_reads()

        if require_swap_buffer:
            dest_buffer.data.copy_(param.ds_tensor.data)
            # Release swap buffer memory assignment. Note, this will mark the parameter not available.
            self.remove_partition_and_release_buffers([param])

    #assign a buffer to a param and return the buffer
    # 为当前param确定buffer
    # 1.建立param_id到numel的映射
    # 2.buffer_id = self.available_buffer_ids.pop()，而后建立param_id到buffer_id的映射
    # 3.将截取出的对应当前Param的buffer（就是一个大tensor的一部分，即一个narrow的结果）加入到param_id_to_swap_buffer的映射
    # 4.进一步截取出当前param需要的部分，而不是上一步截取出的一整个buffer。（上一步截取的buffer可能包含了补齐的部分）
    # 📌注意这个buffer的长度仅仅是参数划分后的长度
    def get_buffer(self, param, numel):
        param_id = param.ds_id

        assert self.available_swap_in_buffers(
        ) > 0, f"No swap buffers to allocate for fp16 param {param_id} of numel = {numel}"
        assert numel < self.elements_per_buffer, f"More elements {numel} than buffer size {self.elements_per_buffer}"

        self.param_id_to_numel[param_id] = numel
        # 列表最后一个元素出队
        buffer_id = self.available_buffer_ids.pop()
        # 建立当前param到buffer_id（即刚出队的buffer_id）的映射关系
        self.param_id_to_buffer_id[param_id] = buffer_id
        # 若当前param的元素数量能被 对齐元素数量 整模，返回元素数量；不然返回补齐后的元素数量
        aligned_swap_numel = self._io_aligned_numel(self.param_id_to_numel[param_id])
        # 从大buffer中截取出对用当前buffer_id的、对应当前param元素数量的部分（可能被补齐了）
        # narrow(dim, start, length)
        # - dim: 要切片的维度
        # - start: 切片的起始索引
        # - length: 切片的长度
        swap_buffer = self.buffers.narrow(0, int(buffer_id * self.aligned_elements_per_buffer), aligned_swap_numel)

        # 将截取出的对应当前Param的buffer（可见就是一个大tensor的一部分，即一个narrow的结果）加入到param_id_to_swap_buffer的映射
        self.param_id_to_swap_buffer[param_id] = swap_buffer
        # 进一步截取出当前param需要的部分，而不是这一整个buffer
        compute_buffer = swap_buffer.narrow(0, 0, self.param_id_to_numel[param_id])
        print_rank_0(f"param {param.ds_id} is assigned swap in buffer id {buffer_id}")
        return compute_buffer

    # 📍
    def get_buffer(self, layer_idx, param_idx):
        start_pos = self.pinned_buffer_information.get_layer_start_pos_in_buffers(layer_idx)
        buffer_size = self.pinned_buffer_information.get_buffer_size()
        param_start_pos = self.pinned_buffer_information.get_param_start_pos(param_idx)
        param_size = self.pinned_buffer_information.get_param_size(param_idx)
        buffer = self.buffers.narrow(0, start_pos, buffer_size)
        compute_buffer = buffer.narrow(0, param_start_pos, param_size)
        return compute_buffer
    


    def reserve_available_buffers(self):
        buffers = []
        for id in self.available_buffer_ids:
            buffers.append(
                self.buffers.narrow(0, int(id * self.aligned_elements_per_buffer),
                                    int(self.aligned_elements_per_buffer)))
            self.reserved_buffer_ids.append(id)

        self.available_buffer_ids = []
        return buffers

    def release_reserved_buffers(self):
        for id in self.reserved_buffer_ids:
            self.available_buffer_ids.append(id)
        self.reserved_buffer_ids = []

    # numel：一个buffer的大小
    # 若元素数量能被 对齐元素数量 整模，返回元素数量；不然返回补齐后的元素数量
    def _io_aligned_numel(self, numel):
        # 元素数量 % 对齐元素数量
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def _is_io_aligned(self, numel):
        return (numel % self.numel_alignment) == 0

    # 1.计算所有param补齐后的 总参量
    # 2.在cpu上创建一个固定的连续的缓冲区，用于swap
    # 3.使用刚刚创建的缓冲区，实例化一个 SwapBufferPool 类，之后通过该类管理缓冲区
    def reserve_partitioned_swap_space(self, partition_num_elems):
        # 计算所有param补齐后的 总参量
        aligned_numel = sum([self._io_aligned_numel(numel) for numel in partition_num_elems])
        # 在cpu上创建一个固定的连续的缓冲区，用于swap
        self.partitioned_swap_buffer = get_accelerator().pin_memory(torch.zeros(aligned_numel,
                                                                                device='cpu',
                                                                                dtype=self.dtype),
                                                                    align_bytes=0)
        # 使用刚刚创建的缓冲区，实例化一个 SwapBufferPool 类，之后通过该类管理缓冲区
        self.partitioned_swap_pool = SwapBufferPool([self.partitioned_swap_buffer])

    # src_fp32_params：要写入nvme的param list
    # 把fp32 params写入到fp16 params对应的地址上（nvme）
    def swap_out_partitioned_params(self, dst_fp16_params, src_fp32_params):
        assert self.partitioned_swap_buffer is not None, f'partitioned swap buffers for fp16 params not initialized'
        assert self.partitioned_swap_pool is not None, f'partitioned swap pool for fp16 params not initialized'
        assert len(dst_fp16_params) == len(src_fp32_params), \
        f'mismatch in number of fp16 params {len(dst_fp16_params)} and fp32 params {len(src_fp32_params)}'

        # 得到fp16 params对应的path并返回。若当前param的路径不存在，创建一个
        fp16_swap_paths = self._get_swap_paths(dst_fp16_params, must_exist=True)
        self.synchronize_writes()
        # 对buffer pool中的每一个buffer，执行reset()，就是把所有的成员变量全部清空，当然buffer没删
        self.partitioned_swap_pool.reset()
        # 将 src_fp32_params 中的 tensor 都放进buffer pool中
        # 并将对应的fp16 param的ds_tensor的状态置为 AVAILABLE
        for i, fp32_tensor in enumerate(src_fp32_params):
            # 若buffer pool还能装下padding后的tensor（存在一个能装下的buffer）：
            # 1.根据给定的 参数量 和 padding后的参数量 分配buffer
            #   返回 buffer 和 不包含padding部分的 buffer (本质上是同一个buffer，只是长度不同)
            # 2.将给定的 tensor（fp32_tensor）复制到buffer中
            # 3.返回上面两个buffer
            # 传入的地址会被放入buffer的字典中存起来
            swap_tensor, _ = self.partitioned_swap_pool.insert_tensor(fp32_tensor, fp16_swap_paths[i],
                                                                      self._io_aligned_numel(fp32_tensor.numel()))
            assert swap_tensor is not None
            dst_fp16_params[i].ds_tensor.status = PartitionedParamStatus.AVAILABLE

        # 将buffer pool中装的tensor都卸载到nvme中
        # 1.返回所有使用中的buffer中保存的tensor（本质是buffer narrow出来的，即buffer的一部分）；
        #   返回所有使用中的buffer中保存的path
        # 2.使用swap_handle类，异步的将buffer写入到指定的地址
        # 3.若没设置异步操作，等待异步操作完成
        # 写入的地址在调用insert_tensor时已经存好了
        self.partitioned_swap_pool.swap_out(self.aio_write_handle)

        # 将fp16 param的ds_tensor的状态置为 NOT_AVAILABLE
        for param in dst_fp16_params:
            param.ds_tensor.status = PartitionedParamStatus.NOT_AVAILABLE
