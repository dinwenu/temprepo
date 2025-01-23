# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
import queue  # 添加这一行
from collections import OrderedDict as ODict

import torch

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from task_data_struct import Medium, vTask
import threadsafe_data_struct

import time

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

# 将cpu上本地模型的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
def convert_to_pinned(local_model_cpu):
    ''' in-place convert a local model cpu to a pinned model (params and buffers: pinned, local, CPU, no grad) '''
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # no grad
                assert param.grad is None, "convert to pinned model requires no grad in input model"
                param.detach_()
                assert not param.requires_grad
                # pin param
                # pin_memory() 是 PyTorch 中 Tensor 对象的一个方法，用于将张量固定在内存中的特定位置，
                # 例如钉在 GPU 内存或固定内存（pinned memory）中
                param.data = param.pin_memory() # in-place update and let python do the gc 
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                assert not buf.requires_grad # buffer has no grad
                m._buffers[key] = buf.pin_memory() # in-place update and let python do the gc 
                assert not m._buffers[key].requires_grad
    local_model_cpu.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    gc.collect()

class SharedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    # 将层的参数对应的优化器状态全部放入共享内存
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            # 确保当前这个层的参数和对应的优化器状态都不在GPU上
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            # 将参数的梯度初始化为与参数数据形状相同的全零张量
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            # 调用一次优化器的 step 方法，以强制初始化优化器状态
            # print(f"优化器初始化之前：{optimizer.state}")
            # 强制初始化之前，这个字典是空的
            # 另外需要注意，这个字典可能有多个键值对，即多个 参数：{step:,'exp_avg','exp_avg_sq'}
            optimizer.step() 
            # print(f"优化器初始化之后：{optimizer.state}")
            # 3) move optimzer.state to shared memory
            # 将优化器状态放入共享内存（甚至连step数也放进去了）
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    # 将一阶动量和二阶动量加入到共享内存
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # print("当前 Tensor 占用的内存大小:", memory_size_mb, "MB")
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    # 将 step 的值换成tensor值，也放入共享内存 
                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            # 将层的参数的梯度置为None，相当于清空梯度
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            # 手动触发垃圾回收
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    # 1.保险操作，确保逻辑正确（确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外其他的key的value不是tensor）
    # 2.深度拷贝model，其实就是一个层（self.shared_model）
    # 3.将复制的层的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 这个if就是个保险，确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外
        # 其他的key的value不是tensor
        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            # param_groups是一个字典列表，param_groups[0]是一个字典，其中包含模型的params以及一些优化器的参数，
            # 通过调整这些参数可以实现优化器对这一部分params更灵活的控制
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        # 该参数默认为false
        if self.no_pin_model:
            self.pinned_model = None
        # 该参数默认为false，执行else
        else:
            # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            # 深度拷贝model
            self.pinned_model = copy.deepcopy(self.shared_model)
            # 将cpu上本地model的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
            # for s_param, p_param in zip(self.shared_model.parameters(), self.pinned_model.parameters()):
            #     print("Values equal:", torch.all(s_param == p_param))
            #     print("Same object:", s_param.data_ptr() == p_param.data_ptr())
            #     print(p_param)

    # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
    # 模型(就是一层)的 buffer 中，即shared_model.named_buffers()
    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        # 若shared_model中（放在共享内存中的vlayer）有pinned buf这个属性，
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    # 将共享模型中的参数和缓冲区同步到本地的固定内存模型(在pinned memory中)中
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))






# 📍
class PinnedBufferInformation(object):
    
    def __init__(self, layer_count, transformer_layer, rank):
        self.layer_count = layer_count
        self.transformer_layer = transformer_layer
        self.rank = rank

        self.swap_element_size = torch.tensor([], dtype=torch.float32).element_size()
        self.aligned_bytes = 1024
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # for id in range(self.layer_count):
        #     self.pinned_model.append(PinnedModel())

        # 一个transformer层内的param到其参数量的映射
        self.param_idx_in_layer_to_numel = {}
        self.param_idx_in_layer_to_aligned_numel = {}
        self.param_idx_to_start_pos = {}

        self.buffer_size = self.cal_pinned_buffer_size()
        self.buffers_size = self.buffer_size * self.layer_count

        self.layer_idx_to_buffer_start_pos = {}

        self.buf_start_pos = 0

        for idx in range(self.layer_count):
            self.layer_idx_to_buffer_start_pos[idx] = idx * self.buffer_size


    # 统计需要多大的内存（包括补齐的部分）
    def cal_pinned_buffer_size(self):
        total_aligned_size = 0
        # 计算一个transformer层的参数大小（内部的每个param单独进行对齐）
        global_param_idx = 0
        for m in self.transformer_layer.modules():

            for id, (key, param) in enumerate(m._parameters.items()):
                param_size = param.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = param_size
                aligned_size = self._io_aligned_numel(param_size)
                self.param_idx_in_layer_to_aligned_numel[global_param_idx] = aligned_size
                self.param_idx_to_start_pos[global_param_idx] = total_aligned_size
                total_aligned_size += aligned_size
                global_param_idx += 1

            for key, buf in m._buffers.items():
                buf_size = buf.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = buf_size
                aligned_size = self._io_aligned_numel(buf_size)
                self.param_idx_in_layer_to_aligned_numel[global_param_idx] = aligned_size
                self.param_idx_to_start_pos[global_param_idx] = total_aligned_size
                total_aligned_size += aligned_size
                global_param_idx += 1

        print(f"rank:{self.rank}, 单层transformer模型需要的对齐字参数量:{total_aligned_size}, {total_aligned_size*4/1024/1024}MB")
        return total_aligned_size

    def get_buffer_size(self):
        return self.buffer_size

    def get_buffers_size(self):
        return self.buffers_size

    # 
    def _io_aligned_numel(self, numel):
        # 元素数量 % 对齐元素数量
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)
    
    def get_layer_start_pos_in_buffers(self, layer_idx):
        return self.layer_idx_to_buffer_start_pos[layer_idx]
    
    def get_param_start_pos(self, param_idx):
        return self.param_idx_to_start_pos[param_idx]
    
    def get_param_size(self, param_idx):
        return self.param_idx_in_layer_to_numel[param_idx]

    def get_param_aligned_size(self, param_idx):
        return self.param_idx_in_layer_to_aligned_numel[param_idx]
    

# 上面那一个是以sub layer为粒度的卸载，现在一次卸载整个transformer层
class PinnedBufferInformation_2(object):
    
    def __init__(self, layer_count, transformer_layer, rank):
        self.layer_count = layer_count
        self.transformer_layer = transformer_layer
        self.rank = rank

        self.swap_element_size = torch.tensor([], dtype=torch.float32).element_size()
        self.aligned_bytes = 1024
        self.numel_alignment = self.aligned_bytes // self.swap_element_size

        # 一个transformer层内的param到其参数量的映射
        self.param_idx_in_layer_to_numel = {}
        self.param_idx_to_start_pos = {}

        self.layer_size, self.buffer_size = self.cal_pinned_buffer_size()
        
        self.buffers_size = self.buffer_size * self.layer_count

        self.layer_idx_to_buffer_start_pos = {}

        self.buf_start_pos = 0

        for idx in range(self.layer_count):
            self.layer_idx_to_buffer_start_pos[idx] = idx * self.buffer_size


    # 统计需要多大的内存（包括补齐的部分）
    def cal_pinned_buffer_size(self):
        total_size = 0
        # 计算一个transformer层的参数大小
        global_param_idx = 0
        for m in self.transformer_layer.modules():

            for id, (key, param) in enumerate(m._parameters.items()):
                param_size = param.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = param_size

                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += param_size
                global_param_idx += 1

            for key, buf in m._buffers.items():
                buf_size = buf.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = buf_size
                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += buf_size
                global_param_idx += 1

        aligned_total_size = self._io_aligned_numel(total_size)
        print(f"rank:{self.rank}, 单层transformer模型原本的参数量{total_size} ,需要的对齐字参数量:{aligned_total_size}, {total_size*4/1024/1024}MB")
        
        return total_size, aligned_total_size

    def get_layer_size(self):
        return self.layer_size

    def get_buffer_size(self):
        return self.buffer_size

    def get_buffers_size(self):
        return self.buffers_size

    # 
    def _io_aligned_numel(self, numel):
        # 元素数量 % 对齐元素数量
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)
    
    def get_layer_start_pos_in_buffers(self, layer_idx):
        return self.layer_idx_to_buffer_start_pos[layer_idx]
    
    def get_param_start_pos(self, param_idx):
        return self.param_idx_to_start_pos[param_idx]
    
    def get_param_size(self, param_idx):
        return self.param_idx_in_layer_to_numel[param_idx]

# 双buffer，一个buffer卸载时，另一个buffer可以读取
class PinnedBufferInformation_double_buffer(object):
    def __init__(self, layer_count, transformer_layer, rank, num_buffers=2):
        self.layer_count = layer_count
        self.transformer_layer = transformer_layer
        self.rank = rank
        self.num_buffers = num_buffers

        self.swap_element_size = torch.tensor([], dtype=torch.float32).element_size()
        self.aligned_bytes = 1024
        self.numel_alignment = self.aligned_bytes // self.swap_element_size
        
        # 一个transformer层内的param到其参数量的映射
        self.param_idx_in_layer_to_numel = {}
        self.param_idx_to_start_pos = {}

        # 一层transformer模型的大小
        self.layer_size, self.aligned_layer_size = self.cal_pinned_buffer_size()
        # 初始化每个buffer的信息
        self.buffer_info = self._initialize_buffer_info()

        self.buffer_size = self.layer_count * self.aligned_layer_size

    def _initialize_buffer_info(self):
        """为每个buffer初始化相同的信息"""
        buffer_info = {}
        current_offset = 0
        for buffer_id in range(self.num_buffers):
            buffer_info[buffer_id] = {
                'layer_sizes': {},  # layer_idx -> size mapping
                'layer_offsets': {},  # layer_idx -> offset mapping
            }
            # 计算每个layer的大小和偏移量
            for layer_idx in range(self.layer_count):
                buffer_info[buffer_id]['layer_sizes'][layer_idx] = self.layer_size
                buffer_info[buffer_id]['layer_offsets'][layer_idx] = current_offset
                current_offset += self.aligned_layer_size
        return buffer_info
    
    def cal_pinned_buffer_size(self):
        total_size = 0
        # 计算一个transformer层的参数大小
        global_param_idx = 0
        for m in self.transformer_layer.modules():

            for id, (key, param) in enumerate(m._parameters.items()):
                param_size = param.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = param_size

                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += param_size
                global_param_idx += 1

            for key, buf in m._buffers.items():
                buf_size = buf.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = buf_size
                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += buf_size
                global_param_idx += 1

        aligned_total_size = self._io_aligned_numel(total_size)
        print(f"rank:{self.rank}, 单层transformer模型原本的参数量{total_size} ,需要的对齐字参数量:{aligned_total_size}, {total_size*4/1024/1024}MB")
        
        return total_size, aligned_total_size

    # 根据buffer_id和layer_idx获取对应的buffer
    def get_buffer(self, buffer_id, layer_idx):
        return self.buffer_info[buffer_id]['layer_offsets'][layer_idx]
    
    def get_layer_size(self):
        return self.layer_size

    def get_aligned_layer_size(self):
        return self.aligned_layer_size

    def get_buffers_size(self):
        """返回所有buffer需要的总大小"""
        return self.layer_count * self.num_buffers * self.aligned_layer_size  # 所有buffer大小相同
    
    def get_buffer_start_pos(self, buffer_id):
        return self.buffer_info[buffer_id]['layer_offsets'][0]
    
    def get_layer_start_pos_in_buffers(self, buffer_id, layer_idx):
        return self.buffer_info[buffer_id]['layer_offsets'][layer_idx]

    def get_param_start_pos(self, param_idx):
        return self.param_idx_to_start_pos[param_idx]
    
    def get_param_size(self, param_idx):
        return self.param_idx_in_layer_to_numel[param_idx]

    # 
    def _io_aligned_numel(self, numel):
        # 元素数量 % 对齐元素数量
        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

class SharedOptimCPU_for_worker5(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    # 将层的参数对应的优化器状态全部放入共享内存
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            # 确保当前这个层的参数和对应的优化器状态都不在GPU上
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            # 将参数的梯度初始化为与参数数据形状相同的全零张量
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            # 调用一次优化器的 step 方法，以强制初始化优化器状态
            # print(f"优化器初始化之前：{optimizer.state}")
            # 强制初始化之前，这个字典是空的
            # 另外需要注意，这个字典可能有多个键值对，即多个 参数：{step:,'exp_avg','exp_avg_sq'}
            optimizer.step() 
            # print(f"优化器初始化之后：{optimizer.state}")
            # 3) move optimzer.state to shared memory
            # 将优化器状态放入共享内存（甚至连step数也放进去了）
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    # 将一阶动量和二阶动量加入到共享内存
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # print("当前 Tensor 占用的内存大小:", memory_size_mb, "MB")
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    # 将 step 的值换成tensor值，也放入共享内存 
                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            # 将层的参数的梯度置为None，相当于清空梯度
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            # 手动触发垃圾回收
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)
    
    # 1.保险操作，确保逻辑正确（确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外其他的key的value不是tensor）
    # 2.深度拷贝model，其实就是一个层（self.shared_model）
    # 3.将复制的层的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
    @torch.no_grad()
    def init_in_subproc(self, id, cpu_layers, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 这个if就是个保险，确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外
        # 其他的key的value不是tensor
        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            # param_groups是一个字典列表，param_groups[0]是一个字典，其中包含模型的params以及一些优化器的参数，
            # 通过调整这些参数可以实现优化器对这一部分params更灵活的控制
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))
            
            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))

        # 废弃
        # 📍记住当前层各个param原始的形状
        # for param in self.shared_model.parameters():
        #     param.ds_shape = param.shape
        # for name, buffer in self.shared_model.named_buffers():
        #     # print(f"Buffer name: {name}")
        #     # print(f"Buffer shape: {buffer.shape}")
        #     buffer.ds_shape = buffer.shape

        # 该参数默认为false
        if self.no_pin_model:
            self.pinned_model = None
        # 该参数默认为false，执行else
        elif id in cpu_layers:
             # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            # 深度拷贝model
            # print(f"rank{self.rank}拷贝第{id}层:{self.shared_model}")
            self.pinned_model = copy.deepcopy(self.shared_model)
            # 将cpu上本地model的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
        # 若当前遍历到的layer不属于当前rank，则不会对该layer创建一个pinned版本
        else:
            self.pinned_model = None

    # 重新初始化优化器
    def re_init(self):
        for param in self.shared_model.parameters():
            assert param.data.is_shared()
            assert param.requires_grad 
        for param, state in self.shared_optimizer.state.items():
            for k, v in state.items():
                assert isinstance(v, torch.Tensor) and v.is_shared()

        self.pinned_model = copy.deepcopy(self.shared_model)
        convert_to_pinned(self.pinned_model)
        

    # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
    # 模型(就是一层)的 buffer 中，即shared_model.named_buffers()
    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        # 若shared_model中（放在共享内存中的vlayer）有pinned buf这个属性，
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    # 将共享模型中的参数和缓冲区同步到本地的固定内存模型(在pinned memory中)中
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))

    @torch.no_grad()
    def sync_pinned_buffer(self, pinned_buffer):
        for pinned_param, shared_param in zip(pinned_buffer.parameters(), self.shared_model.parameters()):
            print(f"rank:{self.rank}, pinned_param:{pinned_param}, shared_param:{shared_param}, is shared?{shared_param.is_shared()}")
            pinned_param.data.copy_(shared_param.view(-1).data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(pinned_buffer.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.view(-1).data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)

# 暂时废弃，因为优化器的状态可以就装在shared memory上不用管
class PinnedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """
    # 将层的参数对应的优化器状态全部放入共享内存
    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """ 
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU
            # 确保当前这个层的参数和对应的优化器状态都不在GPU上
            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient 
            # 将参数的梯度初始化为与参数数据形状相同的全零张量
            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)
            # 调用一次优化器的 step 方法，以强制初始化优化器状态
            # print(f"优化器初始化之前：{optimizer.state}")
            # 强制初始化之前，这个字典是空的
            # 另外需要注意，这个字典可能有多个键值对，即多个 参数：{step:,'exp_avg','exp_avg_sq'}
            optimizer.step() 
            # print(f"优化器初始化之后：{optimizer.state}")
            # 3) move optimzer.state to shared memory
            # 将优化器状态放入共享内存（甚至连step数也放进去了）
            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    # 将一阶动量和二阶动量加入到共享内存
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # print("当前 Tensor 占用的内存大小:", memory_size_mb, "MB")
                        # 
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    # 将 step 的值换成tensor值，也放入共享内存 
                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient
            # 将层的参数的梯度置为None，相当于清空梯度
            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"
            # 手动触发垃圾回收
            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)

    # 📍废弃, 我们不卸载优化器状态, 没必要把优化器移动到Pinned memory
    # 将一层的shared optimizer转化为pinned optimizer
    def from_shared_to_pinned(self):
        if self.shared_optimizer is not None:
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():
                    # 将一阶动量和二阶动量加入到共享内存
                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                        # print("当前 Tensor 占用的内存大小:", memory_size_mb, "MB")
                        # 
                        v.pin_memory(); assert v.is_pinned()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))

                    # 将 step 的值换成tensor值，也放入共享内存 
                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage 
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64) 
                        state[k].pin_memory(); assert state[k].is_pinned()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            gc.collect()
            self.pinned_optimizer = self.shared_optimizer
            self.shared_optimizer = None

    # 📍
    # 对一直放在cpu上的层，依然会创建layer的pinned 版本
    # 否则，直接将shared optimizer转化为pinned optimizer，并且不创建pinned model（layer的pinned版本）
    @torch.no_grad()
    def init_in_subproc(self, id, cpu_layers, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 
        # if id not in cpu_layers:
        #     self.from_shared_to_pinned()

        # 这个if就是个保险，确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外
        # 其他的key的value不是tensor
        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared     
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad 
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()
            # param_groups是一个字典列表，param_groups[0]是一个字典，其中包含模型的params以及一些优化器的参数，
            # 通过调整这些参数可以实现优化器对这一部分params更灵活的控制
            for k, v in self.shared_optimizer.param_groups[0].items():    
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
        # elif self.pinned_optimizer is not None:
        #     for param in self.pinned_optimizer.parameters():
        #         assert param.data.is_pinned()
        #         assert param.requires_grad 
        #         # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
        #     # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
        #     for param, state in self.pinned_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.). 
        #         for k, v in state.items():
        #             assert isinstance(v, torch.Tensor) and v.is_pinned()
        #     # param_groups是一个字典列表，param_groups[0]是一个字典，其中包含模型的params以及一些优化器的参数，
        #     # 通过调整这些参数可以实现优化器对这一部分params更灵活的控制
        #     for k, v in self.pinned_optimizer.param_groups[0].items():    
        #         if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
        #             assert (not isinstance(v, torch.Tensor))

        # 该参数默认为false
        if self.no_pin_model:
            self.pinned_model = None
        # 该参数默认为false，执行else
        elif id in cpu_layers:
             # initialize local pinned model (params and buffs)
            # with torch.no_grad():
            # 深度拷贝model
            # print(f"rank{self.rank}拷贝第{id}层:{self.shared_model}")
            self.pinned_model = copy.deepcopy(self.shared_model)
            # 将cpu上本地model的参数和buffer移动到固定内存中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
        # 若当前遍历到的layer不属于当前rank，则不会对该layer创建一个pinned版本
        else:
            self.pinned_model = None

    # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
    # 模型(就是一层)的 buffer 中，即shared_model.named_buffers()
    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """  
        if self.no_pin_grad_buf:
            return
        
        # 若shared_model中（放在共享内存中的vlayer）有pinned buf这个属性，
        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:    
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
    
    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))
    
    # 将共享模型中的参数和缓冲区同步到本地的固定内存模型(在pinned memory中)中
    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """  
        if self.no_pin_model:
            return
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))



""" CPU update and sync model in background thread """
class UpdateInBkgd(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)
            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 等待最后一个放进put_queue中的任务从get_queue中拿出来，即执行完成
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

def unified_ray_get(futs):
    for item in futs:
        if isinstance(item, tuple) and len(item) == 2:
            stream, event = item
            stream.wait_event(event)
        else:
            item.result()

def _layer_waiting_futs(_layer, _layer_id):
    assert hasattr(_layer, "_gl_futs_offloading_at_bwd"), "layer_{} has not attribute: _gl_futs_offloading_at_bwd".format(_layer_id)
    unified_ray_get(_layer._gl_futs_offloading_at_bwd)
    del _layer._gl_futs_offloading_at_bwd


############################ my version #######################################
# 主要是添加新的属性 local_model，用于在UDP时等待对应层的卸载和删除完成
class UpdateInBkgd_2(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, local_model, lr_scheduler, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        self.local_model = local_model

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)
            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self, vt, layer_id):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        if self.nvprof: nvtx_range_push("__task{}(L{}) UPD(W): waiting swap and delete finish".format(vt.idx, layer_id)) 
        _layer_waiting_futs(self.local_model[layer_id].model, layer_id)
        if self.nvprof: nvtx_range_pop() 

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            if l != vt.layers[0]:
                print(f"rank:{self.rank}, 开始等待layer{l}的卸载完成")
                self._wait_swapout(vt, l)
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        # print(f"rank:{self.rank}, 更新前的参数和buffer, {vt.show_layers()}({vt.type})")
        # for l in vt.layers:
        #     for name,param in self.shared_optimizer[l].shared_model.named_parameters():
        #         print(f"rank:{self.rank}, L({l}), Parameter {name} is {param}")
        #     for name, shared_buf in self.shared_optimizer[l].shared_model.named_buffers():
        #         if shared_buf is not None:    
        #             print(f"rank:{self.rank}, L({l}), buffer {name} is {shared_buf}")
        # torch.set_printoptions(precision=15)
        # print(f"rank:{self.rank}, 在step之前打印CPU上的梯度, {vt.show_layers()}({vt.type})")
        # for l in vt.layers:
        #     for name,param in self.shared_optimizer[l].shared_model.named_parameters():
        #         print(f"rank:{self.rank}, L({l}), Parameter {name} is {param.grad}")
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 

            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

        # print(f"rank:{self.rank}, 更新后的参数和buffer, {vt.show_layers()}({vt.type})")
        # for l in vt.layers:
        #     for name,param in self.shared_optimizer[l].shared_model.named_parameters():
        #         print(f"rank:{self.rank}, L({l}), Parameter {name} is {param}")
        #     for name, shared_buf in self.shared_optimizer[l].shared_model.named_buffers():
        #         if shared_buf is not None:    
        #             print(f"rank:{self.rank}, L({l}), buffer {name} is {shared_buf}")

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 等待最后一个放进put_queue中的任务从get_queue中拿出来，即执行完成
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))


############################ my version 3 #######################################
# 主要是添加新的属性 local_model，用于在UDP时等待对应层的卸载和删除完成
# 3版新特性：
#  与原版和第二版不同，一次处理一个layer，用于将当前层的更新和前一层的BWD overlap
class UpdateInBkgd_3(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, local_model, lr_scheduler, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        self.local_model = local_model

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt, layer_id = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt, layer_id) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt, layer_id)
            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self, vt, layer_id):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        if self.nvprof: nvtx_range_push("__task{}(L{}) UPD(W): waiting swap and delete finish".format(vt.idx, layer_id)) 
        _layer_waiting_futs(self.local_model[layer_id].model, layer_id)
        if self.nvprof: nvtx_range_pop() 

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt, layer_id):
        """ update B of this pack """  
        print(f"rank:{self.rank}, 开始等待layer{layer_id}的卸载完成")
        self._wait_swapout(vt, layer_id)
        # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
        # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
        self.shared_optimizer[layer_id].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt, layer_id):
        """ update W,K of this pack """  
        assert vt.In['dW'][layer_id].medium == "LOC"
        assert vt.In['W'][layer_id].medium == "SHM"  
        assert vt.In['K'][layer_id].medium == "SHM"
        assert vt.Out['W'][layer_id].medium == "SHM"
        assert vt.Out['K'][layer_id].medium == "SHM" 

        # 使用共享优化器更新参数
        self.shared_optimizer[layer_id].step() # Update shared model and optim using swap-out'ed local .grad
        # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
        if self.lr_scheduler != []: # "gpt2_huggingface"
            if self.lr_scheduler[layer_id] is not None:
                assert self.shared_optimizer[layer_id].shared_optimizer is not None
                self.lr_scheduler[layer_id].step() 
            else:
                assert self.shared_optimizer[layer_id].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt, layer_id):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt, layer_id)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 等待最后一个放进put_queue中的任务从get_queue中拿出来，即执行完成
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

# work5最初的版本，嵌套线程，即卸载到nvme丢给swap_to_nvme_handler执行
# 添加功能：
# 1.在CPU上更新完成后，将该vt所有layer卸载到nvme
class UpdateInBkgd_for_worker5(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, swap_to_nvme_handler, lr_scheduler, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        # 📍
        self.swap_to_nvme_handle: SwapToNVMeInBkgd = swap_to_nvme_handler

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)

            # 📍目前是顺序卸载
            self.swap_to_nvme_handle.iput(vt)

            vt_idx = self.swap_to_nvme_handle.get()
            print(f"rank:{self.rank}, vt[{vt.layers}]已从swap_to_nvme get_queue中拿到")
            assert vt_idx == vt.idx

            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

    # 该类新添加的函数，专门用来在更新后卸载参数到NVMe
    # 直觉上应该把该vt给另一个线程,后台进行到nvme的卸载
    # def swap_out

# 上一个类把卸载到nvme的任务丢给异步线程执行，该类串行执行，不嵌套线程
class UpdateInBkgd_for_worker5_param_sync_version(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, layer_id_to_layer_idx, shared_model_nvme, layer_num, cpu_layer_id, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)

            # 📍目前是顺序卸载
            self._swap_out_from_pinned_buffer(vt)

            # 将shared_model的data和grad置为空
            self.shared_model_nvme.delete_vts_shared_model_param_grad_buf(vt)

            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

    # 该类新添加的函数，专门用来在更新后卸载参数到NVMe
    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            # if vt.has_data and layer_id == vt.layers[0]:
            #     continue
            # if layer_id == self.layer_num-2:
            #     continue
            # if layer_id == self.layer_num-3:
            #     continue
            # # 最后一层同理
            # if vt.has_criterion and layer_id == vt.layers[-1]:
            #     continue
            if layer_id in self.cpu_layer_id:
                continue

            # print(f"rank:{self.rank}, 准备要卸载的vt的idx为:{vt.idx}, 类型为{vt.type}")
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            # print(f"rank:{self.rank}, layer{layer_id}取到的layer_idx为{layer_idx}")
            print(f"rank:{self.rank}, {layer_id}准备从pinned buffer卸载", flush=True)
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync_2(layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)

# UpdateInBkgd_for_worker5类把卸载到nvme的任务丢给异步线程执行，该类串行执行，不嵌套线程
class UpdateInBkgd_for_worker5_2(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, layer_id_to_layer_idx, shared_model_nvme, layer_num, cpu_layer_id, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            start_time = time.time()
            self._step(vt)
            end_time = time.time()
            step_time = end_time - start_time
            print(f"rank:{self.rank}, {vt.layers}更新完成, 耗时{step_time:.6f}秒", flush=True)

            # 📍目前是顺序卸载
            start_time = time.time()
            self._swap_out_from_pinned_buffer(vt)
            end_time = time.time()
            swap_out_time = end_time - start_time
            print(f"rank:{self.rank}, {vt.layers}从pinned buffer卸载完成, 耗时{swap_out_time:.6f}秒", flush=True)

            # 将shared_model的data和grad置为空
            self.shared_model_nvme.delete_vts_shared_model_param_grad_buf(vt)

            # 任务的idx加入到get_queue队列中
            self.get_queue.add((vt.idx, step_time, swap_out_time))
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx, _, _ = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

    # 该类新添加的函数，专门用来在更新后卸载参数到NVMe
    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            # if vt.has_data and layer_id == vt.layers[0]:
            #     continue
            # if layer_id == self.layer_num-2:
            #     continue
            # if layer_id == self.layer_num-3:
            #     continue
            # # 最后一层同理
            # if vt.has_criterion and layer_id == vt.layers[-1]:
            #     continue
            if layer_id in self.cpu_layer_id:
                continue

            # print(f"rank:{self.rank}, 准备要卸载的vt的idx为:{vt.idx}, 类型为{vt.type}")
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            # print(f"rank:{self.rank}, layer{layer_id}取到的layer_idx为{layer_idx}")
            print(f"rank:{self.rank}, {layer_id}准备从pinned buffer卸载", flush=True)
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync(layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)

# 25/1/9添加
# 与上一个的区别：针对new cpu layer进行针对处理
class UpdateInBkgd_for_worker5_2_new_cpu_layer(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, layer_id_to_layer_idx, shared_model_nvme, layer_num, cpu_layer_id, new_cpu_layer_id, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id
        self.new_cpu_layer_id = new_cpu_layer_id

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)

            # 📍目前是顺序卸载
            self._swap_out_from_pinned_buffer(vt)

            # 将shared_model的data和grad置为空
            self.shared_model_nvme.delete_vts_shared_model_param_grad_buf_for_new_cpu_layer(vt, self.new_cpu_layer_id)

            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

    # 该类新添加的函数，专门用来在更新后卸载参数到NVMe
    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id or layer_id in self.new_cpu_layer_id:
                continue

            # print(f"rank:{self.rank}, 准备要卸载的vt的idx为:{vt.idx}, 类型为{vt.type}")
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            # print(f"rank:{self.rank}, layer{layer_id}取到的layer_idx为{layer_idx}")
            print(f"rank:{self.rank}, {layer_id}准备从pinned buffer卸载", flush=True)
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync(layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)


# 版本3：双buffer版本
class UpdateInBkgd_for_worker5_double_buffer(object):
    """ Handles CPU update model in background thread for runtime.py 
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task. 
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, layer_id_to_layer_idx, shared_model_nvme, layer_num, cpu_layer_id, rank, nvprof=False):
        self.swapout_stream = swapout_stream # 实际为默认计算流
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.shared_model_nvme: SharedModelNVMe_double_buffer = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # 实例化一个线程安全的队列，用于put
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread
        # 实例化一个线程安全的队列，用于get
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread
        # 创建一个update线程
        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True
        # 开启update线程
        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        # 
        self.the_last_put = None

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
    # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
    # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上shared_model.pinned_buf中的数据复制到shared_model共享内存的vlayer的 buffer 中，
    #   即shared_model自身上（shared_model.named_buffers())
    # 3.更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    # 4.将完成的任务的idx加入到get_queue队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
            # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
            self._wait_swapout()
            # 对给定vtask中所有layer执行更新buffer操作：将cpu pinned memory中的buffer数据复制到
            # 位于共享内存的 layer 自身的 buffer 中，即shared_model自身上（shared_model.named_buffers())
            self._update_buf(vt) # if using local pinned model for B'
            # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
            self._step(vt)

            # 获取buffer_id，现在是UPD任务，buffer已在BWD任务时获取
            buffer_id = self.shared_model_nvme.get_buffer_id(vt)

            # 📍目前是顺序卸载
            self._swap_out_from_pinned_buffer(buffer_id, vt)

            # TODO:
            # 1.释放buffer_id
            # 2.将shared_model的data和grad置为空

            # 1.释放buffer_id
            self.shared_model_nvme.release_buffer(buffer_id)
            # 2.将shared_model的data和grad置为空
            self.shared_model_nvme.delete_vts_shared_model_param_grad_buf(vt)

            # 任务的idx加入到get_queue队列中
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 在默认流上阻塞，直到其中所有任务完成。实际情况是该线程在swapout dW B后工作，其实就是在等默认流上的swap out完成
    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)

    # 对给定vtask中所有layer执行更新buffer操作：将pinned memory中的buffer数据复制到shared_model共享内存的vlayer的 buffer 中，
    # 即shared_model自身上（shared_model.named_buffers())
    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """  
        # 调用vt中的所有的layer的
        for l in vt.layers:
            # 将pinned memory (shared_model.pinned_buf) 中保存的buffer tensor复制到位于共享内存的
            # 模型(就是一层)的 buffer 中，即shared_model.named_buffers() 
            self.shared_optimizer[l].update_buf()
    
    # 更新vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """  
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"  
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM" 
            # 使用共享优化器更新参数
            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad
            # 如果配置了学习率调度器，则调用相应的学习率调度器来更新学习率
            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step() 
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。
    # 📌这意味着UpdateInBkgd实例的线程会开始执行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
    # 
    def synchronize(self): 
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx 
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))

    # 该类新添加的函数，专门用来在更新后卸载参数到NVMe
    def _swap_out_from_pinned_buffer(self, buffer_id, vt):
        for layer_id in vt.layers:
            # if vt.has_data and layer_id == vt.layers[0]:
            #     continue
            # if layer_id == self.layer_num-2:
            #     continue
            # if layer_id == self.layer_num-3:
            #     continue
            # # 最后一层同理
            # if vt.has_criterion and layer_id == vt.layers[-1]:
            #     continue
            # 如果该layer在cpu_layer_id中，则跳过
            if layer_id in self.cpu_layer_id:
                continue

            # print(f"rank:{self.rank}, 准备要卸载的vt的idx为:{vt.idx}, 类型为{vt.type}")
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            # print(f"rank:{self.rank}, layer{layer_id}取到的layer_idx为{layer_idx}")
            self.shared_model_nvme.swap_out_from_pinned_buffer(buffer_id, layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)


def delete_param_grad_buf_for_shared_model(top_module, manual_gc=False):
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                # 删除参数：创建了一个形状为空的张量，并将其放置在 CPU 上。这个张量没有任何元素，因为它的形状是 (0,)
                param.data = torch.empty(0, device="cpu")
                # delete grad
                # 则将梯度置为 None，相当于删除梯度
                if param.grad is not None:
                    param.grad = None
                # 将参数从计算图中分离，使其成为叶子节点，以防止梯度传播
                # 📍这块不用detach_()，使param的required_grad一直为true
                # param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                # assert not param.requires_grad
        # 对于每个模块的缓冲区，通过 for key, buf in m._buffers.items() 遍历模块的缓冲区字典。如果缓冲区不为 None，
        # 则将其替换为一个空的零张量，相当于删除缓冲区数据
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    # apply：用于递归地应用一个函数到模块的每个子模块上
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams

# 不删除grad
def delete_param_buf_for_shared_model(top_module, manual_gc=False):
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param
                # 删除参数：创建了一个形状为空的张量，并将其放置在 CPU 上。这个张量没有任何元素，因为它的形状是 (0,)
                param.data = torch.empty(0, device="cpu")
                # delete grad
                # 则将梯度置为 None，相当于删除梯度
                # if param.grad is not None:
                #     param.grad = None
                # 将参数从计算图中分离，使其成为叶子节点，以防止梯度传播
                # 📍这块不用detach_()，使param的required_grad一直为true
                # param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                # assert not param.requires_grad
        # 对于每个模块的缓冲区，通过 for key, buf in m._buffers.items() 遍历模块的缓冲区字典。如果缓冲区不为 None，
        # 则将其替换为一个空的零张量，相当于删除缓冲区数据
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")
    # apply：用于递归地应用一个函数到模块的每个子模块上
    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams

from deepspeed.runtime.swap_tensor.partitioned_param_swapper_2 import AsyncPartitionedParameterSwapper

# 专门提供对VLayer类(就是模型的一层)的NVMe的交互，即对里面的参数调用param_swapper的方法
# 以transformer的子层为粒度进行通信
class SharedModelNVMe(object):
    # empty model就是参数、梯度、缓冲区全为空的layer列表，方便我们进行pinned model的创建（因为不用复制数据）
    def __init__(self, shared_model, empty_model, param_swapper, layer_count, transformer_layer_idx_to_shape, cpu_layer_id, rank):
        self.shared_model = shared_model
        self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.layer_count = layer_count
        self.transformer_layer_idx_to_shape = transformer_layer_idx_to_shape
        self.rank = rank
        
        # 📌一个pinned buffer就是一个transformer层
        self.pinned_buffer = []
        self.empty_model = empty_model

        # 注册一个成员, 用于标记无需卸载到nvme的层
        self.cpu_layer_id = cpu_layer_id

        # 创建pinned buffer
        for layer_idx in range(self.layer_count):
            self.initial_pinned_buffer(layer_idx)

    # 
    def initial_pinned_buffer(self, layer_idx):
        pinned_buffer = copy.deepcopy(self.empty_model[1]) # 第0层是embedding
        global_param_idx = 0
        for m in pinned_buffer.modules():
            for param_idx,(key, param) in enumerate(m._parameters.items()):
                compute_buffer, swap_buffer = self.param_swapper._allocate_and_return_buffers_for_swap_in_2(layer_idx, global_param_idx)
                param.data = compute_buffer.data
                global_param_idx+=1

            for key, buf in m._buffers.items():
                compute_buffer, swap_buffer = self.param_swapper._allocate_and_return_buffers_for_swap_in_2(layer_idx, global_param_idx)
                buf.data = compute_buffer.data
                global_param_idx += 1

        self.pinned_buffer.append(pinned_buffer)

    def get_pinned_buffer(self, layer_idx):
        return self.pinned_buffer[layer_idx]

    # 按照子层的粒度卸载layer
    def swap_out_from_pinned_buffer(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0

        for pinned_m in pinned_buffer.modules():
            for key, param in pinned_m._parameters.items():
                if param is not None:
                    # pinned_m._parameters[key].data.copy_(param.data, non_blocking=False)
                    self.param_swapper.swap_out_2(param, layer_id, global_param_idx)
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    self.param_swapper.swap_out_2(buf, layer_id, global_param_idx)
                    global_param_idx += 1

    def swap_out_from_pinned_buffer_sync(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        for pinned_m in pinned_buffer.modules():
            for key, param in pinned_m._parameters.items():
                if param is not None:
                    self.param_swapper.swap_out_2_sync(param, layer_id, global_param_idx)
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    self.param_swapper.swap_out_2_sync(buf, layer_id, global_param_idx)
                    global_param_idx += 1

    # 真正的同步，底层C++库不会放到线程中，而是直接执行
    def swap_out_from_pinned_buffer_sync_2(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        for pinned_m in pinned_buffer.modules():
            for key, param in pinned_m._parameters.items():
                if param is not None:
                    self.param_swapper.swap_out_2_sync_2(param, layer_id, global_param_idx)
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    self.param_swapper.swap_out_2_sync_2(buf, layer_id, global_param_idx)
                    global_param_idx += 1

    # 无需先放到pinned buffer上，直接从shared memory中卸载到nvme
    def swap_out_from_shared_memory_and_release(self, layer_id):
        global_param_idx = 0
        print(f"rank:{self.rank}, 正在卸载layer{layer_id}",flush=True)
        for cpu_m in self.shared_model[layer_id][0].modules(): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # one_dim_param = param.view(-1)
                    # 📍内部会等待异步卸载的完成，因此该函数执行完卸载就一定完成了
                    # print(f"rank:{self.rank}, 初始化卸载时，layer{layer_id}param:{param}, 形状为:{param.shape}")
                    self.param_swapper.swap_out_2_sync_2(param, layer_id, global_param_idx)
                    global_param_idx += 1

            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # one_dim_param = buf.view(-1)
                    # print(f"rank:{self.rank}, 初始化卸载时，layer{layer_id}buffer:{buf}, 形状为:{param.shape}")
                    self.param_swapper.swap_out_2_sync_2(buf, layer_id, global_param_idx)
                    global_param_idx += 1

        # 卸载完成后，删除cpu上shared model的参数、梯度、缓冲区
        self._delete_param_grad_buf(self.shared_model[layer_id][0])

    def _delete_param_grad_buf(self, model, manual_gc=False):
        delete_param_grad_buf_for_shared_model(model, manual_gc=manual_gc)
        
    # 从nvme向pinned buffer复制
    def swap_in(self, layer_id, layer_idx):
        # 
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        # for pinned_m, cpu_m in zip(pinned_model.modules(), self.shared_model[layer_idx].modules()):
        #     for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
        #         if param is not None:
        #             # pinned_m._parameters[key].data.copy_(param.data, non_blocking=False)
        #             self.param_swapper.swap_in_2(layer_id, layer_idx, global_param_idx)
        #             global_param_idx += 1

        #     for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
        #         if buf is not None:
        #             self.param_swapper.swap_in_2(layer_id, layer_idx, global_param_idx)

        for pinned_m in pinned_buffer.modules():
            # print(f"rank:{self.rank},xxxxxx: {pinned_m}")
            for key, param in pinned_m._parameters.items():
                
                if param is not None:
                    print(f"rank:{self.rank}, 正在读取{layer_id}-{global_param_idx}", flush=True)
                    # pinned_m._parameters[key].data.copy_(param.data, non_blocking=False)
                    self.param_swapper.swap_in_2(layer_id, layer_idx, global_param_idx, async_op=False)
                    # print(f"rank:{self.rank}, 刚刚接收的layer{layer_id}:{param}, 形状为:{param.shape}")
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    print(f"rank:{self.rank}, 正在读取{layer_id}-{global_param_idx}", flush=True)
                    self.param_swapper.swap_in_2(layer_id, layer_idx, global_param_idx, async_op=False)
                    global_param_idx += 1

    def swap_in_sync(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0

        for pinned_m in pinned_buffer.modules():
            # print(f"rank:{self.rank},xxxxxx: {pinned_m}")
            for key, param in pinned_m._parameters.items():
                
                if param is not None:
                    print(f"rank:{self.rank}, 正在读取{layer_id}-{global_param_idx}", flush=True)
                    # pinned_m._parameters[key].data.copy_(param.data, non_blocking=False)
                    self.param_swapper.swap_in_2_sync(layer_id, layer_idx, global_param_idx)
                    # print(f"rank:{self.rank}, 刚刚接收的layer{layer_id}:{param}, 形状为:{param.shape}")
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    print(f"rank:{self.rank}, 正在读取{layer_id}-{global_param_idx}", flush=True)
                    self.param_swapper.swap_in_2_sync(layer_id, layer_idx, global_param_idx)
                    global_param_idx += 1

    # 真正的同步，底层C++库不会放到线程中，而是直接执行
    def swap_in_sync_2(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        for pinned_m in pinned_buffer.modules():
            for key, param in pinned_m._parameters.items():
                if param is not None:
                    self.param_swapper.swap_in_2_sync_2(layer_id, layer_idx, global_param_idx)
                    global_param_idx += 1

            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    self.param_swapper.swap_in_2_sync_2(layer_id, layer_idx, global_param_idx)
                    global_param_idx += 1

    def delete_vts_shared_model_param_grad_buf(self, vt):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])

    # 将shared model的参数的data指向pinned model（pinned buffer）的data
    def make_shared_model_point_to_pinned(self, layer_id, layer_idx):
        print(f"rank:{self.rank}, 正在进行 layer{layer_id} 的shared model data指向pinned buffer")
        pinned_buffer = self.get_pinned_buffer(layer_idx)

        global_param_idx = 0
        for cpu_m, pinned_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()):
            for key, param in pinned_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    cpu_m._parameters[key].data = param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx+=1
            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    cpu_m._buffers[key] = buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx += 1

# 与上一个的区别：以层为粒度进行cpu-nvme通信
class SharedModelNVMe_2(object):
    # empty model就是参数、梯度、缓冲区全为空的layer列表，方便我们进行pinned model的创建（因为不用复制数据）
    def __init__(self, shared_model, empty_model, param_swapper, pinned_buffer_information, layer_count, transformer_layer_idx_to_shape, layer_id_to_layer_idx, layer_num, cpu_layer_id, rank):
        self.shared_model = shared_model
        self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.pinned_buffer_information: PinnedBufferInformation_2 = pinned_buffer_information
        self.layer_count = layer_count
        self.transformer_layer_idx_to_shape = transformer_layer_idx_to_shape
        self.rank = rank
        
        # 📌一个pinned buffer就是一个transformer层
        self.pinned_buffer = []
        self.empty_model = empty_model


        # 创建pinned buffer
        for layer_idx in range(self.layer_count):
            self.initial_pinned_buffer(layer_idx)

        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id

    # 与上一个的区别，不返回每个param的补齐部分，因为根本不需要，因为不是以param粒度卸载的
    def initial_pinned_buffer(self, layer_idx):
        pinned_buffer = copy.deepcopy(self.empty_model[1]) # 第0层是embedding
        global_param_idx = 0
        for m in pinned_buffer.modules():
            for param_idx,(key, param) in enumerate(m._parameters.items()):
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param(layer_idx, global_param_idx)
                param.data = compute_buffer.data
                global_param_idx+=1

            for key, buf in m._buffers.items():
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param(layer_idx, global_param_idx)
                buf.data = compute_buffer.data
                global_param_idx += 1

        self.pinned_buffer.append(pinned_buffer)

    def get_pinned_buffer(self, layer_idx):
        return self.pinned_buffer[layer_idx]

    # 直接卸载整个layer
    def swap_out_from_pinned_buffer(self, layer_id, layer_idx):
        # start_time = time.perf_counter()
        self.param_swapper.swap_out_transformer_layer(layer_id, layer_idx)
        # end_time = time.perf_counter()
        # print(f"rank:{self.rank}，swap_out_from_pinned_buffer执行时间: {(end_time - start_time):.4f} 秒", flush=True)

    def swap_out_from_pinned_buffer_sync(self, layer_id, layer_idx):
        self.param_swapper.swap_out_transformer_layer_sync(layer_id, layer_idx)

    def _delete_param_grad_buf(self, model, manual_gc=False):
        delete_param_grad_buf_for_shared_model(model, manual_gc=manual_gc)

    def copy_shared_model_to_pinned_buffer(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        for cpu_m, pin_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # one_dim_param = param.view(-1)
                    # 📍内部会等待异步卸载的完成，因此该函数执行完卸载就一定完成了
                    # print(f"rank:{self.rank}, 初始化卸载时，layer{layer_id}param:{param}, 形状为:{param.shape}")
                    pin_m._parameters[key].data.copy_(param.data.view(-1), non_blocking=MEMCPY_NONBLK)

            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    pin_m._buffers[key].data.copy_(buf.data.view(-1), non_blocking=MEMCPY_NONBLK)

    def swap_out_from_shared_memory_through_pinned_buffer(self, bwd_vts):
        for vt in bwd_vts:
            for layer_id in vt.layers:
                if layer_id in self.cpu_layer_id:
                    continue
                layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
                self.copy_shared_model_to_pinned_buffer(layer_id, layer_idx)
                self.swap_out_from_pinned_buffer(layer_id, layer_idx)
                # print(f"rank:{self.rank}, 卸载layer{layer_id}之前")
                # for param in self.shared_model[layer_id][0].parameters():
                #     print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")
                self._delete_param_grad_buf(self.shared_model[layer_id][0])
                # print(f"rank:{self.rank}, 卸载layer{layer_id}完成")
                # for param in self.shared_model[layer_id][0].parameters():
                #     print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")


    # 废弃，还需要根据bwd vt来选择需要读取的layer，太麻烦
    def swap_in_from_nvme_to_shared_memory_through_pinned_buffer(self, bwd_vts_need_reload):
        for bwd_vt in bwd_vts_need_reload:
            for layer_id in bwd_vt.layers:
                if layer_id not in self.cpu_layer_id and layer_id in self.new_cpu_layer_id:
                    continue
                layer_idx = self.layer_id_to_layer_idx[bwd_vt.idx][layer_id]
                self.swap_in_sync(layer_id, layer_idx)
                self.alloc_shared_model_param_buf_and_reload_from_pinned_buffer(layer_id, layer_idx)

    # 直接根据传入
    def swap_in_from_nvme_to_shared_memory_through_pinned_buffer_without_bwdvt(self, new_cpu_layer_id):
        for i in range(0, len(new_cpu_layer_id), self.layer_count):
            # 获取当前批次的layer ids
            # batch_layer_ids = new_cpu_layer_id[i:i+self.layer_count]
            batch_layer_ids = new_cpu_layer_id[i:min(i + self.layer_count, len(new_cpu_layer_id))]
            print(f"rank:{self.rank}, 从NVMe读取第{i//self.layer_count + 1}批layers: {batch_layer_ids}")
            
            # 对当前批次的每个layer进行处理
            layer_idx = 0
            for layer_id in batch_layer_ids:
                # 使用0作为layer_idx，因为我们直接读取整个layer
                # 从NVMe读取数据到pinned buffer
                self.swap_in_sync(layer_id, layer_idx)
                # 分配shared model的空间并从pinned buffer加载数据
                self.alloc_shared_model_param_buf_and_reload_from_pinned_buffer(layer_id, layer_idx)
                layer_idx += 1

    # 一次性直接读一层transformer模型
    def swap_in(self, layer_id, layer_idx): # reload_from_nvme_to_pinned_buffer
        self.param_swapper.swap_in_transformer_layer(layer_id, layer_idx)

    def swap_in_sync(self, layer_id, layer_idx):
        self.param_swapper.swap_in_transformer_layer_sync(layer_id, layer_idx)

    # 将shared model的参数的data指向pinned model（pinned buffer）的data
    def make_shared_model_point_to_pinned(self, layer_id, layer_idx):
        # print(f"rank:{self.rank}, 正在进行 layer{layer_id} 的shared model data指向pinned buffer")
        pinned_buffer = self.get_pinned_buffer(layer_idx)

        global_param_idx = 0
        for cpu_m, pinned_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()):
            for key, param in pinned_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    cpu_m._parameters[key].data = param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx+=1
            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    cpu_m._buffers[key].data = buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx += 1

    def delete_vts_shared_model_param_grad_buf(self, vt):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])


    def delete_other_shared_model(self, other_layers):
        for layer_id in other_layers:
            if layer_id in self.cpu_layer_id:
                    continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])

    def delete_vts_shared_model_param_grad_buf_for_new_cpu_layer(self, vt, new_cpu_layer_id):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id or layer_id in new_cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])

    # 废弃，根本没法恢复，layer的shared model版本在首次卸载到nvme时被删掉了，一删掉就失去了共享特性
    # 因为data被用“=”赋值了
    # 恢复指定 layer_id 的shared memory版本的参数和 buffer。
    @torch.no_grad()
    def alloc_shared_model_param_buf_and_reload_from_pinned_buffer(self, layer_id, layer_idx):
        """
        从 pined buffer 中恢复 shared_model 中指定 layer_id 的参数和 buffer。
        
        参数：
            layer_id (int): 要恢复的 transformer 层的编号。
        """
        # 获取对应层的 pinned buffer 信息
        pinned_buffer_info = self.pinned_buffer_information
        
        # 获取 pinned buffer 的数据
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        
        shared_layer = self.shared_model[layer_id][0]  # 获取 shared_model 中的对应层

        print(f"rank:{self.rank}, 正在恢复{layer_id}")
        if layer_id == 5:
            for param in shared_layer.parameters():
                print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")
        global_param_idx = 0
        for shared_m, pinned_m in zip(shared_layer.modules(), pinned_buffer.modules()):
            # 复制参数
            for key, param in pinned_m._parameters.items():
                if param is not None:
                    # 创建共享内存张量并复制数据
                    shared_param_data = torch.empty(
                        self.transformer_layer_idx_to_shape[global_param_idx],
                        device='cpu'
                    ).share_memory_()
                    shared_param_data.copy_(param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK)
                    
                    # 将共享内存张量赋值给 shared_layer 的参数
                    shared_m._parameters[key].data = shared_param_data
                    if layer_id == 5:
                        print(f"\trank:{self.rank}, layer{layer_id}, shared_param.shape:{shared_m._parameters[key].shape}")
                    global_param_idx += 1

            # 复制缓冲区
            for key, buf in pinned_m._buffers.items():
                if buf is not None:
                    # 创建共享内存张量并复制数据
                    shared_buf_data = torch.empty(
                        self.transformer_layer_idx_to_shape[global_param_idx],
                        device='cpu'
                    ).share_memory_()
                    shared_buf_data.copy_(buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK)
                    
                    # 将共享内存张量赋值给 shared_layer 的缓冲区
                    shared_m._buffers[key].data = shared_buf_data
                    global_param_idx += 1

        # for m in shared_layer.modules():
        #     for key, param in m._parameters.items():
        #         if param is not None:
        #             # 获取参数对应的形状
        #             param_shape = transformer_layer_shape[global_param_idx]
        #             param_numel = pinned_buffer_info.param_idx_in_layer_to_numel[global_param_idx]
        #             param_start_pos = pinned_buffer_info.param_idx_to_start_pos[global_param_idx]
                    
        #             # 重新分配参数张量
        #             new_param_data = torch.empty(param_shape, device='cpu').share_memory_()
                    
        #             # 从 pinned buffer 中复制数据
        #             pinned_data = pinned_buffer.data[param_start_pos:param_start_pos + param_numel]
        #             new_param_data.copy_(pinned_data.view(param_shape))
                    
        #             # 赋值给 shared_model 的参数
        #             shared_layer._parameters[key].data = new_param_data
                    
        #             global_param_idx += 1

        #     for key, buf in m._buffers.items():
        #         if buf is not None:
        #             # 获取缓冲区对应的形状
        #             buf_shape = transformer_layer_shape[global_param_idx]
        #             buf_numel = pinned_buffer_info.param_idx_in_layer_to_numel[global_param_idx]
        #             buf_start_pos = pinned_buffer_info.param_idx_to_start_pos[global_param_idx]
                    
        #             # 重新分配缓冲区张量
        #             new_buf_data = torch.empty(buf_shape, device='cpu').share_memory_()
                    
        #             # 从 pinned buffer 中复制数据
        #             pinned_buf_data = pinned_buffer.data[buf_start_pos:buf_start_pos + buf_numel]
        #             new_buf_data.copy_(pinned_buf_data.view(buf_shape))
                    
        #             # 赋值给 shared_model 的缓冲区
        #             shared_layer._buffers[key].data = new_buf_data
                    
        #             global_param_idx += 1

        print(f"rank{self.rank}: 成功从 pinned buffer 中恢复 layer {layer_id} 的参数和缓冲区。")
        if layer_id == 5:
            for param in shared_layer.parameters():
                print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")

# 版本3：双buffer版本
# 与上一个的区别，构建多个buffer, 一个buffer卸载时，另一个buffer可以读取
class SharedModelNVMe_double_buffer(object):
    def __init__(self, shared_model, empty_model, param_swapper, layer_count, transformer_layer_idx_to_shape, layer_id_to_layer_idx, layer_num, cpu_layer_id, rank, num_buffers=2):
        self.shared_model = shared_model
        self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.layer_count = layer_count
        self.transformer_layer_idx_to_shape = transformer_layer_idx_to_shape
        self.rank = rank
        self.num_buffers = num_buffers
        
        # 创建多个pinned buffer
        self.pinned_buffers = []
        self.empty_model = empty_model

        self.buffer_lock = threading.Lock()
        self.available_buffers = queue.Queue()

        # 创建多个pinned buffer并加入可用队列
        for buffer_id in range(num_buffers):
            buffer_list = []
            for layer_idx in range(self.layer_count):
                buffer_list.append(self.initial_pinned_buffer(buffer_id, layer_idx))
            self.pinned_buffers.append(buffer_list)
            self.available_buffers.put(buffer_id)

        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id

        # 记录每个buffer的使用状态
        self.buffer_status = {i: {'in_use': False, 'layers': set()} for i in range(num_buffers)}

    def initial_pinned_buffer(self, buffer_id, layer_idx):
        pinned_buffer = copy.deepcopy(self.empty_model[1]) # 第0层是embedding
        global_param_idx = 0
        for m in pinned_buffer.modules():
            for param_idx,(key, param) in enumerate(m._parameters.items()):
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param_double_buffer(buffer_id, layer_idx, global_param_idx)
                param.data = compute_buffer.data
                global_param_idx+=1

            for key, buf in m._buffers.items():
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param_double_buffer(buffer_id, layer_idx, global_param_idx)
                buf.data = compute_buffer.data
                global_param_idx += 1

        return pinned_buffer

    def get_available_buffer(self, vt):
        """获取一个可用的buffer id"""
        try:
            buffer_id = self.available_buffers.get_nowait()
            with self.buffer_lock:
                self.buffer_status[buffer_id]['in_use'] = True
                self.buffer_status[buffer_id]['layers'] = vt.layers
            return buffer_id
        except queue.Empty:
            return None

    # 在之前已经获得buffer的情况下，获取buffer_id
    def get_buffer_id(self, vt):
        with self.buffer_lock:
            for buffer_id, status in self.buffer_status.items():
                if status['in_use'] and status['layers'] == vt.layers:
                    return buffer_id
        return None

    def release_buffer(self, buffer_id):
        """释放一个buffer"""
        with self.buffer_lock:
            self.buffer_status[buffer_id]['in_use'] = False
            self.buffer_status[buffer_id]['layers'] = []
        self.available_buffers.put(buffer_id)

    def get_pinned_buffer(self, buffer_id, layer_idx):
        """获取指定buffer中的指定层"""
        return self.pinned_buffers[buffer_id][layer_idx]

    def swap_out_from_pinned_buffer(self, buffer_idx, layer_id, layer_idx):
        """从指定的buffer中卸载数据到NVMe"""
        self.param_swapper.swap_out_transformer_layer_double_buffer_sync(buffer_idx, layer_id, layer_idx)
        # 更新buffer状态
        # with self.buffer_lock:
        #     self.buffer_status[buffer_idx]['layers'].add(layer_id)

    def _delete_param_grad_buf(self, model, manual_gc=False):
        delete_param_grad_buf_for_shared_model(model, manual_gc=manual_gc)

    def copy_shared_model_to_pinned_buffer(self, buffer_id, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(buffer_id, layer_idx)
        global_param_idx = 0
        for cpu_m, pin_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # one_dim_param = param.view(-1)
                    # 📍内部会等待异步卸载的完成，因此该函数执行完卸载就一定完成了
                    # print(f"rank:{self.rank}, 初始化卸载时，layer{layer_id}param:{param}, 形状为:{param.shape}")
                    pin_m._parameters[key].data.copy_(param.data.view(-1), non_blocking=MEMCPY_NONBLK)

            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    pin_m._buffers[key].data.copy_(buf.data.view(-1), non_blocking=MEMCPY_NONBLK)

    def swap_out_from_shared_memory_through_pinned_buffer(self, bwd_vts):
        for vt in bwd_vts:
            buffer_idx = self.get_available_buffer(vt)
            assert buffer_idx is not None, f"rank:{self.rank}, 没有可用的buffer..."

            for layer_id in vt.layers:
                if layer_id in self.cpu_layer_id:
                    continue
                layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
                self.copy_shared_model_to_pinned_buffer(buffer_idx, layer_id, layer_idx)
                self.swap_out_from_pinned_buffer(buffer_idx, layer_id, layer_idx)
                self._delete_param_grad_buf(self.shared_model[layer_id][0])

            self.release_buffer(buffer_idx)
        return True

    def swap_in(self, buffer_id, layer_id, layer_idx):
        """从NVMe通过pinned buffer加载到GPU"""
        self.param_swapper.swap_in_transformer_layer_double_buffer_sync(buffer_id, layer_id, layer_idx)


    def make_shared_model_point_to_pinned(self, buffer_id, layer_id, layer_idx):
        """让shared model指向指定buffer中的数据"""
        pinned_buffer = self.get_pinned_buffer(buffer_id, layer_idx)
        # ... 原有的指针重定向逻辑 ...
        global_param_idx = 0
        for cpu_m, pinned_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()):
            for key, param in pinned_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    cpu_m._parameters[key].data = param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx+=1
            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    cpu_m._buffers[key].data = buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx += 1

    def delete_vts_shared_model_param_grad_buf(self, vt):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])


class SwapToNVMeInBkgd(object):
    def __init__(self, shared_model_nvme, layer_id_to_layer_idx, bwd_vts, layer_num, rank, nvprof=False):
        self.rank = rank
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        # self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.the_last_put = None
        self.layer_num = layer_num
        self.nvprof = nvprof

        self.layer_id_to_layer_idx = layer_id_to_layer_idx

        print(f"rank:{self.rank}, 准备进行初始化卸载",flush=True)
        self.swap_to_nvme_at_start(bwd_vts)
        gc.collect()

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        self.get_queue = threadsafe_data_struct.Queue()
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()

    # 将给定vt（除了首层和最后一层）卸载到nvme上
    def swap_to_nvme_at_start(self, vts):
        for vt in vts:
            self._swap_out_from_shared(vt)


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CPU->NVMe(W、B)".format(vt.idx, vt.show_layers())) 
            self._swap_out_from_pinned_buffer(vt)
            print(f"rank:{self.rank}, vt.[{vt.layers}]从pinned buffer卸载完成", flush=True)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            if vt.has_data and layer_id == vt.layers[0]:
                continue
            if layer_id == self.layer_num-2:
                continue
            if layer_id == self.layer_num-3:
                continue
            # 最后一层同理
            if vt.has_criterion and layer_id == vt.layers[-1]:
                continue

            # print(f"rank:{self.rank}, 准备要卸载的vt的idx为:{vt.idx}, 类型为{vt.type}")
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            # print(f"rank:{self.rank}, layer{layer_id}取到的layer_idx为{layer_idx}")
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync_2(layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)

    def _swap_out_from_shared(self, vt):
        """ update W,K of this pack """  
        for layer_id in vt.layers:
            if vt.has_data and layer_id == vt.layers[0]:
                continue
            if layer_id == self.layer_num-2:
                continue
            if layer_id == self.layer_num-3:
                continue
            # 最后一层同理
            if vt.has_criterion and layer_id == vt.layers[-1]:
                continue

            self.shared_model_nvme.swap_out_from_shared_memory_and_release(layer_id) # Update shared model and optim using swap-out'ed local .grad

    def iput(self, vt):
        ''' Call by upstream thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()


class SwapInCpuInBkgd(object):
    def __init__(self, syncpin_handler, shared_model_nvme, layer_id_to_layer_idx, layer_num, rank, nvprof=False):
        self.rank = rank
        self.nvprof = nvprof
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        self.syncpin_handler = syncpin_handler
        self.layer_num = layer_num

        # self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.the_last_put = None

        self.layer_id_to_layer_idx = layer_id_to_layer_idx

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        self.get_queue = threadsafe_data_struct.Queue()
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
    
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 拿到put_queue中的第一个vtask
            # 若队列中没有任务，会一直等待，直到出现任务此处才会继续进行
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) NVMe->CPU(W、B)".format(vt.idx, vt.show_layers())) 
            print(f"rank:{self.rank}, 开始swap in cpu {vt.layers}")
            self._swap_in(vt)
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    def _swap_in(self, vt):
        """ update W,K of this pack """  
        for layer_id in vt.layers:
            print(f"rank:{self.rank}, 正在swap in layer{layer_id}")
            # 整个模型的首层永远存在pinned model，而不是pinned buffer，不会卸载到nvme
            if vt.has_data and layer_id == vt.layers[0]:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-3:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-2:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            if layer_id == self.layer_num-1:
                self.syncpin_handler.input_one_layer(layer_id, vt)
                continue
            # if vt.has_criterion and layer_id == vt.layers[-2]:
            #     self.syncpin_handler.input_one_layer(layer_id, vt)
            
            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
            self.shared_model_nvme.swap_in_sync_2(layer_id, layer_idx) # Update shared model and optim using swap-out'ed local .grad
            print(f"rank:{self.rank},layer{layer_id} SWAP IN完成", flush=True)
            self.shared_model_nvme.make_shared_model_point_to_pinned(layer_id, layer_idx)
            print(f"rank:{self.rank},layer{layer_id}的sharedmodel已指向pinned buffer", flush=True)
    # def sync_to_shared_model(self):


    def iput(self, vt):
        ''' Call by upstream thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()

# 
class SyncPinModelInBkgd(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            # 两种情况：
            # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            # 2.若W和B已经被固定在rank上了，则什么都不用做
            self._sync_pinned_model(vt)
            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                self.shared_optimizer[l].sync_pinned_model()
            # 若W和B已经被固定在rank上了，则什么都不用做
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()

# ================ my version =========================
class SyncPinModelInBkgd_2(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            layer_id, vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            # 两种情况：
            # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            # 2.若W和B已经被固定在rank上了，则什么都不用做
            self._sync_pinned_model(layer_id)
            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add(layer_id)
            if self.nvprof: nvtx_range_pop() 

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, layer_id):
        """ sync W,B to local pinned model for this layer """  
        # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
        # if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
        #     self.shared_optimizer[l].sync_pinned_model()
        # # 若W和B已经被固定在rank上了，则什么都不用做
        # elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
        #     pass
        # else: # P2P
        #     raise ValueError("Underdevelopment")
        self.shared_optimizer[layer_id].sync_pinned_model()
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    def input_one_layer(self, layer_id, vt):
        self.put_queue.add((layer_id, vt))

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    

# ================ my version =========================

# 以param为粒度与nvme通信
# 为解决嵌套问题写的，可同时处理nvme->pinned和shared memory->pinned
class SyncPinModelInBkgd_for_worker5_param_sync_version(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, shared_model_nvme, layer_num, layer_id_to_layer_idx, cpu_layer_id, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

        # 📍
        self.shared_model_nvme: SharedModelNVMe = shared_model_nvme
        self.layer_num = layer_num
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            # 两种情况：
            # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            # 2.若W和B已经被固定在rank上了，则什么都不用做
            self._sync_pinned_model(vt)
            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                if l in self.cpu_layer_id:
                    self.shared_optimizer[l].sync_pinned_model()
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in_sync_2(l, layer_idx)
                    print(f"rank:{self.rank},layer{l} SWAP IN完成", flush=True)
                    # 若vt是BWD任务，还需将共享模型指向固定内存模型，因为要在cpu上进行参数更新
                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
                        print(f"rank:{self.rank},layer{l}的sharedmodel已指向pinned buffer", flush=True)
            # 若W和B已经被固定在rank上了，则什么都不用做
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                if vt.type == 'BWD':
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
            else: # P2P
                raise ValueError("Underdevelopment")
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()


# 为解决线程嵌套问题写的，sync线程原本的从shared_model->pinned_model的同步，现在与nvme->pinned_model的同步
# 串行执行
class SyncPinModelInBkgd_for_worker5(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, shared_model_nvme, layer_num, layer_id_to_layer_idx, cpu_layer_id, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

        # 📍
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.layer_num = layer_num
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            # 两种情况：
            # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            # 2.若W和B已经被固定在rank上了，则什么都不用做
            self._sync_pinned_model(vt)
            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                if l in self.cpu_layer_id:
                    self.shared_optimizer[l].sync_pinned_model()
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in_sync(l, layer_idx)
                    print(f"rank:{self.rank},layer{l} SWAP IN完成", flush=True)
                    # 若vt是BWD任务，还需将共享模型指向固定内存模型，因为要在cpu上进行参数更新
                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
                        print(f"rank:{self.rank},layer{l}的sharedmodel已指向pinned buffer", flush=True)
            # 若W和B已经被固定在rank上了，则什么都不用做
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                if l in self.cpu_layer_id:
                    pass
                else:
                    if vt.type == 'BWD':
                        layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
            else: # P2P
                raise ValueError("Underdevelopment")
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
    
# 25/1/9添加
# layer粒度 + new cpu layer处理
# 与上一个的区别：主要是对new cpu layer进行到pinned buffer的复制，这样无需对这些layer建立pinned model版本
class SyncPinModelInBkgd_for_worker5_2(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, shared_model_nvme, layer_num, layer_id_to_layer_idx, cpu_layer_id, new_cpu_layer_id, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

        # 📍
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.layer_num = layer_num
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.cpu_layer_id = cpu_layer_id
        self.new_cpu_layer_id = new_cpu_layer_id

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            # 两种情况：
            # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            # 2.若W和B已经被固定在rank上了，则什么都不用做
            self._sync_pinned_model(vt)
            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop() 

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                if l in self.cpu_layer_id:
                    self.shared_optimizer[l].sync_pinned_model()
                elif l in self.new_cpu_layer_id:
                    print(f"rank:{self.rank}, layer{l} 是新cpu layer ({vt.layers})", flush=True)
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    pinned_buffer = self.shared_model_nvme.get_pinned_buffer(layer_idx)
                    self.shared_optimizer[l].sync_pinned_buffer(pinned_buffer)
                else:
                    print(f"rank:{self.rank}, layer{l} 是nvme layer ({vt.layers})", flush=True)
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in_sync(l, layer_idx)
                    print(f"rank:{self.rank},layer{l} SWAP IN完成", flush=True)
                    # 若vt是BWD任务，还需将共享模型指向固定内存模型，因为要在cpu上进行参数更新
                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
                        print(f"rank:{self.rank},layer{l}的sharedmodel已指向pinned buffer", flush=True)
            # 若W和B已经被固定在rank上了，则什么都不用做
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()

# 双buffer版本
class SyncPinModelInBkgd_for_worker5_double_buffer(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task. 
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, shared_model_nvme, layer_num, layer_id_to_layer_idx, cpu_layer_id, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize
        # put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，
        # 自身这个线程才能取出东西，并执行后面的步骤，不然会一直阻塞
        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread
        # get_queue 代表逻辑上我这个活干完了，用于通知其他线程我的活干完了，一般是上级线程调用函数拿取这里面的值
        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

        # 📍
        self.shared_model_nvme: SharedModelNVMe_double_buffer = shared_model_nvme
        self.layer_num = layer_num
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.cpu_layer_id = cpu_layer_id

    # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    #   --若W和B已经被固定在rank上了，则什么都不用做
    # 2.将同步完成的vt的idx加入get_queue队列
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers())) 
            
            buffer_id = self.reuse_buffer(vt)
            if buffer_id is None:
                buffer_id = self.get_available_buffer(vt)
                self._sync_pinned_model(vt, buffer_id)
            else:
                for l in vt.layers:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.make_shared_model_point_to_pinned(buffer_id, l, layer_idx)

            # 将同步完成的vt的idx加入get_queue队列
            self.get_queue.add((vt.idx, buffer_id))
            if self.nvprof: nvtx_range_pop() 

    def reuse_buffer(self, vt):
        # 无论是FWD还是BWD，In['W']和In['B']都是有值的
        all_pin = all(vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN' for l in vt.layers)
        if all_pin:
            # 如果是PIN媒介，返回前一个任务使用的buffer_id
            prev_buffer_id = self.shared_model_nvme.get_buffer_id(vt)
            assert prev_buffer_id is not None, "前一个任务使用的buffer_id为None"
            print(f"rank:{self.rank}, ({vt.type}/{vt.layers})使用PIN媒介, 继续使用buffer_id:{prev_buffer_id}")
            return prev_buffer_id

        return None

    def get_available_buffer(self, vt, max_retries=10):
        # 在vt级别获取buffer
        retries = 0
        
        while True:
            buffer_id = self.shared_model_nvme.get_available_buffer(vt)
            if buffer_id is not None:
                print(f"rank:{self.rank}, ---------------------({vt.type}/{vt.layers})成功获取到buffer_id:{buffer_id}")
                return buffer_id

            # print(f"rank:{self.rank}, 没有可用的buffer, 等待... (重试次数: {retries + 1}/{max_retries})")
            time.sleep(0.001)
            retries += 1

        if buffer_id is None:
            error_msg = f"rank:{self.rank}, 在{max_retries}次尝试后仍未获取到可用buffer，程序终止"
            print(error_msg)
            raise RuntimeError(error_msg)

    # 两种情况：
    # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
    # 2.若W和B已经被固定在rank上了，则什么都不用做
    @torch.no_grad()
    def _sync_pinned_model(self, vt, buffer_id):
        """ sync W,B to local pinned model for this pack """  
        for l in vt.layers:
            # 若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                if l in self.cpu_layer_id: # l == 0 or l == self.layer_num-3 or l == self.layer_num-2 or l == self.layer_num-1:
                    self.shared_optimizer[l].sync_pinned_model()
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in(buffer_id, l, layer_idx)
                    print(f"rank:{self.rank},layer{l} SWAP IN完成", flush=True)
                    # 若vt是BWD任务，还需将共享模型指向固定内存模型，因为要在cpu上进行参数更新
                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(buffer_id, l, layer_idx)
                        print(f"rank:{self.rank},layer{l}的sharedmodel已指向pinned buffer", flush=True)
            # 若W和B已经被固定在rank上了，则什么都不用做
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
    
    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)

    # 返回get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
