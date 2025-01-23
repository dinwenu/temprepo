# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from collections import OrderedDict as ODict
import threading

import torch
from torch.autograd import Variable

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from prof_data_struct import ConstMeta, TensorMeta
from task_data_struct import Medium, vTask
import threadsafe_data_struct

# 只有单个GPU和vDP才会用这个
if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

""" Handles local swap-in/out of stashX/X/dX. 

    Assumption:
        0) stateless
        1) during swap, stashX/X/dX has no grad
"""

# 1.在cpu上的pinned memory上创建一个空tensor
# 2.异步的将gpu上的tensor拷贝到刚刚分配的空tensor上
# 返回cpu_named_tensors
@torch.no_grad()
def swapout(cuda_named_tensors, pin_memory=True): 
    """ Argument: cuda_named_tensors (StashX of vPP, LocalX of vDP)
        Return: cpu_named_tensors in pinned memory
    """
    cpu_named_tensors = ODict()
    for name,tensor in cuda_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            assert tensor.is_cuda and not tensor.requires_grad
            # 在cpu上的pinned memory上创建一个空tensor
            cpu_named_tensors[name] = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=pin_memory)
            # 异步的将gpu上的tensor拷贝到刚刚分配的空tensor上
            cpu_named_tensors[name].copy_(tensor, non_blocking=MEMCPY_NONBLK) # inplace copy from cuda tensor to pinned memory 
            # assert cpu_named_tensors[name].is_pinned()
        elif isinstance(tensor, (float,int)):
            cpu_named_tensors[name] = tensor
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                assert t.is_cuda and not t.requires_grad
                tmp.append( torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=pin_memory) )
                tmp[-1].copy_(t, non_blocking=MEMCPY_NONBLK)
                # assert tmp[-1].is_pinned()
            cpu_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))
    del cuda_named_tensors
    return cpu_named_tensors

# 将字典中的tensor替换为GPU版本，即移动到GPU上
@torch.no_grad()
def swapin(cpu_named_tensors): 
    """ Argument: cpu_named_tensors (stashX of vPP, stashX/X/dX of vDP) 
        Return: cuda_named_tensors
    """
    cuda_named_tensors = ODict()
    for name,tensor in cpu_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            assert not tensor.is_cuda and not tensor.requires_grad
            # assert tensor.is_pinned()
            # 将tensor移动到GPU上
            cuda_named_tensors[name] = tensor.cuda(non_blocking=MEMCPY_NONBLK) # to(device='cuda', non_blocking=True) # create new cuda tensor
        elif isinstance(tensor, (float,int)):
            cuda_named_tensors[name] = tensor
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                assert not t.is_cuda and not t.requires_grad
                # assert t.is_pinned()
                tmp.append( t.cuda(non_blocking=MEMCPY_NONBLK) )
            cuda_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))
    # del cpu_named_tensors
    return cuda_named_tensors

""" Prefetch StashX/X/dX """

class SwapIn(object):
    """ Handle prefetch StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        Simliar to P2P.prerecv with double buffering.
        
        Step:
        1) main thread: waits for the running prefetch finish
        2) main thread: allocate or reuse buffer on GPU
        3) swapin thread: synchronize X on CPU
        4) swapin thread: copy in X in swapin_cudastream

        Assumption:
        0) statefull
        1) during swap, StashX/X/dX has no grad (but double buffering can have grad) 
        2) FIFO ordering. put each layerId, and get prefetched X. 
    """
    # sync_fn：即接收stashX或local_X的线程的recv函数
    def __init__(self, sync_fn, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.sync_fn = sync_fn
        self.rank = rank
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() 
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ layerId, layerId, ...]  # between main and swapin thread
        self.get_queue = threadsafe_data_struct.Queue() # [ #0X, #1X, ...] # between swapin and main thread
        self.swapin_thread = threading.Thread(target=self._thread_func)
        self.swapin_thread.daemon = True
        self.swapin_thread.start()
        # for preget
        self.is_running = False
        self.ubatch_idx = int(0)
        self.double_bufs = [None, None]
        # print("[SwapIn] rank{} started swapin_thread".format(self.rank))
    
    # 将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素
    # 若get_queue中没有东西，这里就会一直等待
    def _wait(self): 
        ''' Wait for the running swapin. Called in main thread.
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors.
        '''
        assert self.is_running, "no running swapin"
        self.is_running = False
        return self.get_queue.remove()

    # def _wait(self): # Deprecated: causing strange slowdown
    #     ''' Wait for the running swapin (iput). Called in main thread.
    #         Assumption: only one swapin can be running.
    #         Return: swapined cuda_named_tensors.
    #     '''
    #     assert self.is_running, "no running swapin"
    #     self.is_running = False
    #     # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
    #     cuda_named_tensors, ev_swapin = self.get_queue.remove()
    #     self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
    #     # torch.cuda.default_stream(self.rank).synchronize() # wait Compute (PlanC)
    #     # if self.nvprof: nvtx_range_pop() 
    #     return cuda_named_tensors
    
    # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
    # 若给定了buffer，则该函数直接返回buffer，什么也不做
    def _allocate(self, layer_id, named_metas, buffer): 
        ''' Allocate or reuse the buffer for next swapin. Called in main thread.
            Argument: named_metas = XMETA.get(ubatchsize, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                       buffer = allocated cuda_named_tensors
            Return: newly allocated or reused buffer
        '''
        if buffer is None: # allocate new tensors
            if self.nvprof: nvtx_range_push("L{} Alloc(X)".format(layer_id)) 
            cuda_named_tensors = ODict()
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    assert meta.is_ubatch
                    cuda_named_tensors[name] = torch.empty(meta.shape, dtype=meta.dtype, device="cuda:%d"%self.rank) 
                elif isinstance(meta, ConstMeta): 
                    cuda_named_tensors[name] = meta.const
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    cuda_named_tensors[name] = [ torch.empty(m.shape, dtype=m.dtype, device="cuda:%d"%self.rank) for m in meta ]
                else:
                    raise ValueError("unknown meta={}".format(meta))   
            if self.nvprof: nvtx_range_pop() 
            return cuda_named_tensors
        else: # reuse buffered tensors
            # TODO: confirm named_metas matches buffer
            return buffer

    # 1.检测is_running是否为false，即不允许有正在执行的swap in
    # 2.将 is_running 置为true
    # 3.将(layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，这意味着SwapIn线程会开始cpu到gpu的拷贝操作
    def _sync_copyin(self, layer_id, cuda_named_tensors, ev_compute=None):
        ''' Put allocated buffer to background swapin. Call by main thread thread. 
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors. '''
        assert not self.is_running, "the swapin is still running"
        self.is_running = True
        self.put_queue.add((layer_id, cuda_named_tensors, ev_compute))

    # 将cpu上的tesnor拷贝到gpu上的tensor。该函数会返回一个记录的事件，用于让compute stream等待
    @torch.no_grad() 
    def _copyin(self, cpu_named_tensors, cuda_named_tensors):
        ''' Call by background swapin thread.
            Argument: cpu_named_tensors = src buffer on CPU
                      cuda_named_tensors = dst buffers on GPU
            (Return: this cuda_named_tensors with filled data)
        '''
        #
        with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
            for (cname,ctensor), (gname, gtensor) in zip(cpu_named_tensors.items(), cuda_named_tensors.items()): # { name: tensor/const, name: [tensors] }
                assert cname == gname and type(ctensor) == type(gtensor)
                if isinstance(ctensor, (torch.Tensor,Variable)):
                    assert not ctensor.requires_grad, \
                    "{} (requires_grad:{})".format(ctensor, ctensor.requires_grad)
                    # assert ctensor.is_pinned() 
                    gtensor.data.copy_(ctensor.data, non_blocking=MEMCPY_NONBLK)
                elif isinstance(ctensor, (float,int)):
                    assert ctensor == gtensor
                elif isinstance(ctensor, list): # output tuple of bert pretrainhead 
                    for ct,gt in zip(ctensor,gtensor):
                        assert not ct.requires_grad
                        # assert ct.is_pinned()
                        gt.data.copy_(ct.data, non_blocking=MEMCPY_NONBLK)
                else:
                    raise ValueError("unknown tensor={}".format(tensor)) # 📌代码写错了，ctensor
        return self.swapin_stream.record_event() # record a swapin event in this stream for compute stream to wait
        # # wait for copy stream 
        # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete. 
                
    # 不断从 put_queue 中拿取 layer_id, cuda_named_tensors, ev_compute 并执行：
    # 1.如果 ev_compute 不为 None，则在当前 CUDA 流上等待该事件的完成。这是为了确保在执行后续操作之前，
    #   必须等待先前的计算完成
    # 2.调用 msg_stashx.recv 方法，即拿到从src_rank传来的 cpu_tensor，若没有tensor会被阻塞住
    #   2.1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
    #   2.2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
    #       即一个 （name, tensor）
    #       若内部没有tensor显然会被阻塞住(wait)
    # 3.将cpu上的tesnor拷贝到gpu上的tensor。该函数会返回一个记录的事件，用于让compute stream等待
    # 4.将 (cuda_named_tensors, ev_swapin) 放入 get_queue 队列中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each iput'ed element
            layer_id, cuda_named_tensors, ev_compute = self.put_queue.remove() # blk
            # 如果 ev_compute 不为 None，则在当前 CUDA 流上等待该事件的完成。这是为了确保在执行后续操作之前，
            # 必须等待先前的计算完成
            if ev_compute is not None:
                self.swapin_stream.wait_event(ev_compute) # Stream waits for this event 
                # ev_compute.synchronize() # this CPU thread waits for this event. # Deprecated (too slow)
            # if self.nvprof: nvtx_range_push("__L{} SyncCopyIn(X)".format(layer_id)) 
            # sync
            #
            # sync_fn即 msg_stashx.recv 方法
            # 1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
            # 2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
            #   即一个 （name, tensor）
            #   若内部没有tensor显然会被阻塞住(wait)
            cpu_named_tensors = self.sync_fn(layer_id) # thread safe dict
            # copyin
            # 将cpu上的tesnor拷贝到gpu上的tensor。该函数会返回一个记录的事件，用于让compute stream等待
            ev_swapin = self._copyin(cpu_named_tensors, cuda_named_tensors)
            # ready to use
            # 将 (cuda_named_tensors, ev_swapin) 放入 get_queue 队列中
            self.get_queue.add( (cuda_named_tensors, ev_swapin) )
            # clean up reference
            del cuda_named_tensors
            # if self.nvprof: nvtx_range_pop() 
   
    def fetch(self, layer_id, named_metas):
        ''' Blocking fetch current X on GPU. Call by main thread.
            Feature: 
            0) Stateless
            1) Blocking compute stream (otherwise fetch ubatches can drift from compute ubatches) by cuda events
        '''
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # fetch current one
        cuda_named_tensors = self._allocate(layer_id, named_metas, None)
        self._sync_copyin(layer_id, cuda_named_tensors, ev_compute)
        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        cuda_named_tensors, ev_swapin = self._wait()
        self.compute_stream.wait_event(ev_swapin)
        # if self.nvprof: nvtx_range_pop() 
        return cuda_named_tensors # to be delete'd by runtime

    # 1.若没有正在执行的预取操作:
    #   1.1.在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
    #   1.2.将 is_running 置为true
    #   1.3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
    #       这意味着SwapIn线程会开始cpu到gpu的拷贝操作
    #   --否则，若有正在执行的接收操作，直接执行2
    # 2.将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素 (cur_named_tensors, ev_swapin)
    #   若get_queue中没有东西，这里就会一直等待
    # 3.等待ev_swapin完成
    # 4.若当前不是最后一个microbatch的stashX的预取操作，会为下一个microabtch预取stashX
    def prefetch(self, layer_id, named_metas, is_end=False): 
        ''' Blocking wait current X and unblocking pre-swapin next X. Call by main thread. 
            Assumption: 
                      1) use double buffering for parallel compute and swapin
                      2) double buffering doesn't change in shape TODO: fix
                      3) PlanA: use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                      4) no prefetch for successor group's X
            Argument: layer_id = vLayer id of current StashX/X/dX
                      named_metas = current { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      is_end = whether ends pre-swapin after current one
            Return: recved current named_tensors. '''    
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # indexing double buffer
        cur_buf_idx = self.ubatch_idx % 2 # for compute
        next_buf_idx = (self.ubatch_idx+1) % 2 # for pre-swapin

        # wait current one (if no current one, have one)
        # 📌从第二个microbatch开始，这里才会略过，因为第一个Micorbatch预取了第二个的X
        # 因此if为false，直接执行下面的wait等待接收完毕（也有可能已经接收完了）
        if not self.is_running:
            # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
            # 若给定了buffer，则该函数直接返回buffer，什么也不做
            self.double_bufs[cur_buf_idx] = self._allocate(layer_id, named_metas, None)
            # 1.检测is_running是否为false，即不允许有正在执行的swap in
            # 2.将 is_running 置为true，表示SwapIn实例的线程要开始工作了
            # 3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
            #   📌这意味着会触发SwapIn线程cpu到gpu的拷贝操作
            #  （就是往self.double_bufs[cur_buf_idx]这个字典中的tensor拷贝）
            self._sync_copyin(layer_id, self.double_bufs[cur_buf_idx], ev_compute)

        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        # 将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素
        # 若get_queue中没有东西，这里就会一直等待
        cur_named_tensors, ev_swapin = self._wait()
        # 等待swap in完成
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # if self.nvprof: nvtx_range_pop() 
        # pre-swapin next one if exsits
        # 若当前不是最后一个microbatch
        if not is_end:
            # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
            # 若给定了buffer，则该函数直接返回buffer，什么也不做
            # 📌注意第一次运行到这里时，虽然形式上给了buffer(第3个参数)，但本质上传进去的是None
            self.double_bufs[next_buf_idx] = self._allocate(layer_id, named_metas, self.double_bufs[next_buf_idx])
            # 触发SwapIn线程从cpu到gpu的拷贝操作
            # 即当前layer要接收的下一个microbatch运行所需的X
            self._sync_copyin(layer_id, self.double_bufs[next_buf_idx], ev_compute)
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.double_bufs 
            self.double_bufs = [None, None]
        return cur_named_tensors # reference only; to be deleted by runtime
    
    # suc_info：
    # 若后继任务是BWD（非第一个BWD），且输入媒介是MSG，返回 (l(后继任务的首层id), 后级任务输入X的元数据) 。非MSG直接返回None
    # 其他情况直接返回None

    # 1.在默认的计算流上记录一个事件，ev_compute
    # 2.在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
    # 3.
    #   3.1.检测is_running是否为false，即不允许有正在执行的swap in
    #   3.2.将 is_running 置为true
    #   3.3.将(layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，这意味着SwapIn线程会开始cpu到gpu的拷贝操作
    def prefetch_suc(self, suc_info): 
        ''' Prefetc successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (layer_id, named_metas)
        '''  
        if suc_info is None:
            return
        suc_layer_id, suc_named_metas = suc_info
        # if self.nvprof: nvtx_range_push("PrefetchSuc(L{}-X)".format(suc_layer_id)) 
        # must after is_end
        assert self.ubatch_idx == 0 and self.double_bufs == [None, None]
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # pre-swapin successor group's 1st ubatch if exsits
        # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
        self.double_bufs[0] = self._allocate(suc_layer_id, suc_named_metas, None)
        # 1.检测is_running是否为false，即不允许有正在执行的swap in
        # 2.将 is_running 置为true
        # 3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，这意味着SwapIn线程会开始cpu到gpu的拷贝操作
        self._sync_copyin(suc_layer_id, self.double_bufs[0], ev_compute)
        # if self.nvprof: nvtx_range_pop() 

# 
class SwapOut(object):
    """ Handle offload StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        
        Step:
        1) main thread: allocate pinned memory on CPU
        2) main thread: copy out X in swapout_stream
        3) main thread: optional delete
        4) swapout thread: wait for copy out
        5) swapout thread: isend to downstream ubatchconvert/msgstashx/localx

        Assumption:
        0) stateless
        1) during swap, StashX/X/dX has no grad
        2) FIFO ordering. put layer-by-layer, and swapout layer-by-layer. 
    """
    def __init__(self, output_fn, rank, swapout_stream=None, compute_stream=None, blocking=False, pin_memory=True, nvprof=False): # compute_stream2=None, 
        self.output_fn = output_fn # args: layer_id, named_tensor
        self.rank = rank
        self.swapout_stream = swapout_stream if swapout_stream is not None else torch.cuda.Stream(device=rank)
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        assert self.rank == torch.cuda.current_device() 
        self.blocking = blocking # 默认为false
        self.pin_memory = pin_memory
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ (layer_id, named_tensor, ev_compute), ...]  # between main and swapout thread
        # self.get_queue = threadsafe_data_struct.Queue() # [ ev_swapout, ...] # between swapout and main thread
        self.swapout_thread = threading.Thread(target=self._thread_func)
        self.swapout_thread.daemon = True
        self.swapout_thread.start()
        #
        # print("[SwapOut] rank{} started swapout_thread".format(self.rank))
    

    # flag传进来的时候为None
    #
    # 1.记录一个默认计算流上的事件 ev_compute
    # 2.在swapout stream上等待 ev_compute 事件完成，即确保计算完成后才能swapout
    # 3.在swapout_stream上:
    #   3.1.在cpu上的pinned memory上创建一个空tensor
    #   3.2.异步的将gpu上的tensor拷贝到刚刚分配的空tensor上
    #   返回cpu_named_tensors
    # 4.将 (layer_id, cpu_named_tensors, ev_swapout, flag) 添加到 put_queue 队列中。📌这意味着当前实例的线程会将已经卸载
    #   到cpu上的tensor放到 MSGstashX 实例的 send_dict 中。
    #   📌这也意味着 MSGstashX 的发送线程将向 dst_rank 发送此 tensor
    def offload(self, layer_id, cuda_named_tensors, flag=None):
        ''' Call by main thread. '''
        # record previous compute event for swapout stream to wait
        # 1.记录一个默认计算流上的事件 ev_compute
        ev_compute = self.compute_stream.record_event()
        # Allocate and CopyOut and (Delete)
        # 2.在swapout stream上等待 ev_compute 事件完成，即确保计算完成后才能swapout
        self.swapout_stream.wait_event(ev_compute) # Stream waits for this event 
        # self.swapout_stream.wait_event(ev_compute2) # Stream waits for this event 
        # if self.nvprof: nvtx_range_push("L{} SwapOut(X)".format(layer_id)) 
        # 3.在swapout_stream上:
        #   3.1.在cpu上的pinned memory上创建一个空tensor
        #   3.2.异步的将gpu上的tensor拷贝到刚刚分配的空tensor上
        #   返回cpu_named_tensors
        with torch.cuda.stream(self.swapout_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
            cpu_named_tensors = swapout(cuda_named_tensors, self.pin_memory)
        # 在 swapout_stream 上记录一个事件
        ev_swapout = self.swapout_stream.record_event() # record a swapout event in this stream for compute stream to wait
        # 该参数默认为false，不用管
        if self.blocking: # optional blocking
            self.compute_stream.wait_event(ev_swapout)
        # if self.nvprof: nvtx_range_pop() 
        # wait in background thread
        # 将 (layer_id, cpu_named_tensors, ev_swapout, flag) 添加到 put_queue 队列中
        self.put_queue.add((layer_id, cpu_named_tensors, ev_swapout, flag))
            
    # 确保tensor在完全offload到cpu后，再添加到MSGX线程的send_dict中，发送出去
    # 不断尝试从 put_queue 这个线程安全队列中拿取 layer_id, cpu_named_tensors, ev_swapout, flag，执行：
    # 1.等待ev_swapout事件执行完成
    # 2.调用 output_fn 函数
    #   2.1.调用 MSGstashX 实例的 isend 方法，向 send_dict 这个线程安全字典中的 odict[layer_id] 这个
    #       list添加：self.odict[layer_id].append(named_tensors). 这意味着MSGstashX实例的发送线程将会开始向目标rank
    #       发送 tensor
    #   2.2.调用UBatchSizeConverter的 isend 方法，将layer_id和input2：cpu_named_tensors加入到 input_queue 队列中，这意味
    #       着 UBatchSizeConverter 实例的线程将开始执行tensor大小的转换，而后还是调用MSGstashX的isend方法，将convert好的
    #       tensor列表加入到MSGstashX的send_dict中，后续就与2.1一样了
    #       
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each layer
            layer_id, cpu_named_tensors, ev_swapout, flag = self.put_queue.remove()
            # get ready for downstream
            # if self.nvprof: nvtx_range_push("__L{} WaitCopyOut(X)".format(layer_id)) 
            # if MEMCPY_NONBLK: self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. 
            
            # 默认为非阻塞，这里执行
            # 若非阻塞，这里需要等一下，ev_swapout 可能还没执行完
            if MEMCPY_NONBLK: ev_swapout.synchronize() # this CPU thread waits for this event. 
            
            # 起码在vPP的场景下，flag就是None，没发现任何将其置为True的操作
            self.output_fn(layer_id, cpu_named_tensors) if flag is None else \
            self.output_fn(layer_id, cpu_named_tensors, flag)
            # if self.nvprof: nvtx_range_pop() 


# SwapIn的变体类，无需GPUtensor的创建和CPU→GPU的复制
class StashIn(object):
    """ Handle prefetch StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        Simliar to P2P.prerecv with double buffering.
        
        Step:
        1) main thread: waits for the running prefetch finish
        2) main thread: allocate or reuse buffer on GPU
        3) swapin thread: synchronize X on CPU
        4) swapin thread: copy in X in swapin_cudastream

        Assumption:
        0) statefull
        1) during swap, StashX/X/dX has no grad (but double buffering can have grad) 
        2) FIFO ordering. put each layerId, and get prefetched X. 
    """
    # sync_fn：即接收stashX或local_X的线程的recv函数
    def __init__(self, sync_fn, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.sync_fn = sync_fn
        self.rank = rank
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() 
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ layerId, layerId, ...]  # between main and swapin thread
        self.get_queue = threadsafe_data_struct.Queue() # [ #0X, #1X, ...] # between swapin and main thread
        self.swapin_thread = threading.Thread(target=self._thread_func)
        self.swapin_thread.daemon = True
        self.swapin_thread.start()
        # for preget
        self.is_running = False
        self.ubatch_idx = int(0)
        self.double_bufs = [None, None]
        # print("[SwapIn] rank{} started swapin_thread".format(self.rank))
    
    # 将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素
    # 若get_queue中没有东西，这里就会一直等待
    def _wait(self): 
        ''' Wait for the running swapin. Called in main thread.
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors.
        '''
        assert self.is_running, "no running swapin"
        self.is_running = False
        return self.get_queue.remove()

    # 单纯的检测以下是否是is_running+放进put_queue中
    def _sync_copyin(self, layer_id, ev_compute=None):
        ''' Put allocated buffer to background swapin. Call by main thread thread. 
            Assumption: only one swapin can be running.
            Return: swapined cuda_named_tensors. '''
        assert not self.is_running, "the swapin is still running"
        self.is_running = True
        self.put_queue.add((layer_id, ev_compute))

    # 现在单纯的等待默认计算流的执行完成+调用底层的存储结构拿数据
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each iput'ed element
            layer_id, ev_compute = self.put_queue.remove() # blk
            # 如果 ev_compute 不为 None，则在当前 CUDA 流上等待该事件的完成。这是为了确保在执行后续操作之前，
            # 必须等待先前的计算完成
            if ev_compute is not None:
                self.compute_stream.wait_event(ev_compute) # Stream waits for this event 
                # ev_compute.synchronize() # this CPU thread waits for this event. # Deprecated (too slow)
            # if self.nvprof: nvtx_range_push("__L{} SyncCopyIn(X)".format(layer_id)) 
            # sync
            #
            # sync_fn即 msg_stashx.recv 方法
            # 1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
            # 2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
            #   即一个 （name, tensor）
            #   若内部没有tensor显然会被阻塞住(wait)
            cuda_named_tensors = self.sync_fn(layer_id) # thread safe dict
            # ready to use
            self.get_queue.add( (cuda_named_tensors) )
            # clean up reference
            del cuda_named_tensors
            # if self.nvprof: nvtx_range_pop() 

    # 删除了GPU上tensor的分配，直接通过_sync_copyin激活线程去数据结构拿数据
    def fetch(self, layer_id):
        ''' Blocking fetch current X on GPU. Call by main thread.
            Feature: 
            0) Stateless
            1) Blocking compute stream (otherwise fetch ubatches can drift from compute ubatches) by cuda events
        '''
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # fetch current one
        self._sync_copyin(layer_id, ev_compute)
        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        cuda_named_tensors = self._wait()
        # if self.nvprof: nvtx_range_pop() 
        return cuda_named_tensors # to be delete'd by runtime

    # 1.若没有正在执行的预取操作:
    #   1.1.在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
    #   1.2.将 is_running 置为true
    #   1.3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
    #       这意味着SwapIn线程会开始cpu到gpu的拷贝操作
    #   --否则，若有正在执行的接收操作，直接执行2
    # 2.将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素 (cur_named_tensors, ev_swapin)
    #   若get_queue中没有东西，这里就会一直等待
    # 3.等待ev_swapin完成
    # 4.若当前不是最后一个microbatch的stashX的预取操作，会为下一个microabtch预取stashX
    def prefetch(self, layer_id, named_metas, is_end=False): 
        ''' Blocking wait current X and unblocking pre-swapin next X. Call by main thread. 
            Assumption: 
                      1) use double buffering for parallel compute and swapin
                      2) double buffering doesn't change in shape TODO: fix
                      3) PlanA: use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                      4) no prefetch for successor group's X
            Argument: layer_id = vLayer id of current StashX/X/dX
                      named_metas = current { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      is_end = whether ends pre-swapin after current one
            Return: recved current named_tensors. '''    
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # indexing double buffer
        cur_buf_idx = self.ubatch_idx % 2 # for compute
        next_buf_idx = (self.ubatch_idx+1) % 2 # for pre-swapin

        # wait current one (if no current one, have one)
        # 📌从第二个microbatch开始，这里才会略过，因为第一个Micorbatch预取了第二个的X
        # 因此if为false，直接执行下面的wait等待接收完毕（也有可能已经接收完了）
        if not self.is_running:
            # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
            # 若给定了buffer，则该函数直接返回buffer，什么也不做
            self.double_bufs[cur_buf_idx] = self._allocate(layer_id, named_metas, None)
            # 1.检测is_running是否为false，即不允许有正在执行的swap in
            # 2.将 is_running 置为true，表示SwapIn实例的线程要开始工作了
            # 3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
            #   📌这意味着会触发SwapIn线程cpu到gpu的拷贝操作
            #  （就是往self.double_bufs[cur_buf_idx]这个字典中的tensor拷贝）
            self._sync_copyin(layer_id, self.double_bufs[cur_buf_idx], ev_compute)

        # if self.nvprof: nvtx_range_push("L{} Wait(X)".format(layer_id)) 
        # 将is_running置为false，并弹出线程安全队列 get_queue 中的首个元素
        # 若get_queue中没有东西，这里就会一直等待
        cur_named_tensors, ev_swapin = self._wait()
        # 等待swap in完成
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # if self.nvprof: nvtx_range_pop() 
        # pre-swapin next one if exsits
        # 若当前不是最后一个microbatch
        if not is_end:
            # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
            # 若给定了buffer，则该函数直接返回buffer，什么也不做
            # 📌注意第一次运行到这里时，虽然形式上给了buffer(第3个参数)，但本质上传进去的是None
            self.double_bufs[next_buf_idx] = self._allocate(layer_id, named_metas, self.double_bufs[next_buf_idx])
            # 触发SwapIn线程从cpu到gpu的拷贝操作
            # 即当前layer要接收的下一个microbatch运行所需的X
            self._sync_copyin(layer_id, self.double_bufs[next_buf_idx], ev_compute)
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.double_bufs 
            self.double_bufs = [None, None]
        return cur_named_tensors # reference only; to be deleted by runtime
    
    # suc_info：
    # 若后继任务是BWD（非第一个BWD），且输入媒介是MSG，返回 (l(后继任务的首层id), 后级任务输入X的元数据) 。非MSG直接返回None
    # 其他情况直接返回None

    # 1.在默认的计算流上记录一个事件，ev_compute
    # 2.在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
    # 3.
    #   3.1.检测is_running是否为false，即不允许有正在执行的swap in
    #   3.2.将 is_running 置为true
    #   3.3.将(layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，这意味着SwapIn线程会开始cpu到gpu的拷贝操作
    def prefetch_suc(self, suc_info): 
        ''' Prefetc successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (layer_id, named_metas)
        '''  
        if suc_info is None:
            return
        suc_layer_id, suc_named_metas = suc_info
        # if self.nvprof: nvtx_range_push("PrefetchSuc(L{}-X)".format(suc_layer_id)) 
        # must after is_end
        assert self.ubatch_idx == 0 and self.double_bufs == [None, None]
        # record previous compute event for swapin stream to wait
        ev_compute = self.compute_stream.record_event()
        # pre-swapin successor group's 1st ubatch if exsits
        # 在GPU上按照给定的元数据的shape和类型生成一个空tensor，装进named字典返回
        self.double_bufs[0] = self._allocate(suc_layer_id, suc_named_metas, None)
        # 1.检测is_running是否为false，即不允许有正在执行的swap in
        # 2.将 is_running 置为true
        # 3.将(suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，这意味着SwapIn线程会开始cpu到gpu的拷贝操作
        self._sync_copyin(suc_layer_id, self.double_bufs[0], ev_compute)
        # if self.nvprof: nvtx_range_pop() 

class StashOut(object):
    """ Handle offload StashX in vPP/vDP and LocalX (X/dX) in vDP in background thread.
        
        Step:
        1) main thread: allocate pinned memory on CPU
        2) main thread: copy out X in swapout_stream
        3) main thread: optional delete
        4) swapout thread: wait for copy out
        5) swapout thread: isend to downstream ubatchconvert/msgstashx/localx

        Assumption:
        0) stateless
        1) during swap, StashX/X/dX has no grad
        2) FIFO ordering. put layer-by-layer, and swapout layer-by-layer. 
    """
    def __init__(self, output_fn, rank, swapout_stream=None, compute_stream=None, blocking=False, pin_memory=True, nvprof=False): # compute_stream2=None, 
        self.output_fn = output_fn # args: layer_id, named_tensor
        self.rank = rank
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        assert self.rank == torch.cuda.current_device() 
        self.blocking = blocking # 默认为false
        self.pin_memory = pin_memory
        self.nvprof = nvprof
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [ (layer_id, named_tensor, ev_compute), ...]  # between main and swapout thread
        # self.get_queue = threadsafe_data_struct.Queue() # [ ev_swapout, ...] # between swapout and main thread
        self.swapout_thread = threading.Thread(target=self._thread_func)
        self.swapout_thread.daemon = True
        self.swapout_thread.start()
        #
        # print("[SwapOut] rank{} started swapout_thread".format(self.rank))
    
    # flag传进来的时候为None
    #
    # 仅用作等待默认流中的计算完成，不执行卸载
    def offload(self, layer_id, cuda_named_tensors, flag=None):
        ''' Call by main thread. '''
        # record previous compute event for swapout stream to wait
        # 1.记录一个默认计算流上的事件 ev_compute
        ev_compute = self.compute_stream.record_event()
        # Allocate and CopyOut and (Delete)
        # 2.在swapout stream上等待 ev_compute 事件完成，即确保计算完成后才能swapout
        self.compute_stream.wait_event(ev_compute) # Stream waits for this event 
        # if self.nvprof: nvtx_range_pop() 
        # wait in background thread
        # 将 (layer_id, cpu_named_tensors, ev_swapout, flag) 添加到 put_queue 队列中
        self.put_queue.add((layer_id, cuda_named_tensors, flag))
            
    # 确保tensor在完全offload到cpu后，再添加到MSGX线程的send_dict中，发送出去
    # 不断尝试从 put_queue 这个线程安全队列中拿取 layer_id, cpu_named_tensors, ev_swapout, flag，执行：
    # 1.等待ev_swapout事件执行完成
    # 2.调用 output_fn 函数
    #   2.1.调用 MSGstashX 实例的 isend 方法，向 send_dict 这个线程安全字典中的 odict[layer_id] 这个
    #       list添加：self.odict[layer_id].append(named_tensors). 这意味着MSGstashX实例的发送线程将会开始向目标rank
    #       发送 tensor
    #   2.2.调用UBatchSizeConverter的 isend 方法，将layer_id和input2：cpu_named_tensors加入到 input_queue 队列中，这意味
    #       着 UBatchSizeConverter 实例的线程将开始执行tensor大小的转换，而后还是调用MSGstashX的isend方法，将convert好的
    #       tensor列表加入到MSGstashX的send_dict中，后续就与2.1一样了
    #       
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each layer
            layer_id, cuda_named_tensors, flag = self.put_queue.remove()
            # get ready for downstream
            # if self.nvprof: nvtx_range_push("__L{} WaitCopyOut(X)".format(layer_id)) 
            # if MEMCPY_NONBLK: self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. 

            # 起码在vPP的场景下，flag就是None，没发现任何将其置为True的操作
            self.output_fn(layer_id, cuda_named_tensors) if flag is None else \
            self.output_fn(layer_id, cuda_named_tensors, flag)
            # if self.nvprof: nvtx_range_pop() 
