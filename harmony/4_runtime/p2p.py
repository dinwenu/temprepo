# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import gc
from collections import OrderedDict as ODict

import torch
import torch.distributed as dist
from torch.autograd import Variable

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from prof_data_struct import ConstMeta, TensorMeta


class P2PX(object):
    # 使用NCCL处理 X/dx 的P2P发送和接收
    """ Handle P2P send-recv for X/dX of vPP with NCCL

        Feature: 
            0) Stateless recv (GPU memory are free'ed each microbatch) and Stateful prerecv (with double buffering)
            1) NCCL broadcast emulated send-recv
            2) Everything in the "Main" thread 
            3) Send-recv rank pairs use their own groups/communicators/cudaStreams
                --> Deadlock free, Full Duplex, Unblocking send and recv
            4) Support concurrent isend/recv/irecv for multiple tensors 
                *Plan-A: use multiple isend unblockingly (catch: PyTorch send-recv only works for a single tensor --> blocking)
                Plan-B: move multiple isend-recv to background thread, just like Gloo simpleCommHandler (bad: P2P(NCCL) is not thread-safe! )
                Plan-C: concatenate multiple tensors into one big tensor, then isend (catch: extra memory/time for concatenate/split)
            5) Support microbatch size conversion by using 'UBatchSizeConverterP2P'
                last-FWD sending to 1st-BWD needs converting microbatch sizes on GPUs. (catch: extra memory/time for concatenate/split)
            6) Built-in cuda event:
                <cudaEventRecord & cudaStreamWaitEvent> -> 
                launch nccl kernel (broadcast) -> 
                <cudaEventRecord, cudaEventCreateWithFlags, cudaEventRecord, cudaStreamWaitEvent> (ireq.wait)

        Assumption:
            0) distributed environment has already been initialized with gloo
            1) during send-recv, X/dX has no grad (after send-recv, can set them to requires_grad for BWD)
    """
    # 1.访问每一个rank，在每个rank和其下一个rank间建立一个NCCL通信组。若当前rank包含在正在建立的通信组中，
    #   就为字典 self.groups 添加一个值：{ "r1->r2": dist.group_obj }
    # 2.
    #   2.1.创建一个包含单个元素的张量，用于初始化 NCCL 通信器
    #   2.2.将当前rank所在的通信组取出，进行一次r1->r2的点对点通信
    def __init__(self, rank, world_size, reverse_bwd=True, verbose=False, nvprof=False):
        assert dist.get_backend() == "gloo"
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)
        # build two-process group for NCCL broadcast (r1, r2)  
        self.groups = ODict() # { "r1->r2": dist.group_obj }
        # in-order round-robin
        # 访问每一个rank，在每个rank和其下一个rank间建立一个NCCL通信组。若当前rank包含在正在建立的通信组中，
        # 就为字典 self.groups 添加一个值：{ "r1->r2": dist.group_obj }
        for r1 in range(self.world_size):
            # r2即r1后面那个rank
            r2 = (r1+1) % self.world_size
            pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
            # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
            # This new_group only creates empty NCCL groups (communicator and cudaStream are not initialized yet)
            if self.rank in [r1,r2]:
                self.groups["{}->{}".format(r1,r2)] = pgroup
        # reverse round-robin
        # 该参数默认为false，不用管
        if reverse_bwd:
            for r1 in range(self.world_size):
                r2 = (r1-1) % self.world_size
                pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
                if self.rank in [r1,r2]:
                    self.groups["{}->{}".format(r1,r2)] = pgroup

        ################################################
        ############## 手动设置PP通信组 #################
        # r1 = 0
        # r2 = 3
        # pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
        # if self.rank in [r1,r2]:
        #     self.groups["{}->{}".format(r1,r2)] = pgroup
        ################################################

        # print("[P2P] rank={}, world_size={}, self.groups = {}".format(self.rank, self.world_size, self.groups))
        # initialize NCCL communicator and its cudaStream in mainthread
        # 创建一个包含单个元素的张量，用于初始化 NCCL 通信器
        tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda:%d"%(self.rank))
        # 将当前rank所在的通信组取出，进行一次r1->r2的点对点通信
        for key, group in self.groups.items():
            dist.broadcast(tensor, group=group, src=int(key.split("->")[0])) # init communicator should be in-order
            # print("[P2P] rank={} init'ed NCCL communicator[{}] and its cudaStream".format(self.rank, key))
        # clean up
        del tensor; gc.collect(); torch.cuda.empty_cache()
        # for pre-recv
        self.is_irecving = False
        self.ubatch_idx = int(0)
        self.double_bufs = [None, None]
        # for bytes counter
        if self.verbose: 
            self.send_byte_cnt = 0
            self.recv_byte_cnt = 0

    # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
    @torch.no_grad()
    def _isend_tensor(self, tensor, dst):
        """ Non-Blocking send a tensor via NCCL broadcast """
        # print("[P2P]\trank{}: _isend_tensor({},{}) to dst:{}".format(self.rank, tensor.shape, tensor.dtype, dst))
        assert tensor.is_cuda
        group_key = "{}->{}".format(self.rank,dst)
        ireq = dist.broadcast(tensor, src=self.rank, group=self.groups[group_key], async_op=True)
        # print("[P2P]\trank{}: _isend_tensor'ed".format(self.rank))
        # tensor.nelement():返回张量中元素的总数
        # tensor.element_size():返回张量中每个元素的字节数
        if self.verbose: self.send_byte_cnt += tensor.nelement()*tensor.element_size()
        return ireq
    
    # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
    # 2.取出以src作为发射源的通信组，接收从源节点src发送到当前节点的tensor，并且以非阻塞的方式执行操作
    # 返回 tensor，ireq（异步的工作句柄）
    @torch.no_grad()
    def _irecv_tensor(self, tensor=None, shape=None, dtype=torch.float32, src=-1):
        """ Non-Blocking recv a tensor via NCCL broadcast.
            If tensor is None, then its shape (e.g. () or (1,) or (2,2)) must be given to create a tensor, to receive, and to return this GPU tensor. 
            Else, return filled GPU tensor.

            case-1: _irecv_tensor(shape=(1,2), src=123) # create new
            case-2: _irecv_tensor(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and shape is not None) or (tensor is not None and shape is None)
        # 若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor
        tensor = torch.empty(shape, dtype=dtype, device="cuda:%d"%self.rank) if tensor is None else tensor
        assert tensor.is_cuda
        # print("[P2P]\trank{}: _irecv_tensor({},{}) from src:{}".format(self.rank, tensor.shape, tensor.dtype, src))
        # 取出点对点的广播通信组
        group_key = "{}->{}".format(src, self.rank)
        # 接收从源节点src发送到当前节点的tensor，并且以非阻塞的方式执行操作
        # 若当前rank是接收rank，这个tensor就是保存接收数据的tensor
        ireq = dist.broadcast(tensor, src=src, group=self.groups[group_key], async_op=True)
        # ireq.wait() # blocking
        # print("[P2P]\trank{}: _irecv_tensor'ed".format(self.rank))
        if self.verbose: self.recv_byte_cnt += tensor.nelement()*tensor.element_size()
        return tensor, ireq
    
    @torch.no_grad()
    def _isend_const(self, const, dst):
        """ Non-Blocking send a const int/float via NCCL """
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        assert isinstance(const, (int,float))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank)
        ireq = self._isend_tensor(tensor, dst) # """ nccl unblocking isend works better than gloo """
        del tensor # It works
        # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
        return ireq

    @torch.no_grad()
    def _irecv_const(self, tensor=None, const=None, src=-1):
        """ Non-Blocking send a const scalar via NCCL.
            If tensor is None, then const must be given (int/float) to create a tensor, to receive, and to return this GPU tensor.
            Else, return filled GPU tensor.
            
            case-1: _irecv_const(const=123, src=123) # create new
            case-2: _irecv_const(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and isinstance(const, (int,float))) or (tensor is not None and const is None)
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank) if tensor is None else tensor
        tensor, ireq = self._irecv_tensor(tensor=tensor, src=src)
        # ireq.wait() # blocking
        # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
        return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number, once received
        
    # @torch.no_grad() # Deprecated
    # def _isend_const(self, const, dst, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     ireq = dist.isend(torch.tensor(const), dst=dst, tag=tag)
    #     ireq.wait() 
    #     """ must blocking for const, otherwise the CPU const can be deleted too soon, causing irecv wait forever """
    #     """ But if really blocks, it causes P2P deadlock. """
    #     # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
    #     return ireq

    # @torch.no_grad() # Deprecated
    # def _irecv_const(self, const, src, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     tensor = torch.tensor(const, device="cpu", pin_memory=True)
    #     ireq = dist.irecv(tensor, src=src, tag=tag)
    #     # ireq.wait() # blocking
    #     # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
    #     return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number
    
    # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
    def isend(self, named_tensors, dst):
        ''' Call by main thread. Nonblocking send. '''    
        # print("[P2P]\trank{}: isend entered".format(self.rank))
        for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
            if isinstance(tensor, (torch.Tensor,Variable)):
                # 确保tensor在GPU上，且不需要梯度
                assert tensor.is_cuda and not tensor.requires_grad
                # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
                self._isend_tensor(tensor, dst)
            elif isinstance(tensor, (float,int)):
                self._isend_const(tensor, dst)
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                for t in tensor:
                    assert t.is_cuda and not t.requires_grad
                    self._isend_tensor(t, dst)
            else:
                raise ValueError("unknown tensor={}".format(tensor))
            # print("[P2P]\trank{}: isend'ed {}:{} to dst:{}".format(self.rank, name, type(tensor), dst))
    
    # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行
    # 2 等待P2P通信完成，即已经接收到src发送过来的tensor
    # 3.以字典的形式返回接收到的tensor（named_tensors）
    def recv(self, named_metas, src=-1):
        ''' Call by main thread. Blocking recv. 
            Assumption: 
                0) Stateless: always create allocate new tensor, and let it delete by runtime (no double buffering)
                1) Blocking compute stream by cuda events
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
            Return: recved named_tensors. '''    
        # print("[P2P]\trank{}: recv entered".format(self.rank))
        named_tensors = ODict()
        named_ireq = ODict()

        # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
                # 2.取出以src作为发射源的通信组，接收从src发送过来的tensor，以非阻塞的方式执行操作
                # 返回 tensor，ireq（异步的工作句柄）
                named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
            elif isinstance(meta, ConstMeta):
                named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor, tmp_ireq = [], []
                for m in meta:
                    tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                    tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                named_tensors[name] = tmp_tensor
                named_ireq[name] = tmp_ireq
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's irecv'ed {}:{}".format(self.rank, name, meta))

        # wait all tensors recved (by built-in cuda event)
        # 2.等待P2P通信完成，即已经接收到src发送过来的tensor
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number # let python do the gc on cuda tensor
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                # for ireq, tensor in zip(named_ireq[name],named_tensors[name]):
                #     ireq.wait()
                [ ireq.wait() for ireq in named_ireq[name] ]  
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's ireq.waited {}:{} from src:{}".format(self.rank, name, meta, src))
        # print("[P2P]\trank{}: recv's all ireqs waited".format(self.rank))
        # clean up

        # 3.以字典的形式返回接收到的tensor（named_tensors）
        return named_tensors
    
    # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
    # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
    def _irecv(self, named_metas, src, buffer=None): 
        ''' Non-Blocking recv. 
            Assumption: only one irecv can be running.
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      buffer = None or (named_tensors, named_ireq)
            Return: created or input (named_tensors, named_ireq)
        '''
        assert not self.is_irecving, "the irecv is still running"
        self.is_irecving = True
        # print("[P2P]\trank{}: _irecv'ing".format(self.rank))

        # 若没有给定buffer，在_irecv_tensor函数的执行过程中，在GPU上创建一个新的tensor用于接收发过来的tensor
        if buffer is None: # allocate new tensors
            if self.nvprof: nvtx_range_push("P2PIn Alloc & iBcast") 
            named_tensors = ODict()
            named_ireq = ODict()
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
                    # 2.取出以src作为发射源的通信组，接收从src发送过来的tensor，以非阻塞的方式执行操作
                    # 返回 tensor，ireq（异步的工作句柄）
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    tmp_tensor, tmp_ireq = [], []
                    for m in meta:
                        tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                        tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                    named_tensors[name] = tmp_tensor
                    named_ireq[name] = tmp_ireq
                else:
                    raise ValueError("unknown meta={}".format(meta))
            if self.nvprof: nvtx_range_pop() 
            # print("[P2P]\trank{}: _irecv allocated new tensors and requested all".format(self.rank))

        # 若给定了buffer，则使用之前创建的tensor(已经接收过数据了)接收发过来的tensor
        else: # reuse buffered tensors
            named_tensors, named_ireq = buffer
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(tensor=named_tensors[name], src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(tensor=named_tensors[name], src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    assert isinstance(named_tensors[name], list) and isinstance(named_ireq[name], list)
                    for i in range(len(meta)):
                        named_tensors[name][i], named_ireq[name][i] = self._irecv_tensor(tensor=named_tensors[name][i], src=src)
                else:
                    raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: _irecv reused buffered tensors and requested all".format(self.rank))
        return named_tensors, named_ireq
        
    # 调用buffer中保存的异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
    def _wait_irecv(self, named_metas, buffer=None): 
        ''' Wait for the running irecv. 
            Assumption: only one irecv can be running.
            Arguments: the same as _irecv, except buffer is not None
            Return: recved named_tensors.
        '''
        assert self.is_irecving, "no running irecv"
        self.is_irecving = False
        # wait all tensors recved
        assert buffer is not None
        named_tensors, named_ireq = buffer
        recved_named_tensors = ODict() # ref to buffer
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
                recved_named_tensors[name] = named_tensors[name] # ref
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                # named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number
                recved_named_tensors[name] = named_tensors[name].item()
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor = []
                for tensor, ireq in zip(named_tensors[name], named_ireq[name]):
                    ireq.wait()
                    tmp_tensor.append(tensor)
                recved_named_tensors[name] = tmp_tensor # ref
            else:
                raise ValueError("unknown meta={}".format(meta))
        return recved_named_tensors 

    # is_end：若当前是vt的最后一个micro batch，就为True
    #
    # 1.若没有正在执行的接收操作，则从src rank上接收发过来的tensor
    #   --否则，若有正在执行的接收操作，直接执行2
    # 2.调用异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
    # 3.若当前不是最后一个Micro batch，异步的接收下一个microbatch运行时需要的X
    def prerecv(self, named_metas, src, is_end=False): 
        ''' Call by main thread. Blocking recv current one and unblocking pre-recv next one. 
            Assumption: 
                    1) use double buffering for parallel compute and irecv
                    2) double buffering doesn't change in shape TODO: fix
                    3) use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                    4) no prerecv for successor group
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      is_end = whether ends prerecv after current recv
            Return: recved current named_tensors. '''    
        #
        # 0
        cur_buf_idx = self.ubatch_idx % 2 # for compute
        # 1
        next_buf_idx = (self.ubatch_idx+1) % 2 # for pre irecv

        # wait current one (if no current one, _irecv & _wait_irecv one)
        # 若没有正在执行的接收操作，则从src rank上接收发过来的tensor
        # 📌从第二个microbatch开始，这里才会略过，因为第一个Micorbatch预取了第二个的X
        # 因此if为false，直接执行下面的wait等待接收完毕（也有可能已经接收完了）
        if not self.is_irecving:
            # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
            # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
            self.double_bufs[cur_buf_idx] = self._irecv(named_metas, src, None)
        # 调用buffer中保存的异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
        cur_named_tensors = self._wait_irecv(named_metas, self.double_bufs[cur_buf_idx])
        # irecv next one if exists
        # 若当前不是最后一个Micro batch，
        if not is_end:
            # ❓是不是代码写错了，若第二个参数的下标是next_buf_idx，就不算复用已有的tensor了
            # 肯定写错了，self.double_bufs[next_buf_idx]中此时没有tesnor，为空在_irecv_tensor中就得生成一个空tensor
            # 但调用 _irecv_tensor 时根本没给shape
            # 此外，这里应该接收下一个 named_metas，这里传进去的还是当前的 named_metas，对同一个数据重新接收了一遍
            # 📌这里第2个参数传进去的实际还是None，内部还是会正确的执行
            # 📌24/7/9：没写错，这里不是为下一个vt接收输入X，而是为下一个microbatch的执行接收需要的X
            self.double_bufs[next_buf_idx] = self._irecv(named_metas, src, self.double_bufs[next_buf_idx])
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.double_bufs 
            self.double_bufs = [None, None]            
        return cur_named_tensors # reference only; to be deleted by runtime

    # suc_info：
    # 为后继任务准备输入信息，后继为FWD则准备输入X，后继为BWD则准备输入dY
    # 两种情况：
    # 1.后继任务是FWD任务，或第一个BWD任务（包含计算loss层），为其准备输入X的元数据以及（来源）媒介
    #   两种情况：
    #   1.1.后继任务接收X的媒介不为P2P，直接返回None
    #   1.2.否则，返回 (suc_vt输入X的meta，suc_vt接收X的src_rank)
    # 2.后继任务是BWD任务，则返回后继BWD任务dY的元数据以及（来源）媒介
    #   两种情况：
    #   2.1.若当前BWD的后继BWD任务接收dY的媒介不为P2P，返回None
    #   2.2.否则，返回suc_vt的前驱BWD任务首层的接收X的元数据，{ name:TensorMeta }，来代表suc_vt最后一层接收dY的元数据

    # 提前使用P2P方法接收后继任务的输入X/dY
    def prerecv_suc(self, suc_info): 
        ''' Prerecv successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (named_metas, src_rank)
        '''  
        # 若后继任务的信息为None，直接返回
        if suc_info is None:
            return
        # (suc_vt输入X的meta，suc_vt首层的media)
        suc_named_metas, suc_src = suc_info
        # print("\trank{}: P2P.prerecv_suc({}, src{})".format(self.rank, suc_named_metas, suc_src))
        # must after is_end
        assert self.ubatch_idx == 0 and self.double_bufs == [None, None]
        # record previous compute event (built-in)
        # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
        # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
        self.double_bufs[0] = self._irecv(suc_named_metas, suc_src, buffer=None)



class P2PX_2(object):
    # 使用NCCL处理 X/dx 的P2P发送和接收
    """ Handle P2P send-recv for X/dX of vPP with NCCL

        Feature: 
            0) Stateless recv (GPU memory are free'ed each microbatch) and Stateful prerecv (with double buffering)
            1) NCCL broadcast emulated send-recv
            2) Everything in the "Main" thread 
            3) Send-recv rank pairs use their own groups/communicators/cudaStreams
                --> Deadlock free, Full Duplex, Unblocking send and recv
            4) Support concurrent isend/recv/irecv for multiple tensors 
                *Plan-A: use multiple isend unblockingly (catch: PyTorch send-recv only works for a single tensor --> blocking)
                Plan-B: move multiple isend-recv to background thread, just like Gloo simpleCommHandler (bad: P2P(NCCL) is not thread-safe! )
                Plan-C: concatenate multiple tensors into one big tensor, then isend (catch: extra memory/time for concatenate/split)
            5) Support microbatch size conversion by using 'UBatchSizeConverterP2P'
                last-FWD sending to 1st-BWD needs converting microbatch sizes on GPUs. (catch: extra memory/time for concatenate/split)
            6) Built-in cuda event:
                <cudaEventRecord & cudaStreamWaitEvent> -> 
                launch nccl kernel (broadcast) -> 
                <cudaEventRecord, cudaEventCreateWithFlags, cudaEventRecord, cudaStreamWaitEvent> (ireq.wait)

        Assumption:
            0) distributed environment has already been initialized with gloo
            1) during send-recv, X/dX has no grad (after send-recv, can set them to requires_grad for BWD)
    """
    # 1.访问每一个rank，在每个rank和其下一个rank间建立一个NCCL通信组。若当前rank包含在正在建立的通信组中，
    #   就为字典 self.groups 添加一个值：{ "r1->r2": dist.group_obj }
    # 2.
    #   2.1.创建一个包含单个元素的张量，用于初始化 NCCL 通信器
    #   2.2.将当前rank所在的通信组取出，进行一次r1->r2的点对点通信
    def __init__(self, rank, world_size, reverse_bwd=True, verbose=False, nvprof=False):
        assert dist.get_backend() == "gloo"
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)
        # build two-process group for NCCL broadcast (r1, r2)  
        self.groups = ODict() # { "r1->r2": dist.group_obj }
        # in-order round-robin
        # 访问每一个rank，在每个rank和其下一个rank间建立一个NCCL通信组。若当前rank包含在正在建立的通信组中，
        # 就为字典 self.groups 添加一个值：{ "r1->r2": dist.group_obj }
        for r1 in range(self.world_size):
            # r2即r1后面那个rank
            r2 = (r1+1) % self.world_size
            pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
            # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
            # This new_group only creates empty NCCL groups (communicator and cudaStream are not initialized yet)
            if self.rank in [r1,r2]:
                self.groups["{}->{}".format(r1,r2)] = pgroup
        # reverse round-robin
        # 该参数默认为false，不用管
        if reverse_bwd:
            for r1 in range(self.world_size):
                r2 = (r1-1) % self.world_size
                pgroup = dist.new_group(ranks=[r1, r2], backend='nccl')
                if self.rank in [r1,r2]:
                    self.groups["{}->{}".format(r1,r2)] = pgroup
        # print("[P2P] rank={}, world_size={}, self.groups = {}".format(self.rank, self.world_size, self.groups))
        # initialize NCCL communicator and its cudaStream in mainthread
        # 创建一个包含单个元素的张量，用于初始化 NCCL 通信器
        tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda:%d"%(self.rank))
        # 将当前rank所在的通信组取出，进行一次r1->r2的点对点通信
        for key, group in self.groups.items():
            dist.broadcast(tensor, group=group, src=int(key.split("->")[0])) # init communicator should be in-order
            # print("[P2P] rank={} init'ed NCCL communicator[{}] and its cudaStream".format(self.rank, key))
        # clean up
        del tensor; gc.collect(); torch.cuda.empty_cache()
        # for pre-recv
        self.is_irecving = False
        self.ubatch_idx = int(0)
        self.triple_bufs = [None, None, None]
        # for bytes counter
        if self.verbose: 
            self.send_byte_cnt = 0
            self.recv_byte_cnt = 0

    # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
    @torch.no_grad()
    def _isend_tensor(self, tensor, dst):
        """ Non-Blocking send a tensor via NCCL broadcast """
        # print("[P2P]\trank{}: _isend_tensor({},{}) to dst:{}".format(self.rank, tensor.shape, tensor.dtype, dst))
        assert tensor.is_cuda
        group_key = "{}->{}".format(self.rank,dst)
        # print(f"rank:{self.rank}, 发送的tensor为:{tensor}")
        ireq = dist.broadcast(tensor, src=self.rank, group=self.groups[group_key], async_op=True)
        # print("[P2P]\trank{}: _isend_tensor'ed".format(self.rank))
        # tensor.nelement():返回张量中元素的总数
        # tensor.element_size():返回张量中每个元素的字节数
        if self.verbose: self.send_byte_cnt += tensor.nelement()*tensor.element_size()
        return ireq
    
    # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
    # 2.取出以src作为发射源的通信组，接收从源节点src发送到当前节点的tensor，并且以非阻塞的方式执行操作
    # 返回 tensor，ireq（异步的工作句柄）
    @torch.no_grad()
    def _irecv_tensor(self, tensor=None, shape=None, dtype=torch.float32, src=-1):
        """ Non-Blocking recv a tensor via NCCL broadcast.
            If tensor is None, then its shape (e.g. () or (1,) or (2,2)) must be given to create a tensor, to receive, and to return this GPU tensor. 
            Else, return filled GPU tensor.

            case-1: _irecv_tensor(shape=(1,2), src=123) # create new
            case-2: _irecv_tensor(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and shape is not None) or (tensor is not None and shape is None)
        # 若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor
        tensor = torch.empty(shape, dtype=dtype, device="cuda:%d"%self.rank) if tensor is None else tensor
        assert tensor.is_cuda
        # print("[P2P]\trank{}: _irecv_tensor({},{}) from src:{}".format(self.rank, tensor.shape, tensor.dtype, src))
        # 取出点对点的广播通信组
        group_key = "{}->{}".format(src, self.rank)
        # 接收从源节点src发送到当前节点的tensor，并且以非阻塞的方式执行操作
        # 若当前rank是接收rank，这个tensor就是保存接收数据的tensor
        ireq = dist.broadcast(tensor, src=src, group=self.groups[group_key], async_op=True)
        # 加上这个没用，一样收不到，不像是被覆盖的问题
        # ireq.wait() # blocking
        # print(f"rank:{self.rank}, 收到的tensor为:{tensor}")
        # print("[P2P]\trank{}: _irecv_tensor'ed".format(self.rank))
        if self.verbose: self.recv_byte_cnt += tensor.nelement()*tensor.element_size()
        return tensor, ireq
    
    @torch.no_grad()
    def _isend_const(self, const, dst):
        """ Non-Blocking send a const int/float via NCCL """
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        assert isinstance(const, (int,float))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank)
        ireq = self._isend_tensor(tensor, dst) # """ nccl unblocking isend works better than gloo """
        del tensor # It works
        # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
        return ireq

    @torch.no_grad()
    def _irecv_const(self, tensor=None, const=None, src=-1):
        """ Non-Blocking send a const scalar via NCCL.
            If tensor is None, then const must be given (int/float) to create a tensor, to receive, and to return this GPU tensor.
            Else, return filled GPU tensor.
            
            case-1: _irecv_const(const=123, src=123) # create new
            case-2: _irecv_const(tensor=cuda_tensor, src=123) # reuse existing
        """
        assert (tensor is None and isinstance(const, (int,float))) or (tensor is not None and const is None)
        # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
        tensor = torch.tensor(const, device="cuda:%d"%self.rank) if tensor is None else tensor
        tensor, ireq = self._irecv_tensor(tensor=tensor, src=src)
        # ireq.wait() # blocking
        # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
        return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number, once received
        
    # @torch.no_grad() # Deprecated
    # def _isend_const(self, const, dst, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     ireq = dist.isend(torch.tensor(const), dst=dst, tag=tag)
    #     ireq.wait() 
    #     """ must blocking for const, otherwise the CPU const can be deleted too soon, causing irecv wait forever """
    #     """ But if really blocks, it causes P2P deadlock. """
    #     # print("[P2PHandler] rank={}: dist.broadcast'ed".format(self.rank))
    #     return ireq

    # @torch.no_grad() # Deprecated
    # def _irecv_const(self, const, src, tag=7777777):
    #     """ Non-Blocking send a const int/float via Gloo """
    #     # print("[P2PHandler] rank={}: isend(tensor={} with shape={})".format(self.rank, tensor, tensor.shape))
    #     assert isinstance(const, (int,float))
    #     tensor = torch.tensor(const, device="cpu", pin_memory=True)
    #     ireq = dist.irecv(tensor, src=src, tag=tag)
    #     # ireq.wait() # blocking
    #     # print("[rank{}]\tP2PX._irecv_const: irecv'ed {}".format(self.rank, tensor))
    #     return tensor, ireq # Need tensor.item() to convert a 0-dim tensor to a python number
    
    # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
    def isend(self, named_tensors, dst):
        ''' Call by main thread. Nonblocking send. '''    
        # print("[P2P]\trank{}: isend entered".format(self.rank))
        for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
            if isinstance(tensor, (torch.Tensor,Variable)):
                # 确保tensor在GPU上，且不需要梯度
                assert tensor.is_cuda and not tensor.requires_grad
                # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
                self._isend_tensor(tensor, dst)
            elif isinstance(tensor, (float,int)):
                self._isend_const(tensor, dst)
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                for t in tensor:
                    assert t.is_cuda and not t.requires_grad
                    self._isend_tensor(t, dst)
            else:
                raise ValueError("unknown tensor={}".format(tensor))
            # print("[P2P]\trank{}: isend'ed {}:{} to dst:{}".format(self.rank, name, type(tensor), dst))
    
    # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行
    # 2 等待P2P通信完成，即已经接收到src发送过来的tensor
    # 3.以字典的形式返回接收到的tensor（named_tensors）
    def recv(self, named_metas, src=-1):
        ''' Call by main thread. Blocking recv. 
            Assumption: 
                0) Stateless: always create allocate new tensor, and let it delete by runtime (no double buffering)
                1) Blocking compute stream by cuda events
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
            Return: recved named_tensors. '''    
        # print("[P2P]\trank{}: recv entered".format(self.rank))
        named_tensors = ODict()
        named_ireq = ODict()

        # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
                # 2.取出以src作为发射源的通信组，接收从src发送过来的tensor，以非阻塞的方式执行操作
                # 返回 tensor，ireq（异步的工作句柄）
                named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
            elif isinstance(meta, ConstMeta):
                named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor, tmp_ireq = [], []
                for m in meta:
                    tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                    tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                named_tensors[name] = tmp_tensor
                named_ireq[name] = tmp_ireq
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's irecv'ed {}:{}".format(self.rank, name, meta))

        # wait all tensors recved (by built-in cuda event)
        # 2.等待P2P通信完成，即已经接收到src发送过来的tensor
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number # let python do the gc on cuda tensor
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                # for ireq, tensor in zip(named_ireq[name],named_tensors[name]):
                #     ireq.wait()
                [ ireq.wait() for ireq in named_ireq[name] ]  
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: recv's ireq.waited {}:{} from src:{}".format(self.rank, name, meta, src))
        # print("[P2P]\trank{}: recv's all ireqs waited".format(self.rank))
        # clean up

        # 3.以字典的形式返回接收到的tensor（named_tensors）
        # for name, meta in named_metas.items():
        #     print(f"rank:{self.rank}, name:{name}, 收到的tensor为:{named_tensors[name]}")
        return named_tensors
    
    # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
    # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
    def _irecv(self, named_metas, src, buffer=None): 
        ''' Non-Blocking recv. 
            Assumption: only one irecv can be running.
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      buffer = None or (named_tensors, named_ireq)
            Return: created or input (named_tensors, named_ireq)
        '''
        assert not self.is_irecving, "the irecv is still running"
        self.is_irecving = True
        # print("[P2P]\trank{}: _irecv'ing".format(self.rank))

        # 若没有给定buffer，在_irecv_tensor函数的执行过程中，在GPU上创建一个新的tensor用于接收发过来的tensor
        if buffer is None: # allocate new tensors
            if self.nvprof: nvtx_range_push("P2PIn Alloc & iBcast") 
            named_tensors = ODict()
            named_ireq = ODict()
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    # 1.若没有给定tensor，在当前GPU上按照给定的shape和dtype创建一个空tensor，用于保存接收到的tensor
                    # 2.取出以src作为发射源的通信组，接收从src发送过来的tensor，以非阻塞的方式执行操作
                    # 返回 tensor，ireq（异步的工作句柄）
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(shape=meta.shape, dtype=meta.dtype, src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(const=meta.const, src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    tmp_tensor, tmp_ireq = [], []
                    for m in meta:
                        tensor, ireq = self._irecv_tensor(shape=m.shape, dtype=m.dtype, src=src)
                        tmp_tensor.append(tensor); tmp_ireq.append(ireq)
                    named_tensors[name] = tmp_tensor
                    named_ireq[name] = tmp_ireq
                else:
                    raise ValueError("unknown meta={}".format(meta))
            if self.nvprof: nvtx_range_pop() 
            # print("[P2P]\trank{}: _irecv allocated new tensors and requested all".format(self.rank))

        # 若给定了buffer，则使用之前创建的tensor(已经接收过数据了)接收发过来的tensor
        else: # reuse buffered tensors
            named_tensors, named_ireq = buffer
            for name, meta in named_metas.items():
                if isinstance(meta, TensorMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_tensor(tensor=named_tensors[name], src=src)
                elif isinstance(meta, ConstMeta):
                    named_tensors[name], named_ireq[name] = self._irecv_const(tensor=named_tensors[name], src=src)
                elif isinstance(meta, list): # output tuple of bert pretrainhead 
                    assert isinstance(named_tensors[name], list) and isinstance(named_ireq[name], list)
                    for i in range(len(meta)):
                        named_tensors[name][i], named_ireq[name][i] = self._irecv_tensor(tensor=named_tensors[name][i], src=src)
                else:
                    raise ValueError("unknown meta={}".format(meta))
            # print("[P2P]\trank{}: _irecv reused buffered tensors and requested all".format(self.rank))
        return named_tensors, named_ireq
        
    # 调用buffer中保存的异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
    def _wait_irecv(self, named_metas, buffer=None): 
        ''' Wait for the running irecv. 
            Assumption: only one irecv can be running.
            Arguments: the same as _irecv, except buffer is not None
            Return: recved named_tensors.
        '''
        assert self.is_irecving, "no running irecv"
        self.is_irecving = False
        # wait all tensors recved
        assert buffer is not None
        named_tensors, named_ireq = buffer
        recved_named_tensors = ODict() # ref to buffer
        for name, meta in named_metas.items():
            if isinstance(meta, TensorMeta):
                named_ireq[name].wait()
                recved_named_tensors[name] = named_tensors[name] # ref
            elif isinstance(meta, ConstMeta):
                named_ireq[name].wait()
                # named_tensors[name] = named_tensors[name].item() # convert a 0-dim cpu/cuda tensor to a python number
                recved_named_tensors[name] = named_tensors[name].item()
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                tmp_tensor = []
                for tensor, ireq in zip(named_tensors[name], named_ireq[name]):
                    ireq.wait()
                    tmp_tensor.append(tensor)
                recved_named_tensors[name] = tmp_tensor # ref
            else:
                raise ValueError("unknown meta={}".format(meta))
        return recved_named_tensors 

    # is_end：若当前是vt的最后一个micro batch，就为True
    #
    # 1.若没有正在执行的接收操作，则从src rank上接收发过来的tensor
    #   --否则，若有正在执行的接收操作，直接执行2
    # 2.调用异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
    # 3.若当前不是最后一个Micro batch，异步的接收下一个microbatch运行时需要的X
    def prerecv(self, named_metas, src, is_end=False): 
        ''' Call by main thread. Blocking recv current one and unblocking pre-recv next one. 
            Assumption: 
                    1) use double buffering for parallel compute and irecv
                    2) double buffering doesn't change in shape TODO: fix
                    3) use cudaEvent to sync with compute stream, and event calls are still on CPU async w.r.t GPU streams
                    4) no prerecv for successor group
            Argument: named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
                      src = src rank
                      is_end = whether ends prerecv after current recv
            Return: recved current named_tensors. '''    
        #
        cur_buf_idx = self.ubatch_idx % 3  # for compute
        next_buf_idx = (self.ubatch_idx + 1) % 3  # for pre irecv
        prev_buf_idx = (self.ubatch_idx + 2) % 3  # buffer to free

        # wait current one (if no current one, _irecv & _wait_irecv one)
        # 若没有正在执行的接收操作，则从src rank上接收发过来的tensor
        # 📌从第二个microbatch开始，这里才会略过，因为第一个Micorbatch预取了第二个的X
        # 因此if为false，直接执行下面的wait等待接收完毕（也有可能已经接收完了）
        if not self.is_irecving:
            # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
            # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
            self.triple_bufs[cur_buf_idx] = self._irecv(named_metas, src, None)
        # 调用buffer中保存的异步句柄的wait函数，等待P2P通信完成，以named_tensor的形式返回接收到的tensor
        cur_named_tensors = self._wait_irecv(named_metas, self.triple_bufs[cur_buf_idx])
        # irecv next one if exists
        # 若当前不是最后一个Micro batch，
        if not is_end:
            # ❓是不是代码写错了，若第二个参数的下标是next_buf_idx，就不算复用已有的tensor了
            # 肯定写错了，self.double_bufs[next_buf_idx]中此时没有tesnor，为空在_irecv_tensor中就得生成一个空tensor
            # 但调用 _irecv_tensor 时根本没给shape
            # 此外，这里应该接收下一个 named_metas，这里传进去的还是当前的 named_metas，对同一个数据重新接收了一遍
            # 📌这里第2个参数传进去的实际还是None，内部还是会正确的执行
            # 📌24/7/9：没写错，这里不是为下一个vt接收输入X，而是为下一个microbatch的执行接收需要的X
            self.triple_bufs[next_buf_idx] = self._irecv(named_metas, src, self.triple_bufs[next_buf_idx])
            self.ubatch_idx += 1
        else: # clean up
            self.ubatch_idx = 0
            del self.triple_bufs 
            self.triple_bufs = [None, None, None]            
        return cur_named_tensors # reference only; to be deleted by runtime

    # suc_info：
    # 为后继任务准备输入信息，后继为FWD则准备输入X，后继为BWD则准备输入dY
    # 两种情况：
    # 1.后继任务是FWD任务，或第一个BWD任务（包含计算loss层），为其准备输入X的元数据以及（来源）媒介
    #   两种情况：
    #   1.1.后继任务接收X的媒介不为P2P，直接返回None
    #   1.2.否则，返回 (suc_vt输入X的meta，suc_vt接收X的src_rank)
    # 2.后继任务是BWD任务，则返回后继BWD任务dY的元数据以及（来源）媒介
    #   两种情况：
    #   2.1.若当前BWD的后继BWD任务接收dY的媒介不为P2P，返回None
    #   2.2.否则，返回suc_vt的前驱BWD任务首层的接收X的元数据，{ name:TensorMeta }，来代表suc_vt最后一层接收dY的元数据

    # 提前使用P2P方法接收后继任务的输入X/dY
    def prerecv_suc(self, suc_info): 
        ''' Prerecv successor group's 1st ubatch if exists. Call by main thread. 
            Assumption: same 1) 3) above
            Argument: suc_info = None or successor group's (named_metas, src_rank)
        '''  
        # 若后继任务的信息为None，直接返回
        if suc_info is None:
            return
        # (suc_vt输入X的meta，suc_vt首层的media)
        suc_named_metas, suc_src = suc_info
        # print("\trank{}: P2P.prerecv_suc({}, src{})".format(self.rank, suc_named_metas, suc_src))
        # must after is_end
        assert self.ubatch_idx == 0 and self.triple_bufs == [None, None, None]
        # record previous compute event (built-in)
        # 1.在以src作为发射源的通信组上，接收从src发送（广播）过来的tensor，以非阻塞的方式执行。返回 tensor，ireq（异步的工作句柄）
        # 2.以字典的形式返回接收到的tensor（named_tensors）和异步工作句柄
        self.triple_bufs[0] = self._irecv(suc_named_metas, suc_src, buffer=None)


# 直接暂存输出的Y/dX，给同一个GPU上的下一个vt用
# 📌不用管ubatshsizeconverter，这个是给相邻的FWDvt/BWDvt用的，跟FWD->BWD的stashX没关系
class CacheX(object):
    def __init__(self, rank, world_size, reverse_bwd=True, verbose=False, nvprof=False):
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.nvprof = nvprof
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)

        # 底层存储结构
        self.self_dict = ODict()

    def fetch(self, layer_id):
        # return self.self_dict.pop(layer_id)
        print(f"rank:{self.rank},取出layer{layer_id}对应的cuda tensor")
        return self.self_dict[layer_id].pop(0)

    def put(self, layer_id, cuda_named_tensors):
        # self.self_dict[layer_id] = cuda_named_tensors
        if layer_id not in self.self_dict:
            self.self_dict[layer_id] = []
        print(f"rank:{self.rank}，为layer{layer_id}装入了一个cuda tensor")
        self.self_dict[layer_id].append(cuda_named_tensors)

    # def init_layer_ids(self, layer_ids): # always ascending
    #     assert isinstance(layer_ids,list)
    #     for id in sorted(layer_ids): 
    #         self.odict[id] = []
    #     self.layer_ids = list(self.odict.keys())



class P2PModel(object):
    """ Handle P2P allreduce for dW/B of vDP with NCCL 
        
        Feature: 
            0) Stateless (GPU memory are free'ed each microbatch)
            1) Everything in the "Main" thread 
            2) Use a new group/communicator/cudaStream
            3) Blocking
        
        Assumption:
            0) distributed environment has already been initialized with gloo
    """
    def __init__(self, rank, world_size, verbose=False):
        assert dist.get_backend() == "gloo"
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        assert self.rank == torch.cuda.current_device() # torch.cuda.set_device(rank)
        # build a world group for NCCL collectives
        tensor = torch.tensor(1.0, dtype=torch.float32, device="cuda:%d"%(self.rank))
        self.world_group = dist.new_group(ranks=list(range(self.world_size)), backend='nccl')
        dist.all_reduce(tensor, group=self.world_group, op=dist.ReduceOp.SUM)
        # print("[P2PM] rank={} init'ed world communicator of NCCL and its cudaStream".format(self.rank))
        del tensor; gc.collect(); torch.cuda.empty_cache()
        if self.verbose: self.allreduce_byte_cnt = 0

    @torch.no_grad()
    def average_gradients(self, model):
        """ GPU Gradient averaging. """
        size = float(self.world_size)
        for param in model.parameters():
            dist.all_reduce(param.grad.data, group=self.world_group, op=dist.ReduceOp.SUM)
            param.grad.data /= size
            if self.verbose: self.allreduce_byte_cnt += param.grad.data.nelement()*param.grad.data.element_size()
    
    @torch.no_grad()
    def average_buffers(self, model):
        """ Buffer averaging. """
        size = float(self.world_size)
        for buf in model.buffers():
            if isinstance(buf.data, torch.Tensor) and buf.data.dtype in [torch.float16, torch.float32, torch.float64]: 
                if buf.is_cuda:
                    dist.all_reduce(buf.data, group=self.world_group, op=dist.ReduceOp.SUM) # NCCL
                else: 
                    dist.all_reduce(buf.data, op=dist.ReduceOp.SUM) # Gloo
                buf.data /= size
                if self.verbose: self.allreduce_byte_cnt += buf.data.nelement()*buf.data.element_size()
