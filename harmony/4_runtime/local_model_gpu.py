# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
from math import ceil
from collections import OrderedDict as ODict

import torch
from torch.nn import Parameter

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from task_data_struct import Medium, vTask
import threadsafe_data_struct

if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True

# 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
def delete_param_grad_buf(top_module, manual_gc=False):
    ''' Recursively delete all params, grads, and buffers from this module on either CPU or GPU.

        Note: nn.Parameters (namely, Variable) is wrapper of torch.Tenosr. The core tensor can be accessed by param.data, but not recommended (access .data is source of all evils ref: https://discuss.pytorch.org/t/how-to-delete-every-grad-after-training/63644) 
        Note: Delete param? 
            -. param.data/param.data.storage() can not be del'ed
            -. for param in self.local_model_gpu.parameters(): del param # doesn't affect content
            -. param.data = None # TypeError: Variable data has to be a tensor, but got NoneType
            -. del moduel._parameters[key] will leave moduel._parameters[key]=None. Then have to new Parameter(). Then del new Parameter can cause uncollectable alloc on GPU.
            +. param.data = torch.empty(0, device="cpu") # use pytorch's current behavior -- in-place update and let python do the gc, working for both GPU and CPU (equal to del tensor)
        Note: Assign grad?
            -. param.data.grad can not be assigned 
            -. param.grad.data = only tensor, not None
            +. param.grad = * instead, 
            +. param.grad = None works for both GPU and CPU (equal to del tensor)
        Note: Delete buffer?
            +. del _buffer[key] # works
            +. _buffers[key] = fn(buf) # works
            +. buffer has no grad
    '''   
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
                param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                assert not param.requires_grad
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

class LocalModelGPU(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    # 1.保险操作：确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
    # 2.将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
    #   即现在empty_model的该层layer存在于GPU上
    # 3.删除GPU上当前layer的所有参数、梯度和缓冲区
    def __init__(self, pinned_model, shared_model, empty_model, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.pinned_model = pinned_model
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 该参数默认为false，这里执行：
        # 确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
        if not self.no_pin_model:
            # confirm pinned model is pinned, local, CPU, and no grad
            for param in self.pinned_model.parameters():
                assert param.data.is_pinned() and (not param.data.is_shared()) and (not param.data.is_cuda)
                assert (not param.requires_grad) and (param.grad is None)
            for buf in self.pinned_model.buffers():
                assert buf.data.is_pinned() and (not buf.data.is_shared()) and (not buf.data.is_cuda)
                assert (not buf.requires_grad) and (buf.grad is None)
        
        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        # 代表GPU上的模型
        self.model = empty_model
        
        # self.model.cuda() # Moves all model parameters and buffers to the GPU. (replace CPU params and buffers with newly alloacated GPU param and buffers) Return self module.
        # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
        # 即现在empty_model的该层layer存在于GPU上
        self.swapin_param_buf(True)
        # initialize empty shell on GPU
        # 删除empty_model当前layer上的所有参数、梯度和缓冲区
        self.del_param_grad_buf(manual_gc=True)
        # print("[LocalModelGPU][id%d] rank%d: initialized local model on GPU (empty shell)"%(self.id, self.rank))
    
    def del_param_grad_buf(self, manual_gc=False):
        # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
        delete_param_grad_buf(self.model, manual_gc=manual_gc)
   
    # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model
    # 即现在empty_model存在于GPU上
    @torch.no_grad()
    def swapin_param_buf(self, forward_only=True): 
        ''' Recursively allocate and copy-in all params and buffers from cpu module to self.local_model_gpu
            
            Note: if gpu_m._parameters[key] is previously del'ed to None, then swapin needs to create a new Parameter. Then it may leave uncollectable allocation on GPU after del this new'ed Parameter.
        '''
        # 由于no_pin_model参数默认为false，这里的model指代存放在固定内存中的model
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # 将 CPU 模型中的参数数据复制到 GPU 模型中，并在 GPU 上分配内存
                    gpu_m._parameters[key].data = param.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert gpu_m._parameters[key].grad is None and (not gpu_m._parameters[key].requires_grad), "swapin requires no grad for both FP and BP"
                    # if not forward_only: 
                    #     gpu_m._parameters[key].requires_grad_(True)
                    # assert not param.is_cuda
                    # print("\t _parameter[{}]".format(key))
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                    gpu_m._buffers[key] = buf.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert not buf.is_cuda and (not gpu_m._buffers[key].requires_grad)
                    # print("\t _buffers[{}]".format(key))
        # print("[LocalModelGPU] rank{} swapin'ed params and bufs".format(self.rank))
        
        if not forward_only: # backward # move to here for 1) batching swap on GPU, 2) GPU CPU parallelism
            self.set_param_requiresgrad()
    
    # 控制模型的参数是否需要梯度
    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的tensor（分配内存不初始化值）
    @torch.no_grad()
    def alloc_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = torch.empty(param.shape,  dtype=param.dtype, device=self.rank, requires_grad=False)
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key] = torch.empty(buf.shape,  dtype=buf.dtype, device=self.rank, requires_grad=False)
    
    # 将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
    @torch.no_grad()
    def copyin_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy

    @torch.no_grad()
    def copyin_param_buf_blocking(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                   gpu_m._parameters[key].data.copy_(param.data, non_blocking=False) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=False) # inplace copy

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    # 1.在cpu上按照param的shape和类型分配一个零值tensor
    # 2.将tensor拷贝到pinned memory中
    # 3.将刚刚分配的tensor赋给 cpu_param.grad
    # 4.以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上，即拷贝到shared_model上
    # shared_model的param.grad实际上在pinned memory上
    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                # 若shared memory版本模型（每一层都是一个模型）的grad属性为空，为其分配一个pinned 
                # memory上的tensor作为值
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    # 在cpu上按照param的shape和类型分配一个零值tensor
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    # 将tensor拷贝到pinned memory中
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    # 将刚刚分配的tensor赋给 cpu_param.grad
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                # 以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    # shared_model的param.grad实际上在pinned memory上
    @torch.no_grad()
    def swapout_grad_blocking(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                # 若shared memory版本模型（每一层都是一个模型）的grad属性为空，为其分配一个pinned 
                # memory上的tensor作为值
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    # 在cpu上按照param的shape和类型分配一个零值tensor
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    # 将tensor拷贝到pinned memory中
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    # 将刚刚分配的tensor赋给 cpu_param.grad
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                # 以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=False) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    # 1.若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
    # 2.将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
    # 最终这个pinned_buf会以成员变量的形式挂在shared_model上
    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                # 若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            # 将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"
                # 就是一个下三角全为1的矩阵，根本没变，可见不是可训练的参数
                # print(f"rank:{self.rank}, 刚刚卸载的buffer为:{named_pin_buf[name]}")

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    @torch.no_grad()
    def swapout_buf_blocking(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                # 若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            # 将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


class LocalModelGPU_2(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    # 1.保险操作：确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
    # 2.将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
    #   即现在empty_model的该层layer存在于GPU上
    # 3.删除GPU上当前layer的所有参数、梯度和缓冲区
    def __init__(self, pinned_model, shared_model, empty_model, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.pinned_model = pinned_model
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 该参数默认为false，这里执行：
        # 确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
        if not self.no_pin_model:
            # confirm pinned model is pinned, local, CPU, and no grad
            for param in self.pinned_model.parameters():
                assert param.data.is_pinned() and (not param.data.is_shared()) and (not param.data.is_cuda)
                assert (not param.requires_grad) and (param.grad is None)
            for buf in self.pinned_model.buffers():
                assert buf.data.is_pinned() and (not buf.data.is_shared()) and (not buf.data.is_cuda)
                assert (not buf.requires_grad) and (buf.grad is None)
        
        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        # 代表GPU上的模型
        self.model = empty_model
        
        # self.model.cuda() # Moves all model parameters and buffers to the GPU. (replace CPU params and buffers with newly alloacated GPU param and buffers) Return self module.
        # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
        # 即现在empty_model的该层layer存在于GPU上
        self.swapin_param_buf(True)
        # initialize empty shell on GPU
        # 删除empty_model当前layer上的所有参数、梯度和缓冲区
        self.del_param_grad_buf(manual_gc=True)
        # print("[LocalModelGPU][id%d] rank%d: initialized local model on GPU (empty shell)"%(self.id, self.rank))
    
    def del_param_grad_buf(self, manual_gc=False):
        # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
        delete_param_grad_buf(self.model, manual_gc=manual_gc)
   
    # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model
    # 即现在empty_model存在于GPU上
    @torch.no_grad()
    def swapin_param_buf(self, forward_only=True): 
        ''' Recursively allocate and copy-in all params and buffers from cpu module to self.local_model_gpu
            
            Note: if gpu_m._parameters[key] is previously del'ed to None, then swapin needs to create a new Parameter. Then it may leave uncollectable allocation on GPU after del this new'ed Parameter.
        '''
        # 由于no_pin_model参数默认为false，这里的model指代存放在固定内存中的model
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # 将 CPU 模型中的参数数据复制到 GPU 模型中，并在 GPU 上分配内存
                    gpu_m._parameters[key].data = param.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert gpu_m._parameters[key].grad is None and (not gpu_m._parameters[key].requires_grad), "swapin requires no grad for both FP and BP"
                    # if not forward_only: 
                    #     gpu_m._parameters[key].requires_grad_(True)
                    # assert not param.is_cuda
                    # print("\t _parameter[{}]".format(key))
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                        gpu_m._buffers[key] = buf.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert not buf.is_cuda and (not gpu_m._buffers[key].requires_grad)
                    # print("\t _buffers[{}]".format(key))
        # print("[LocalModelGPU] rank{} swapin'ed params and bufs".format(self.rank))
        
        if not forward_only: # backward # move to here for 1) batching swap on GPU, 2) GPU CPU parallelism
            self.set_param_requiresgrad()
    
    # 控制模型的参数是否需要梯度
    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的tensor（分配内存不初始化值）
    @torch.no_grad()
    def alloc_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = torch.empty(param.shape,  dtype=param.dtype, device=self.rank, requires_grad=False)
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key] = torch.empty(buf.shape,  dtype=buf.dtype, device=self.rank, requires_grad=False)
    
    # 将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
    @torch.no_grad()
    def copyin_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy

    @torch.no_grad()
    def copyin_param_buf_blocking(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                   gpu_m._parameters[key].data.copy_(param.data, non_blocking=False) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=False) # inplace copy

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    # 1.在cpu上按照param的shape和类型分配一个零值tensor
    # 2.将tensor拷贝到pinned memory中
    # 3.将刚刚分配的tensor赋给 cpu_param.grad
    # 4.以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上，即拷贝到shared_model上
    # shared_model的param.grad实际上在pinned memory上
    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                # 若shared memory版本模型（每一层都是一个模型）的grad属性为空，为其分配一个pinned 
                # memory上的tensor作为值
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    # 在cpu上按照param的shape和类型分配一个零值tensor
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    # 将tensor拷贝到pinned memory中
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    # 将刚刚分配的tensor赋给 cpu_param.grad
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                # 以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    # shared_model的param.grad实际上在pinned memory上
    @torch.no_grad()
    def swapout_grad_blocking(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                # 若shared memory版本模型（每一层都是一个模型）的grad属性为空，为其分配一个pinned 
                # memory上的tensor作为值
                if cpu_param.grad is None:
                    assert cpu_param.requires_grad
                    # 在cpu上按照param的shape和类型分配一个零值tensor
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    # 将tensor拷贝到pinned memory中
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    # 将刚刚分配的tensor赋给 cpu_param.grad
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                # 以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=False) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    # 1.若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
    # 2.将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
    # 最终这个pinned_buf会以成员变量的形式挂在shared_model上
    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                # 若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            # 将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    @torch.no_grad()
    def swapout_buf_blocking(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                # 若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            # 将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=False)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

# 📍
# 1.专为worker5提供的实现，对当前rank不负责的layer，也没有初始化的必要了，初始化仅仅是把self.model指向一个空的model(层)
# 2.添加copyin_param_buf_from_pinned_buffer方法：进行从共用的pinned buffer到gpu的复制，而不是原来的layer的pinned版本到gpu的复制
class LocalModelGPU_for_worker5(object):
    """ A wrapper class of process-local vlayer on GPU.
    """
    # 1.保险操作：确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
    # 2.将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
    #   即现在empty_model的该层layer存在于GPU上
    # 3.删除GPU上当前layer的所有参数、梯度和缓冲区
    def __init__(self, cpu_layers, pinned_model, shared_model, empty_model, shared_model_nvme, layer_id_to_layer_idx, transformer_layer_idx_to_shape, id, X_names, Y_names, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this in subprocess """ 
        self.pinned_model = pinned_model
        self.shared_model = shared_model
        self.id = id # vlayer_id
        self.X_names = X_names
        self.Y_names = Y_names
        self.rank, self.world_size = rank, world_size
        assert self.rank == torch.cuda.current_device()
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf

        # 📍需要该类的_get_pinned_model方法得到对应的pinned model
        self.shared_model_nvme_handler = shared_model_nvme
        # 📍
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        # 📍
        self.cpu_layers = cpu_layers
        self.transformer_layer_idx_to_shape = transformer_layer_idx_to_shape
        

        # 该参数默认为false，这里执行：
        # 确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
        if not self.no_pin_model and self.id in self.cpu_layers:
            # confirm pinned model is pinned, local, CPU, and no grad
            for param in self.pinned_model.parameters():
                assert param.data.is_pinned() and (not param.data.is_shared()) and (not param.data.is_cuda)
                assert (not param.requires_grad) and (param.grad is None)
            for buf in self.pinned_model.buffers():
                assert buf.data.is_pinned() and (not buf.data.is_shared()) and (not buf.data.is_cuda)
                assert (not buf.requires_grad) and (buf.grad is None)
        
        # use existing empty model one
        assert isinstance(empty_model, torch.nn.Module)
        # 代表GPU上的模型
        self.model = empty_model
        
        # 📍
        # 若当前rank不负责该layer，以下逻辑也没有存在的必要
        if self.id in self.cpu_layers:
            # self.model.cuda() # Moves all model parameters and buffers to the GPU. (replace CPU params and buffers with newly alloacated GPU param and buffers) Return self module.
            # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model 对应的layer的data上
            # 即现在empty_model的该层layer存在于GPU上
            self.swapin_param_buf(True)
            # initialize empty shell on GPU
            # 删除empty_model当前layer上的所有参数、梯度和缓冲区
            self.del_param_grad_buf(manual_gc=True)
            # print("[LocalModelGPU][id%d] rank%d: initialized local model on GPU (empty shell)"%(self.id, self.rank))
        
    # 感觉local model的初始化中的这两行没啥用啊，暂时不进行re_init
    def re_init(self):
        self.swapin_param_buf(True)
        self.del_param_grad_buf(manual_gc=True)

    def del_param_grad_buf(self, manual_gc=False):
        # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
        delete_param_grad_buf(self.model, manual_gc=manual_gc)
   
    # 将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给 empty_model
    # 即现在empty_model存在于GPU上
    @torch.no_grad()
    def swapin_param_buf(self, forward_only=True): 
        ''' Recursively allocate and copy-in all params and buffers from cpu module to self.local_model_gpu
            
            Note: if gpu_m._parameters[key] is previously del'ed to None, then swapin needs to create a new Parameter. Then it may leave uncollectable allocation on GPU after del this new'ed Parameter.
        '''
        # 由于no_pin_model参数默认为false，这里的model指代存放在固定内存中的model
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # 将 CPU 模型中的参数数据复制到 GPU 模型中，并在 GPU 上分配内存
                    gpu_m._parameters[key].data = param.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert gpu_m._parameters[key].grad is None and (not gpu_m._parameters[key].requires_grad), "swapin requires no grad for both FP and BP"
                    # if not forward_only: 
                    #     gpu_m._parameters[key].requires_grad_(True)
                    # assert not param.is_cuda
                    # print("\t _parameter[{}]".format(key))
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    # if len(buf.shape) == 0: # Scalar buffer stays on CPU
                    #     gpu_m._buffers[key] = buf.clone().detach().requires_grad_(False)
                    # else:
                    gpu_m._buffers[key] = buf.cuda(non_blocking=MEMCPY_NONBLK) # Returns a copy of this tensor object in CUDA memory.
                    # assert not buf.is_cuda and (not gpu_m._buffers[key].requires_grad)
                    # print("\t _buffers[{}]".format(key))
        # print("[LocalModelGPU] rank{} swapin'ed params and bufs".format(self.rank))
        
        if not forward_only: # backward # move to here for 1) batching swap on GPU, 2) GPU CPU parallelism
            self.set_param_requiresgrad()
    
    # 控制模型的参数是否需要梯度
    @torch.no_grad()
    def set_param_requiresgrad(self, requires_grad=True): 
        ''' From def swapin_param_buf() '''
        for gpu_m in self.model.modules(): # iterator over all modules in the network
            for key, param in gpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    param.requires_grad_(requires_grad)

    # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的tensor（分配内存不初始化值）
    @torch.no_grad()
    def alloc_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        # print(f"rank:{self.rank}, self.model:{self.model}, cpu_model:{cpu_model}")
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = torch.empty(param.shape,  dtype=param.dtype, device=self.rank, requires_grad=False)
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key] = torch.empty(buf.shape,  dtype=buf.dtype, device=self.rank, requires_grad=False)
    
    # 📍
    @torch.no_grad()
    def alloc_param_buf_2(self, vt_idx, layer_id): 
        # print(f"rank:{self.rank}, vt.idx：{vt_idx}, layer_id:{layer_id}")
        # pinned_model = None
        # if layer_id in self.cpu_layers:
        #     pinned_model = self.pinned_model
        # else:
        #     layer_idx = self.layer_id_to_layer_idx[vt_idx][layer_id]
        #     pinned_model = self.shared_model_nvme_handler.get_pinned_buffer(layer_idx)

        cpu_model = self.shared_model

        global_param_idx = 0
        # print(f"rank:{self.rank}, self.model：{self.model}, pinned model:{pinned_model}")
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    gpu_m._parameters[key].data = torch.empty(self.transformer_layer_idx_to_shape[global_param_idx], dtype=param.dtype, device=self.rank, requires_grad=False)
                    # print(f"rank:{self.rank}, layer{layer_id}，查看GPU上刚刚分配的tensor形状:{gpu_m._parameters[key].shape}")
                    global_param_idx+=1
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key] = torch.empty(self.transformer_layer_idx_to_shape[global_param_idx], dtype=buf.dtype, device=self.rank, requires_grad=False)
                    # print(f"rank:{self.rank}, layer{layer_id}，查看GPU上刚刚分配的buffer形状:{gpu_m._buffers[key].shape}")
                    global_param_idx+=1

    # 将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
    @torch.no_grad()
    def copyin_param_buf(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    print(f"rank:{self.rank}, copy in cpu层, {gpu_m._parameters[key].data.shape}, {param.data.shape}", flush=True)
                    gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy

    # 📍并非从nvme向pinned buffer复制,而是pinned buffer到gpu
    # 📍数据已经在pinned buffer中了，直接copy
    @torch.no_grad()
    def copyin_param_buf_from_pinned_buffer(self, vt_idx, layer_id): 
        layer_idx = self.layer_id_to_layer_idx[vt_idx][layer_id]
        # 得到当前层使用的pinned model(pinned buffer)
        pinned_model = self.shared_model_nvme_handler.get_pinned_buffer(layer_idx)

        ''' From def swapin_param_buf() '''
        # cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), pinned_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    print(f"rank:{self.rank}, copy in 0 or last, {gpu_m._parameters[key].data.shape}, {param.data.shape}")
                    gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    print(f"rank:{self.rank}, layer{layer_id} to gpu, {gpu_m._parameters[key].shape}, {param.shape}")
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy

    # 📍上个方法的完善版本，我们需要确定是从Pinned memory还是pinned buffer向gpu复制
    @torch.no_grad()
    def copyin_param_buf_2(self, vt_idx, layer_id): 
        # print(f"rank:{self.rank}, 准备开始transformer layer{layer_id} 的 cpu->gpu 复制", flush=True)
        # 得到当前层使用的pinned model(pinned buffer)
        # pinned_model = self.pinned_model if self.cpu_layers else self.shared_model_nvme_handler.get_pinned_buffer(layer_idx)
        pinned_model = None
        if layer_id in self.cpu_layers:
            pinned_model = self.pinned_model
            for gpu_m, cpu_m in zip(self.model.modules(), pinned_model.modules()):
                for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                    if param is not None:
                        gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                    if buf is not None:
                        gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy 
        else:
            # 📍先将pinned buffer的各个参数恢复为原来的形状，再复制到GPU
            layer_idx = self.layer_id_to_layer_idx[vt_idx][layer_id]
            pinned_model = self.shared_model_nvme_handler.get_pinned_buffer(layer_idx)
            
            global_param_idx = 0
            for gpu_m, cpu_m in zip(self.model.modules(), pinned_model.modules()):
                for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                    if param is not None:
                        # print(f"\trank:{self.rank}, 准备开始transformer layer{layer_id}-{global_param_idx} param 的 cpu->gpu 复制", flush=True)
                        gpu_m._parameters[key].data.copy_(param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK) # inplace copy
                        global_param_idx+=1
                for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                    if buf is not None:
                        # print(f"\trank:{self.rank}, 准备开始transformer layer{layer_id}-{global_param_idx} buf 的 cpu->gpu 复制", flush=True)
                        gpu_m._buffers[key].data.copy_(buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK) # inplace copy 
                        global_param_idx+=1

    # 📍双buffer版本，与上一个方法的区别：需根据buffer_id来获取pinned model
    def copyin_param_buf_for_double_buffer(self, buffer_id, vt_idx, layer_id):
        if layer_id in self.cpu_layers:
            pinned_model = self.pinned_model
            for gpu_m, cpu_m in zip(self.model.modules(), pinned_model.modules()):
                for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                    if param is not None:
                        gpu_m._parameters[key].data.copy_(param.data, non_blocking=MEMCPY_NONBLK) # inplace copy
                for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                    if buf is not None:
                        gpu_m._buffers[key].data.copy_(buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy 

        else:
            layer_idx = self.layer_id_to_layer_idx[vt_idx][layer_id]
            pinned_model = self.shared_model_nvme_handler.get_pinned_buffer(buffer_id, layer_idx)

            global_param_idx = 0
            for gpu_m, cpu_m in zip(self.model.modules(), pinned_model.modules()):
                for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                    if param is not None:
                        # print(f"\trank:{self.rank}, 准备开始transformer layer{layer_id}-{global_param_idx} param 的 cpu->gpu 复制", flush=True)
                        gpu_m._parameters[key].data.copy_(param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK) # inplace copy
                        global_param_idx+=1
                for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                    if buf is not None:
                        # print(f"\trank:{self.rank}, 准备开始transformer layer{layer_id}-{global_param_idx} buf 的 cpu->gpu 复制", flush=True)
                        gpu_m._buffers[key].data.copy_(buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK) # inplace copy 
                        global_param_idx+=1

    @torch.no_grad()
    def copyin_param_buf_blocking(self): 
        ''' From def swapin_param_buf() '''
        cpu_model = self.shared_model if self.no_pin_model else self.pinned_model
        # model.modules()：返回一个迭代器，用于递归地遍历模型 model 中的所有模块，包括模型本身以及它的子模块
        # 📌无需担心模型本身，因为模型本身的_parameters的长度为0，for循环不会执行
        for gpu_m, cpu_m in zip(self.model.modules(), cpu_model.modules()):
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                   gpu_m._parameters[key].data.copy_(param.data, non_blocking=False) # inplace copy
            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    gpu_m._buffers[key].data.copy_(buf.data, non_blocking=False) # inplace copy

    def __call__(self, *input, **kwargs):
        return self.model(*input, **kwargs)
        # ref: https://github.com/pytorch/pytorch/blob/472be69a736c0b2aece4883be9f8b18e2f3dfbbd/torch/nn/modules/module.py#L487
    
    def average_grad(self, p2p_handler):
        p2p_handler.average_gradients(self.model)

    def average_buf(self, p2p_handler):
        p2p_handler.average_buffers(self.model)

    # 1.在cpu上按照param的shape和类型分配一个零值tensor
    # 2.将tensor拷贝到pinned memory中
    # 3.将刚刚分配的tensor赋给 cpu_param.grad
    # 4.以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上，即拷贝到shared_model上
    # shared_model的param.grad实际上在pinned memory上
    @torch.no_grad()
    def swapout_grad(self):
        ''' in-place copy gradient from local model gpu to local .grad on cpu 
            (lazily allocate local .grad if not exists)
        '''
        for gpu_param, cpu_param in zip(self.model.parameters(), self.shared_model.parameters()):
            if gpu_param.grad is not None:
                # 若shared memory版本模型（每一层都是一个模型）的grad属性为空，为其分配一个pinned 
                # memory上的tensor作为值
                if cpu_param.grad is None:
                    # print(f"rank:{self.rank}, +++++++++++++++++++++++++++++重新创建梯度")
                    assert cpu_param.requires_grad
                    # 在cpu上按照param的shape和类型分配一个零值tensor
                    g = torch.zeros(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False)
                    # 将tensor拷贝到pinned memory中
                    if not self.no_pin_grad_buf:
                        g = g.pin_memory()
                    # 将刚刚分配的tensor赋给 cpu_param.grad
                    cpu_param.grad = g
                    # cpu_param.grad = torch.empty(cpu_param.shape, dtype=cpu_param.dtype, device="cpu", requires_grad=False, pin_memory=not self.no_pin_grad_buf) # NOTE: dont' use empty(pin_memory) with shared memory 
                    assert not cpu_param.grad.is_shared() # and cpu_param.grad.is_pinned()
                # 以非阻塞的方式将gpu参数的grad.data 拷贝到刚刚分配的tensor上
                cpu_param.grad.data.copy_(gpu_param.grad.data, non_blocking=MEMCPY_NONBLK) # in-place copies the elements from src tensor into self tensor (and returns self)
            else:
                print(ValueError("model has grad = None to swap"))
                cpu_param.grad = None
        # print("[LocalModelGPU] rank{} swapout'ed grad".format(self.rank))

    # 1.若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
    # 2.将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
    # 最终这个pinned_buf会以成员变量的形式挂在shared_model上
    @torch.no_grad()
    def swapout_buf(self): 
        ''' in-place copy buffers from local model gpu to local pinned buf or shared buf on cpu 
            (lazily allocate local pinned buf if not exists)
        '''
        if self.no_pin_grad_buf:
            for gpu_buf, cpu_buf in zip(self.model.buffers(), self.shared_model.buffers()):
                if gpu_buf is not None:
                    cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert cpu_buf is None, "local_model_gpu has unexpected None buffer to swap"
        else:
            if not hasattr(self.shared_model, "pinned_buf"):
                self.shared_model.pinned_buf = ODict()
                # 若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                for name, buf in self.shared_model.named_buffers():
                    if buf is not None:
                        pin_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu", requires_grad=False).pin_memory()
                        # NOTE: dont' use torch.empty(pin_memory) with shared memory 
                        assert not pin_buf.is_shared() and pin_buf.is_pinned()
                    else:
                        pin_buf = None
                    self.shared_model.pinned_buf[name] = pin_buf
                for (gpu_name, gpu_buf), (pin_name, pin_buf) in zip(self.model.named_buffers(), self.shared_model.pinned_buf.items()):
                    assert gpu_name == pin_name
                    assert (gpu_buf is not None and pin_buf is not None) or \
                           (gpu_buf is None and pin_buf is None) 
            named_pin_buf = self.shared_model.pinned_buf
            # 将gpu上的buffer tensor以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
            for name, gpu_buf in self.model.named_buffers():
                if gpu_buf is not None:
                    named_pin_buf[name].data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local_model_gpu has unexpected None buffer to swap"

        # for gpu_buf, cpu_buf in zip(self.model.buffers(), self.pinned_model.buffers()):
        #     cpu_buf.data.copy_(gpu_buf.data, non_blocking=MEMCPY_NONBLK) # Copies the elements from src tensor into self tensor (and returns self); in-place copy (no new tensor created)
        # print("[LocalModelGPU] rank{} swapout'ed buf".format(self.rank))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

# 分析：put_queue 相当于一个启动条件，即来活了，其中的元素是其他线程或自己添加进去的，这个里面得有东西，自身这个线程才能取出东西，并执行后面的步骤
#      不然会一直阻塞，直到里面有东西能够取出来。取出来就意味着前提条件已经执行完了（可以简单这么理解，在这个类中取出来后还需要显式的等待执行完）
#      get_queue 中装着(vt.idx,ev_swapin)，表示这个ev_swapin事件正在执行或已经执行完了，
""" Prefetch LocalModelGPU  """
class PrefetchLocalModelGPU(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        # 若没有给定一个stream，则在当前rank上创建一个新的CUDA stream
        # 目前来看，swapin_stream就是一个新的流
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        # 该参数是给定了的，其实和不给定执行else一样，都是cuda上的默认流
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    # input就是向put_queue中加元素，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    #
    # 为向GPU复制W、B做准备工作：
    # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
    # 2.在默认流上（ev_compute）记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
    # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
    # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    def iput(self, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if vt is None:
            return
        # 断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
        assert not self.is_running, "the prefetch is still running"
        # 将 self.is_running 标志设置为 True，表示当前有 iput 操作正在执行。
        self.is_running = True
        assert isinstance(vt, vTask)
        # record previous compute event for swapin stream to wait
        # 在 compute_stream （默认流）上记录事件 ev_compute，若参数为空分配一个新的event，用于后续在 swapin 流上等待
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        # 若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
            # 若该层的W和B的媒介为SHM
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
                self.local_model[l].alloc_param_buf()
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
        # do the rest in background thread
        # 将(vt, ev_compute)添加到 put_queue 中
        self.put_queue.add((vt,ev_compute))
        if self.nvprof: nvtx_range_pop() 

    # 不断尝试从put_queue中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B）)，执行（若队列是空的会被阻塞）：
    # 1.从 put_queue 队列中弹出 (vt, ev_compute)，若队列没有元素会被阻塞在这
    # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    # 3.在 CUDA 流 self.swapin_stream 上等待事件 ev_compute 的完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
    #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
    # 4.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 5.在当前 CUDA 流上记录一个事件 ev_swapin
    # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 从 put_queue 队首弹出一个元素，若队列是空的，会被阻塞在这里
            vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            # 返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。
            # 即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
            # 📌根据syncpin_handler中的注释来看，从共享内存到固定内存的同步操作是阻塞的（尽管代码上直观来看是非阻塞的），所以这里一定执行完了
            #   从另一个角度看，这个get()本身就是在等待sync线程执行完一次同步操作
            syncpin_vt_idx = self.syncpin_handler.get()
            # 确保当前在GPU上初始化W和B的任务和共享层到pinned层复制任务的层，目标是同一个vt
            # 下面就是要把该任务对应的layer pack的W和B swap in到GPU上
            assert syncpin_vt_idx == vt.idx
            # let swapin stream waits for this compute event 
            # 等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
            # 即等待任务vt的所有层已经拿到GPU上
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            # 在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
            # 显示的设置参数 param 的 requires_grad 属性为 True
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                for l in vt.layers:
                    # 若W和B在的通讯媒介为共享内存，将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
                    if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                        self.local_model[l].copyin_param_buf()
                        # 若vt是BWD任务，还需将模型的参数设置为需要梯度
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    # 若W和B被pin在设备上，显然不用执行拷贝操作
                    elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    else: # P2P
                        raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            # 在当前 CUDA 流上记录一个事件
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            # 将任务的idx和 swap_in 事件以元组的形式加入到 get_queue 中
            self.get_queue.add( (vt.idx,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
    # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)
    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    # 该函数附带保险操作（实际上可能之前就没调用过input来为当前线程添加预取任务）：
    # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件。拿取只代表逻辑上执行完，
    # 实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx
    # 📌分析：input和get是一一对应的，input将is_running置为true，get(调用wait)将is_running置为false。不调用get，is_running就不可能为false
    #         若发现is_running不为true，就说明之前根本就没执行过layer的预取
    
    # 1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
    # 2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
    # 3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 4.调用_wait将is_running 置为false，返回get_queue 的首个 (vt.idx，ev_swapin)
    # 5.self.compute_stream.wait_event(ev_swapin)
    # 6.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1相同
    def get(self, vt, suc_vt=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        assert isinstance(vt, vTask)
        # wait current one (if no current one, get this one)
        # 若当前没有正在GPU上分配vt的所有层
        if not self.is_running:
            # self.put_queue.add(vt)
            # 这意味着 syncpin_handler 这个线程开始执行vt的模型的从共享内存到固定内存的复制
            self.syncpin_handler.iput(vt)
            # 为向GPU复制W、B做准备工作：
            # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
            # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
            # 
            # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
            # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin_stream 这个流上等待
            # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
            # 4.将 (vt, ev_compute) 添加到 put_queue 中
            self.iput(vt)
        # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
        # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)，即正在或已经执行玩的swap_in事件，(vt.idx,ev_swapin)
        #   若 _thread_func, 即 swap_in 没在swapin_stream上分配完，会阻塞在 remove() 上
        cur_vt_idx, ev_swapin = self._wait()
        # 等待该vt上所有的层在GPU上完成分配空tensor
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # syncpin next one if exsits
        # 若给定了后继任务，
        if suc_vt is not None:
            self.syncpin_handler.iput(suc_vt)
        # 返回 cur_vt_idx
        return cur_vt_idx


# ================ my version =========================
class PrefetchLocalModelGPU_2(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        # 若没有给定一个stream，则在当前rank上创建一个新的CUDA stream
        # 目前来看，swapin_stream就是一个新的流
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        # 该参数是给定了的，其实和不给定执行else一样，都是cuda上的默认流
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        # self.is_running = False
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    # input就是向put_queue中加元素，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    #
    # 为向GPU复制W、B做准备工作：
    # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
    # 2.在默认流上（ev_compute）记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
    # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
    # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    def iput(self, layer_id, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if layer_id is None:
            return
        # 将 self.is_running 标志设置为 True，表示当前有 iput 操作正在执行。
        # self.is_running = True
        # record previous compute event for swapin stream to wait
        # 在 compute_stream （默认流）上记录事件 ev_compute，若参数为空分配一个新的event，用于后续在 swapin 流上等待
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        # 若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        # for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
        #     # 若该层的W和B的媒介为SHM
        #     if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
        #         # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
        #         self.local_model[l].alloc_param_buf()
        #     elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
        #         pass
        #     else: # P2P
        #         raise ValueError("Underdevelopment")
        
        if vt.In['W'][layer_id].medium=='SHM' and vt.In['B'][layer_id].medium=='SHM':
            # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
            self.local_model[layer_id].alloc_param_buf()
        elif vt.In['W'][layer_id].medium=='PIN' and vt.In['B'][layer_id].medium=='PIN':
            pass
        else: # P2P
            raise ValueError("Underdevelopment")

        # do the rest in background thread
        # 将(vt, ev_compute)添加到 put_queue 中
        self.put_queue.add((layer_id, vt, ev_compute))
        if self.nvprof: nvtx_range_pop() 

    # 不断尝试从put_queue中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B）)，执行（若队列是空的会被阻塞）：
    # 1.从 put_queue 队列中弹出 (vt, ev_compute)，若队列没有元素会被阻塞在这
    # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    # 3.在 CUDA 流 self.swapin_stream 上等待事件 ev_compute 的完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
    #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
    # 4.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 5.在当前 CUDA 流上记录一个事件 ev_swapin
    # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 从 put_queue 队首弹出一个元素，若队列是空的，会被阻塞在这里
            layer_id, vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}(L{}) call CopyIn(W,B)".format(vt.idx, layer_id)) 
            # get sync'ed pinned model 
            # 返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。
            # 即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
            # 📌根据syncpin_handler中的注释来看，从共享内存到固定内存的同步操作是阻塞的（尽管代码上直观来看是非阻塞的），所以这里一定执行完了
            #   从另一个角度看，这个get()本身就是在等待sync线程执行完一次同步操作
            syncpin_layer_id = self.syncpin_handler.get()
            # 确保当前在GPU上初始化W和B的任务和共享层到pinned层复制任务的层，目标是同一个vt
            # 下面就是要把该任务对应的layer pack的W和B swap in到GPU上
            assert syncpin_layer_id == layer_id
            # let swapin stream waits for this compute event 
            # 等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
            # 即等待任务vt的所有层已经拿到GPU上
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            # 在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
            # 显示的设置参数 param 的 requires_grad 属性为 True
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                # 若W和B在的通讯媒介为共享内存，将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
                # print(f"rank:{self.rank}, 正在预取layer{layer_id}")
                if vt.In['W'][layer_id].medium=='SHM' and vt.In['B'][layer_id].medium=='SHM':
                    self.local_model[layer_id].copyin_param_buf()
                    # 若vt是BWD任务，还需将模型的参数设置为需要梯度
                    if vt.type == 'BWD':
                        self.local_model[layer_id].set_param_requiresgrad()
                # 若W和B被pin在设备上，显然不用执行拷贝操作
                elif vt.In['W'][layer_id].medium=='PIN' and vt.In['B'][layer_id].medium=='PIN':
                    if vt.type == 'BWD':
                        self.local_model[layer_id].set_param_requiresgrad()
                else: # P2P
                    raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            # 在当前 CUDA 流上记录一个事件
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            # 将任务的idx和 swap_in 事件以元组的形式加入到 get_queue 中
            self.get_queue.add( (layer_id,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
    # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)
    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    # 该函数附带保险操作（实际上可能之前就没调用过input来为当前线程添加预取任务）：
    # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件。拿取只代表逻辑上执行完，
    # 实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx
    # 📌分析：input和get是一一对应的，input将is_running置为true，get(调用wait)将is_running置为false。不调用get，is_running就不可能为false
    #         若发现is_running不为true，就说明之前根本就没执行过layer的预取
    
    # 1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
    # 2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
    # 3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 4.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1相同
    def get(self, layer_id, vt):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        # wait current one (if no current one, get this one)
        # 若当前没有正在GPU上分配vt的所有层

        # self.put_queue.add(vt)
        # 这意味着 syncpin_handler 这个线程开始执行vt的模型的从共享内存到固定内存的复制
        self.syncpin_handler.input_one_layer(layer_id, vt)
        # 为向GPU复制W、B做准备工作：
        # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
        # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
        # 
        # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
        # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin_stream 这个流上等待
        # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        # 4.将 (vt, ev_compute) 添加到 put_queue 中
        self.iput(layer_id, vt)
        # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
        # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)，即正在或已经执行玩的swap_in事件，(vt.idx,ev_swapin)
        #   若 _thread_func, 即 swap_in 没在swapin_stream上分配完，会阻塞在 remove() 上
        # layer_id, ev_swapin = self._wait()
        # 等待该vt上所有的层在GPU上完成分配空tensor
        # self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # 返回 cur_vt_idx

        layer_id, ev_swapin = self.get_queue.remove()
        return layer_id, ev_swapin, self.compute_stream
    

    def layer_waiting_futs(self):
        layer_id, ev_swapin = self.get_queue.remove()
        self.compute_stream.wait_event(ev_swapin)

# 现已使用非嵌套线程，因此该类暂时废弃，不添加根据self.cpu_layer_id判断双区域shared memory的逻辑
# 怎么嵌套了：使用的swap_in_cpu_handler中又嵌套了syncpin_handler
class PrefetchLocalModelGPU_for_worker5(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, swap_in_cpu_handler, local_model, layer_num, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        # 若没有给定一个stream，则在当前rank上创建一个新的CUDA stream
        # 目前来看，swapin_stream就是一个新的流
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        # 该参数是给定了的，其实和不给定执行else一样，都是cuda上的默认流
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        self.is_running = False

        # 📍
        self.swap_in_cpu_handler = swap_in_cpu_handler
        self.layer_num = layer_num
        
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    # input就是向put_queue中加元素，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    #
    # 为向GPU复制W、B做准备工作：
    # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
    # 2.在默认流上（ev_compute）记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
    # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
    # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    def iput(self, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if vt is None:
            return
        # 断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
        assert not self.is_running, "the prefetch is still running"
        # 将 self.is_running 标志设置为 True，表示当前有 iput 操作正在执行。
        self.is_running = True
        assert isinstance(vt, vTask)
        # record previous compute event for swapin stream to wait
        # 在 compute_stream （默认流）上记录事件 ev_compute，若参数为空分配一个新的event，用于后续在 swapin 流上等待
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        # 若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
            # 若该层的W和B的媒介为SHM
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
                # 📍
                if vt.has_data and l == 0:
                    self.local_model[l].alloc_param_buf()
                    continue
                elif l == self.layer_num-3:
                    self.local_model[l].alloc_param_buf()
                    continue
                elif l == self.layer_num-2:
                    self.local_model[l].alloc_param_buf()
                    continue
                elif l == self.layer_num-1:
                    self.local_model[l].alloc_param_buf()
                    continue
                self.local_model[l].alloc_param_buf_2(vt.idx, l)
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
        # do the rest in background thread
        # 将(vt, ev_compute)添加到 put_queue 中
        self.put_queue.add((vt,ev_compute))
        if self.nvprof: nvtx_range_pop() 

    # 不断尝试从put_queue中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B）)，执行（若队列是空的会被阻塞）：
    # 1.从 put_queue 队列中弹出 (vt, ev_compute)，若队列没有元素会被阻塞在这
    # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    # 3.在 CUDA 流 self.swapin_stream 上等待事件 ev_compute 的完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
    #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
    # 4.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 5.在当前 CUDA 流上记录一个事件 ev_swapin
    # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 从 put_queue 队首弹出一个元素，若队列是空的，会被阻塞在这里
            vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            # 返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。
            # 即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
            # 📌根据syncpin_handler中的注释来看，从共享内存到固定内存的同步操作是阻塞的（尽管代码上直观来看是非阻塞的），所以这里一定执行完了
            #   从另一个角度看，这个get()本身就是在等待sync线程执行完一次同步操作
            swapin_vt_idx = self.swap_in_cpu_handler.get()
            print(f"rank:{self.rank}, swap in线程已执行完vt nvme->cpu的拿取({vt.layers})", flush=True)
            # 确保当前在GPU上初始化W和B的任务和共享层到pinned层复制任务的层，目标是同一个vt
            # 下面就是要把该任务对应的layer pack的W和B swap in到GPU上
            assert swapin_vt_idx == vt.idx

            # if vt.has_data:
            #     syncpin_layer_id = self.syncpin_handler.get()
            #     assert syncpin_layer_id == vt.layers[0]
            # elif vt.has_criterion:
            #     syncpin_layer_id = self.syncpin_handler.get()
            #     assert syncpin_layer_id == vt.layers[-2]
            
            # let swapin stream waits for this compute event 
            # 等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
            # 即等待任务vt的所有层已经拿到GPU上
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            # 在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
            # 显示的设置参数 param 的 requires_grad 属性为 True
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                for l in vt.layers:
                    # 若W和B在的通讯媒介为共享内存，将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
                    if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                        
                        if vt.has_data and l == vt.layers[0]:
                            print(f"rank:{self.rank}, layer{l}, 等待syncpin_handler.get()")
                            syncpin_layer_id = self.syncpin_handler.get()
                            print(f"rank:{self.rank}, layer{l}, syncpin_handler完成")
                            assert syncpin_layer_id == l
                            self.local_model[l].copyin_param_buf()
                        elif l == self.layer_num-3:
                            syncpin_layer_id = self.syncpin_handler.get()
                            assert syncpin_layer_id == l
                            self.local_model[l].copyin_param_buf()
                            print(f"rank:{self.rank}, layer{l}, 已完成cpu->gpu的拷贝", flush=True)
                        elif l == self.layer_num-2:
                            syncpin_layer_id = self.syncpin_handler.get()
                            assert syncpin_layer_id == l
                            self.local_model[l].copyin_param_buf()
                            print(f"rank:{self.rank}, layer{l}, 已完成cpu->gpu的拷贝", flush=True)
                        elif l == self.layer_num-1:
                            syncpin_layer_id = self.syncpin_handler.get()
                            assert syncpin_layer_id == l
                            self.local_model[l].copyin_param_buf()
                            print(f"rank:{self.rank}, layer{l}, 已完成cpu->gpu的拷贝", flush=True)
                        else:
                            # 📍从pinned buffer拷贝到gpu上
                            print(f"rank:{self.rank}, layer{l}, 准备开始cpu->gpu的拷贝", flush=True)
                            self.local_model[l].copyin_param_buf_2(vt.idx, l)
                            print(f"rank:{self.rank}, layer{l}, 已完成cpu->gpu的拷贝", flush=True)

                        # 若vt是BWD任务，还需将模型的参数设置为需要梯度
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()

                    # 若W和B被pin在设备上，显然不用执行拷贝操作
                    elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    else: # P2P
                        raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            # 在当前 CUDA 流上记录一个事件
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            # 将任务的idx和 swap_in 事件以元组的形式加入到 get_queue 中
            self.get_queue.add( (vt.idx,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
    # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)
    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    # 该函数附带保险操作（实际上可能之前就没调用过input来为当前线程添加预取任务）：
    # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件。拿取只代表逻辑上执行完，
    # 实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx
    # 📌分析：input和get是一一对应的，input将is_running置为true，get(调用wait)将is_running置为false。不调用get，is_running就不可能为false
    #         若发现is_running不为true，就说明之前根本就没执行过layer的预取
    
    # 1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
    # 2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
    # 3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 4.调用_wait将is_running 置为false，返回get_queue 的首个 (vt.idx，ev_swapin)
    # 5.self.compute_stream.wait_event(ev_swapin)
    # 6.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1相同
    def get(self, vt, suc_vt=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        assert isinstance(vt, vTask)
        # wait current one (if no current one, get this one)
        # 若当前没有正在GPU上分配vt的所有层
        print(f"rank:{self.rank}, 准备执行get的vt为{vt.layers}")
        if not self.is_running:

            # 若vt包含首层，该层因为不会卸载到nvme，需要从shared memory复制到Pinned memory
            # if vt.has_data:
            #     self.syncpin_handler.input_one_layer(vt.layers[0], vt)
            # if vt.has_criterion:
            #     self.syncpin_handler.input_one_layer(vt.layers[-2], vt)
            # if 

            # 📍现在无需shared model->pinned model的复制，直接从NVMe拿
            self.swap_in_cpu_handler.iput(vt)


            # 为向GPU复制W、B做准备工作：
            # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
            # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
            # 
            # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
            # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin_stream 这个流上等待
            # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
            # 4.将 (vt, ev_compute) 添加到 put_queue 中
            self.iput(vt)
        # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
        # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)，即正在或已经执行玩的swap_in事件，(vt.idx,ev_swapin)
        #   若 _thread_func, 即 swap_in 没在swapin_stream上分配完，会阻塞在 remove() 上
        cur_vt_idx, ev_swapin = self._wait()
        # 等待该vt上所有的层在GPU上完成分配空tensor
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # syncpin next one if exsits
        # 若给定了后继任务，
        if suc_vt is not None:
            # 📍
            self.swap_in_cpu_handler.iput(suc_vt)
        # 返回 cur_vt_idx
        return cur_vt_idx


# 上一个预取类使用了嵌套的线程拿取模型，即在自己写的swap_in_cpu类的线程中嵌套syncpinmodelinbkgd
# 该类只使用syncpinmodelinbkgd，其中直接使用swap_in_cpu线程的功能
class PrefetchLocalModelGPU_for_worker5_2(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, local_model, layer_num, cpu_layer_id, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.local_model = local_model # list
        self.rank = rank
        # 若没有给定一个stream，则在当前rank上创建一个新的CUDA stream
        # 目前来看，swapin_stream就是一个新的流
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        # 该参数是给定了的，其实和不给定执行else一样，都是cuda上的默认流
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    # input就是向put_queue中加元素，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    #
    # 为向GPU复制W、B做准备工作：
    # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
    # 2.在默认流上（ev_compute）记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
    # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
    # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    def iput(self, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if vt is None:
            return
        # 断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
        assert not self.is_running, "the prefetch is still running"
        # 将 self.is_running 标志设置为 True，表示当前有 iput 操作正在执行。
        self.is_running = True
        assert isinstance(vt, vTask)
        # record previous compute event for swapin stream to wait
        # 在 compute_stream （默认流）上记录事件 ev_compute，若参数为空分配一个新的event，用于后续在 swapin 流上等待
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        # 若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
            # 若该层的W和B的媒介为SHM
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
                # 📍
                if l in self.cpu_layer_id: #l == 0 or l == self.layer_num-3 or l == self.layer_num-2 or l == self.layer_num-1:
                    self.local_model[l].alloc_param_buf()
                    continue
                else:
                    self.local_model[l].alloc_param_buf_2(vt.idx, l)
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
        # do the rest in background thread
        # 将(vt, ev_compute)添加到 put_queue 中
        self.put_queue.add((vt,ev_compute))
        if self.nvprof: nvtx_range_pop() 

    # 不断尝试从put_queue中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B）)，执行（若队列是空的会被阻塞）：
    # 1.从 put_queue 队列中弹出 (vt, ev_compute)，若队列没有元素会被阻塞在这
    # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    # 3.在 CUDA 流 self.swapin_stream 上等待事件 ev_compute 的完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
    #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
    # 4.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 5.在当前 CUDA 流上记录一个事件 ev_swapin
    # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 从 put_queue 队首弹出一个元素，若队列是空的，会被阻塞在这里
            vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            # 返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。
            # 即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
            # 📌根据syncpin_handler中的注释来看，从共享内存到固定内存的同步操作是阻塞的（尽管代码上直观来看是非阻塞的），所以这里一定执行完了
            #   从另一个角度看，这个get()本身就是在等待sync线程执行完一次同步操作
            syncpin_vt_idx = self.syncpin_handler.get()
            # 确保当前在GPU上初始化W和B的任务和共享层到pinned层复制任务的层，目标是同一个vt
            # 下面就是要把该任务对应的layer pack的W和B swap in到GPU上
            assert syncpin_vt_idx == vt.idx
            # let swapin stream waits for this compute event 
            # 等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
            # 即等待任务vt的所有层已经拿到GPU上
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            # 在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
            # 显示的设置参数 param 的 requires_grad 属性为 True
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                for l in vt.layers:
                    # 若W和B在的通讯媒介为共享内存，将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
                    if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                        # 📌
                        self.local_model[l].copyin_param_buf_2(vt.idx, l)
                        # 若vt是BWD任务，还需将模型的参数设置为需要梯度
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    # 若W和B被pin在设备上，显然不用执行拷贝操作
                    elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    else: # P2P
                        raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            # 在当前 CUDA 流上记录一个事件
            ev_swapin = self.swapin_stream.record_event() 
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            # 将任务的idx和 swap_in 事件以元组的形式加入到 get_queue 中
            self.get_queue.add( (vt.idx,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
    # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)
    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    # 该函数附带保险操作（实际上可能之前就没调用过input来为当前线程添加预取任务）：
    # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件。拿取只代表逻辑上执行完，
    # 实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx
    # 📌分析：input和get是一一对应的，input将is_running置为true，get(调用wait)将is_running置为false。不调用get，is_running就不可能为false
    #         若发现is_running不为true，就说明之前根本就没执行过layer的预取
    
    # 1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
    # 2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
    # 3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 4.调用_wait将is_running 置为false，返回get_queue 的首个 (vt.idx，ev_swapin)
    # 5.self.compute_stream.wait_event(ev_swapin)
    # 6.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1相同
    def get(self, vt, suc_vt=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        assert isinstance(vt, vTask)
        # wait current one (if no current one, get this one)
        # 若当前没有正在GPU上分配vt的所有层
        if not self.is_running:
            # self.put_queue.add(vt)
            # 这意味着 syncpin_handler 这个线程开始执行vt的模型的从共享内存到固定内存的复制
            self.syncpin_handler.iput(vt)
            # 为向GPU复制W、B做准备工作：
            # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
            # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
            # 
            # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
            # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin_stream 这个流上等待
            # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
            # 4.将 (vt, ev_compute) 添加到 put_queue 中
            self.iput(vt)
        # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
        # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)，即正在或已经执行玩的swap_in事件，(vt.idx,ev_swapin)
        #   若 _thread_func, 即 swap_in 没在swapin_stream上分配完，会阻塞在 remove() 上
        cur_vt_idx, ev_swapin = self._wait()
        # 等待该vt上所有的层在GPU上完成分配空tensor
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # syncpin next one if exsits
        # 若给定了后继任务，
        if suc_vt is not None:
            self.syncpin_handler.iput(suc_vt)
        # 返回 cur_vt_idx
        return cur_vt_idx

class PrefetchLocalModelGPU_for_worker5_double_buffer(object):
    """ Handles flexible prefetch of local model (W,B) from pinned model in background thread.
        Step:
        1) main thread: allocate (W,B) on GPU
        2) prefetch thread: get sync'ed pinned model (W,B)
        3) prefetch thread: copy in (W,B) in swapin_cudaStream
        4) prefetch thread: turn on grad
        
        Assumption: 
        1) always in FIFO ordering. put each task, and get prefetched task. 
        2) use cuda event for unblocking next CPU ops

        NOTE: why main thread allocates? i) 'no memory reuse across different stream' -- PyTorch, ii) different threads use different CUDA address spaces
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, syncpin_handler, shared_model_nvme, local_model, layer_num, cpu_layer_id, rank, swapin_stream=None, compute_stream=None, nvprof=False):
        self.syncpin_handler = syncpin_handler
        self.shared_model_nvme = shared_model_nvme
        self.local_model = local_model # list
        self.rank = rank
        # 若没有给定一个stream，则在当前rank上创建一个新的CUDA stream
        # 目前来看，swapin_stream就是一个新的流
        self.swapin_stream = swapin_stream if swapin_stream is not None else torch.cuda.Stream(device=rank)
        # 该参数是给定了的，其实和不给定执行else一样，都是cuda上的默认流
        self.compute_stream = compute_stream if compute_stream is not None else torch.cuda.default_stream(device=rank)
        self.nvprof = nvprof
        assert rank == torch.cuda.current_device()
        # initialize
        self.put_queue = threadsafe_data_struct.Queue() # [vt1, ...] # between main and prefetch thread
        self.get_queue = threadsafe_data_struct.Queue() # [vt1.idx, ...] # between prefetch and main thread
        self.prefetch_thread = threading.Thread(target=self._thread_func)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
        
        self.is_running = False

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id
        # print("[PrefetchLocalModelGPU] rank{} started prefetch_thread".format(self.rank))

    # input就是向put_queue中加元素，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    #
    # 为向GPU复制W、B做准备工作：
    # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
    # 2.在默认流上（ev_compute）记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
    # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
    # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
    def iput(self, vt):
        ''' Call by main thread. Blocking Alloc & Nonblocking CopyIn.
            Assumption: only one iput can be running. '''
        if vt is None:
            return
        # 断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
        assert not self.is_running, "the prefetch is still running"
        # 将 self.is_running 标志设置为 True，表示当前有 iput 操作正在执行。
        self.is_running = True
        assert isinstance(vt, vTask)
        # record previous compute event for swapin stream to wait
        # 在 compute_stream （默认流）上记录事件 ev_compute，若参数为空分配一个新的event，用于后续在 swapin 流上等待
        ev_compute = self.compute_stream.record_event()
        # allocation in main thread
        if self.nvprof: nvtx_range_push("task{}({}) Alloc(W,B)".format(vt.idx, vt.show_layers())) 

        # 若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上(pinned memory)参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
        for l in vt.layers: # verified: vt.In['W'].keys==vt.In['B'].keys==vt.layers
            # 若该层的W和B的媒介为SHM
            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                # 分配模型的参数和buffer，即按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor（分配内存不初始化值）
                # 📍
                if l in self.cpu_layer_id:
                    self.local_model[l].alloc_param_buf()
                    continue
                else:
                    self.local_model[l].alloc_param_buf_2(vt.idx, l)
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")
            
        # do the rest in background thread
        # 将(vt, ev_compute)添加到 put_queue 中
        self.put_queue.add((vt,ev_compute))
        if self.nvprof: nvtx_range_pop() 

    # 不断尝试从put_queue中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B）)，执行（若队列是空的会被阻塞）：
    # 1.从 put_queue 队列中弹出 (vt, ev_compute)，若队列没有元素会被阻塞在这
    # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
    # 3.在 CUDA 流 self.swapin_stream 上等待事件 ev_compute 的完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
    #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
    # 4.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 5.在当前 CUDA 流上记录一个事件 ev_swapin
    # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task 
            # 从 put_queue 队首弹出一个元素，若队列是空的，会被阻塞在这里
            vt, ev_compute = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) CopyIn(W,B)".format(vt.idx, vt.show_layers())) 
            # get sync'ed pinned model 
            # 返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。
            # 即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
            # 📌根据syncpin_handler中的注释来看，从共享内存到固定内存的同步操作是阻塞的（尽管代码上直观来看是非阻塞的），所以这里一定执行完了
            #   从另一个角度看，这个get()本身就是在等待sync线程执行完一次同步操作
            syncpin_vt_idx, buffer_id = self.syncpin_handler.get()
            # 确保当前在GPU上初始化W和B的任务和共享层到pinned层复制任务的层，目标是同一个vt
            # 下面就是要把该任务对应的layer pack的W和B swap in到GPU上
            assert syncpin_vt_idx == vt.idx
            # let swapin stream waits for this compute event 
            # 等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
            # 即等待任务vt的所有层已经拿到GPU上
            self.swapin_stream.wait_event(ev_compute) 
            # copy in and turn on grad
            # 在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
            # 显示的设置参数 param 的 requires_grad 属性为 True
            with torch.cuda.stream(self.swapin_stream): # context-manager selects a given stream. All CUDA kernels queued within its context will be enqueued on a selected stream.
                for l in vt.layers:
                    # 若W和B在的通讯媒介为共享内存，将cpu内存model中的参数和缓冲区数据拷贝到gpu model上
                    if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                        # 📌
                        print(f"rank{self.rank}, vt.idx:{vt.idx}({vt.layers}), {vt.type}, {l}, 开始cpu->gpu的复制")
                        self.local_model[l].copyin_param_buf_for_double_buffer(buffer_id, vt.idx, l)
                        # 若vt是BWD任务，还需将模型的参数设置为需要梯度
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    # 若W和B被pin在设备上，显然不用执行拷贝操作
                    elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                        if vt.type == 'BWD':
                            self.local_model[l].set_param_requiresgrad()
                    else: # P2P
                        raise ValueError("Underdevelopment")
            # record this swapin event for compute stream to wait
            # 在当前 CUDA 流上记录一个事件
            ev_swapin = self.swapin_stream.record_event() 

            # 📌完成复制后, FWD即可释放pinned buffer
            if vt.type == 'FWD':
                print(f"rank:{self.rank}, vt.layers:{vt.layers}, vt.Out:{vt.Out}")
                # all_pin = all(vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN' for l in vt.layers)
                not_all_pin = all(not (l in vt.Out['W']) and not (l in vt.Out['B']) for l in vt.layers)
                if not_all_pin:
                    self.shared_model_nvme.release_buffer(buffer_id)
                    print(f"rank:{self.rank}, ---------------------FWD成功释放buffer_id:{buffer_id}")
                else:
                    if all(vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN' for l in vt.layers):
                        print(f"rank:{self.rank}, ---------------------FWD使用PIN媒介, 保留buffer_id:{buffer_id}")
                    
            # if MEMCPY_NONBLK: self.swapin_stream.synchronize() # Wait for all the kernels in this stream to complete.
            # ready to use
            # 将任务的idx和 swap_in 事件以元组的形式加入到 get_queue 中
            self.get_queue.add( (vt.idx,ev_swapin) )
            if self.nvprof: nvtx_range_pop() 

    # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
    # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)
    def _wait(self):
        ''' Wait for the running iput. Called in main thread.
            Assumption: only one iput can be running. '''
        assert self.is_running, "no running prefetch"
        self.is_running = False
        return self.get_queue.remove()

    # 该函数附带保险操作（实际上可能之前就没调用过input来为当前线程添加预取任务）：
    # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件。拿取只代表逻辑上执行完，
    # 实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx
    # 📌分析：input和get是一一对应的，input将is_running置为true，get(调用wait)将is_running置为false。不调用get，is_running就不可能为false
    #         若发现is_running不为true，就说明之前根本就没执行过layer的预取
    
    # 1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
    # 2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
    #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
    # 3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
    #   显示的设置参数 param 的 requires_grad 属性为 True
    # 4.调用_wait将is_running 置为false，返回get_queue 的首个 (vt.idx，ev_swapin)
    # 5.self.compute_stream.wait_event(ev_swapin)
    # 6.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1相同
    def get(self, vt, suc_vt=None):
        ''' Call by main thread. Blocking wait until current prefetch finish. Then launch next sync pin. Return current vt.idx. '''
        assert isinstance(vt, vTask)
        # wait current one (if no current one, get this one)
        # 若当前没有正在GPU上分配vt的所有层
        if not self.is_running:
            # self.put_queue.add(vt)
            # 这意味着 syncpin_handler 这个线程开始执行vt的模型的从共享内存到固定内存的复制
            self.syncpin_handler.iput(vt)
            # 为向GPU复制W、B做准备工作：
            # 在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
            # 将(vt,ev_compute（就是该初始化事件）)加入到 put_queue中，这表示该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
            # 
            # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
            # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin_stream 这个流上等待
            # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
            # 4.将 (vt, ev_compute) 添加到 put_queue 中
            self.iput(vt)
        # 1.在input操作正在执行的过程中，将 is_running 置为false，表示没有正在执行的prefetch操作
        # 2.返回线程安全队列 get_queue 的首个 (vt.idx，ev_swapin)，即正在或已经执行玩的swap_in事件，(vt.idx,ev_swapin)
        #   若 _thread_func, 即 swap_in 没在swapin_stream上分配完，会阻塞在 remove() 上
        cur_vt_idx, ev_swapin = self._wait()
        # 等待该vt上所有的层在GPU上完成分配空tensor
        self.compute_stream.wait_event(ev_swapin) # Makes all future work submitted to compute stream wait for this swapin event
        # syncpin next one if exsits
        # 若给定了后继任务，
        if suc_vt is not None:
            self.syncpin_handler.iput(suc_vt)
        # 返回 cur_vt_idx
        return cur_vt_idx
