# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import os
import sys
import argparse
import json
import numpy as np
import gc
from copy import deepcopy
from collections import OrderedDict as ODict

import torch
from torch.autograd import Variable

from time import perf_counter as pc

from prof_data_struct import *

# 根据meta生成tensor
# 根据输入的 meta，创建一个具有指定形状、数据类型和设备的张量，并根据 requires_grad 参数设置生成的tesnor是否需要梯度
# 注意，即使ubatchsize的大小为1，也会修改meta中保存的 shape 成员变量，即[1024]->[ubatchsize, 1024]
def realize_TensorMeta(meta, ubatchsize=-1, requires_grad=False, force_dtype=None, device="cuda:0", use_rand=True):
    assert type(meta) is TensorMeta
    # 若meta当前记录的输入只是一个单个的sample，并非一个batch，将meta中保存的输入的shape变为ubatchsize的大小，即
    # [1024]->[ubatchsize, 1024]。同时将is_ubatch这个成员变量置为true，即输入是一个batch
    meta.add_ubatch(ubatchsize) # in-place add ubatchsize to shape if not there
    # 若没设置 force_dtype 这个参数，取元数据保存的输入的 dtype
    dtype = force_dtype if force_dtype is not None else meta.dtype
    # 根据 meta (元数据)，创建一个具有指定形状、数据类型和设备的张量
    if dtype == torch.float32:
        tensor = Variable(torch.rand(meta.shape, dtype=torch.float32, device=device)) if use_rand else Variable(torch.ones(meta.shape, dtype=torch.float32, device=device))
    elif dtype == torch.int64:
        tensor = Variable(torch.randint(low=0,high=2,size=meta.shape,dtype=torch.int64, device=device)) if use_rand else Variable(torch.ones(meta.shape,dtype=torch.int64, device=device))
    else:
        raise ValueError("unknown X.dtype={}".format(meta.dtype))
    # 根据 requires_grad 参数设置张量是否需要梯度
    tensor.requires_grad_(requires_grad)
    # 如果需要梯度，保留张量的梯度，以便在反向传播过程中使用
    if requires_grad:
        tensor.retain_grad()
    return tensor

# 根据ubatchsize, vlayer_id, X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor，
# 返回一个有序字典：{name: 生成的tensor}，顺序即参数names中name的顺序
# 1.返回self.stats[ubatchsize][vlayer_id][name]，name就是该层的输入名，这个返回的东西是一个 TensorMeta，即tensor的元数据
# 2.根据meta（元数据，TensorMeta）生成真实的tensor
def realize_X(XMETA, ubatchsize, vlayer_id, names, requires_grad=False, device="cuda:0", use_rand=True):
    named_tensors = ODict() # { name : tensor or const or [tensor,tensor] }
    for name in names: # [XMETA.get(ubatchsize, vlayer_id)[name] for name in X_names]
        # 返回self.stats[ubatchsize][vlayer_id][name]，name就是该层的输入名
        # 这个返回的东西是一个 TensorMeta，即tensor的元数据
        meta = XMETA.get(ubatchsize, vlayer_id)[name]
        # 若输入的名称以 input 为开头，requires_grad参数置为false
        if name.startswith("input"): # in ["input0","input1","input2","input_ids"]: # TODO: add identity chain to input
            requires_grad = False
        if type(meta) is TensorMeta:
            # 根据meta生成真实的tensor
            # 根据输入的 meta，创建一个具有指定形状、数据类型和设备的张量，并根据 requires_grad 参数设置生成的tesnor是否需要
            # 梯度。📌注意，即使ubatchsize的大小为1，也会修改meta中保存的 shape 成员变量，即[1024]->[ubatchsize, 1024]
            # 最后返回生成的tensor
            named_tensors[name] = realize_TensorMeta(meta, ubatchsize, requires_grad, device=device, use_rand=use_rand)
        elif type(meta) is ConstMeta: # output of size(int)
            named_tensors[name] = meta.const
        elif type(meta) is list: # output tuple of bert pretrainhead 
            named_tensors[name] = [realize_TensorMeta(m, ubatchsize, requires_grad, device=device, use_rand=use_rand) for m in meta]
        else:
            raise ValueError("unknown meta={}".format(meta))
    return named_tensors

def realize_D(TMETA, ubatchsize, device="cuda:0", use_rand=True): 
    return realize_X(TMETA, ubatchsize, 0, TMETA.get_names(ubatchsize, vlayer_id=0), requires_grad=False, device=device, use_rand=use_rand)

# 根据TMETA中保存的target tesnor的元数据随机生成一个tensor，放在字典里返回。{"label": 生成的tensor}
def realize_T(TMETA, ubatchsize, device="cuda:0", use_rand=True):
    return realize_X(TMETA, ubatchsize, TMETA.last_vlayer_id, TMETA.target_names, requires_grad=False, device=device, use_rand=use_rand)
    

def realize_dX(XMETA, ubatchsize, vlayer_id, names, device="cuda:0", use_rand=True): # excluding T
    named_gradients = ODict() # { name : tensor or None or [tensor,tensor] }
    for name in names: # [XMETA.get(ubatchsize, vlayer_id)[name] for name in X_names]
        meta = XMETA.get(ubatchsize, vlayer_id)[name]
        # if name in ["input0","input1","input2"]: # TODO: add identity chain to input
        #     requires_grad = False
        if type(meta) is TensorMeta:
            assert meta.is_ubatch
            named_gradients[name] = realize_TensorMeta(meta, requires_grad=False, force_dtype=torch.float32, device=device, use_rand=use_rand)
        elif type(meta) is ConstMeta: # output of size(int)
            named_gradients[name] = None
        elif type(meta) is list: # output tuple of bert pretrainhead 
            named_gradients[name] = [realize_TensorMeta(m, requires_grad=False, force_dtype=torch.float32, device=device, use_rand=use_rand) for m in meta]
        else:
            raise ValueError("unknown meta={}".format(meta))
    return named_gradients


class Profiler(object):
    def __init__(self, model, optimizer=None, compute_loss=None, offload_optim=True, device='cuda:0', verbose=False):
        self.model = model
        self.optimizer = optimizer
        # NOTE: safe to self.model and self.optimizer? 
        #       - yes for profile_forward and profile_backward (stateless)
        #       - no for profile_update (modified model and optimizer state) (so leave update to last phase)
        self.compute_loss = compute_loss
        self.offload_optim = offload_optim
        self.device = device
        self.verbose = verbose
        
        # clean up model grad and graph
        # 清空模型每个vlayer的梯度，并将参数从计算途中分离，并执行垃圾回收和清空cuda缓存
        self.del_model_grad()

    # 将层输出的name和tensor加入到有序字典 named_tensors 中
    def _save_Y_tensors_to_named(self, Y_names, Y_tensors, named_tensors):
        assert type(Y_tensors) is list
        if len(Y_names) == 1 and len(Y_tensors) > 1:
            named_tensors[Y_names[0]] = Y_tensors
        else:
            for name, tensor in zip(Y_names, Y_tensors):
                named_tensors[name] = tensor

    # 用于将模型层的参数缓冲区加载到 CUDA 设备上，并根据函数的第二个参数 设置参数是否需要梯度
    @torch.no_grad()
    def _swapin_param_buf(self, vlayer, requires_grad=False): 
        # 将模型层 vlayer 的参数加载到 CUDA 设备上
        vlayer.cuda()
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():
                if param is not None:
                    # 断言参数的梯度为 None，并且不需要梯度
                    assert param.grad is None and (not param.requires_grad), \
                    "swapin requires no grad for both FWD and BWD (param={}, param.grad={}, param.requires_grad={})".format(param, param.grad, param.requires_grad) 
                    # 设置参数是否需要梯度
                    param.requires_grad_(requires_grad)

    # 1.若vlayer的梯度不为空，将其梯度置为None
    # 2.将参数从计算图中分离（detach），使其成为叶子节点，并且不再保留梯度信息。这样做可以避免梯度的传播
    # 3.执行垃圾回收并清空cuda缓存
    @torch.no_grad()
    def _del_grad(self, vlayer, manual_gc=False):
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters(): 
                if param is not None:
                    if param.grad is not None:
                        param.grad = None
                    # 将参数从计算图中分离（detach），使其成为叶子节点，并且不再保留梯度信息。这样做可以避免梯度的传播
                    param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                    assert not param.requires_grad
        if manual_gc:
            # gc.collect()用于执行Python的垃圾回收
            # torch.cuda.empty_cache()用于清空CUDA缓存，释放GPU上的内存
            gc.collect(); torch.cuda.empty_cache()

    # 清空模型每个vlayer的梯度，并将参数从计算途中分离，并执行垃圾回收和清空cuda缓存
    def del_model_grad(self):
        # 三个参数分别对应：vlayer类、输入名称列表、输出名称列表
        for vlayer, _, _ in self.model:
            # 1.若vlayer的梯度不为空，将其梯度置为None
            # 2.将参数从计算图中分离（detach），使其成为叶子节点，并且不再保留梯度信息。这样做可以避免梯度的传播
            # 3.执行垃圾回收并清空cuda缓存
            self._del_grad(vlayer, manual_gc=True)

    # 分两种情况，所有层的前向/后向计算 和 损失函数的计算
    # 1.根据元数据在GPU上随机生成一个tensor作为层的输入，若是最后计算损失的层，还要实例化一个target tensor，即标签tensor
    #   📌注意，无论ubatchsize是多大，一开始meta的shape都只是1维的，该函数内部会把第一个维度变为ubatchsize的大小
    # 2.计算层的执行时间
    # 3.将输出保存起来，因为若函数最后一个参数置为true(即进行的是反向计算)，这个输出需要返回
    # 4.若当前层不是计算损失的那一层，则根据输出实例化一个元数据，因为该层的输出就是下一层的输入，下一层执行该函数时
    #   需要根据这个元数据随机创造一个输入tensor
    # 📌需要注意的是，requires_grad为True，代表该函数是在检测BWD的过程中被调用的，此时FWD计算的时间要加入到BWD中，
    #   即最终的BWD执行时间包含了重计算当前层的时间
    def _vlayer_forward_an_ubatch(self, ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=False):
        if vlayer_id != len(self.model)-1: # not criterion yet
            # In {X}，在GPU上根据元数据生成一个tensor
            # 根据ubatchsize, vlayer_id, X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor用于当前层的前向计算
            # 返回一个有序字典：{name: 生成的tensor}，顺序即参数names中name的顺序
            # 📌注意，无论ubatchsize是多大，一开始meta的shape都只是1维的，该函数内部会把第一个维度变为ubatchsize的大小
            # 1.1.返回self.stats[ubatchsize][vlayer_id][name]，name就是该层的输入名，这个返回的东西是一个 TensorMeta，即tensor的元数据
            # 1.2.根据meta（元数据，TensorMeta）生成真实的tensor
            named_tensors = realize_X(XMETA, ubatchsize, vlayer_id, X_names, requires_grad, self.device)
            # Forward on GPU
            # print(f"生成输入X后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated()/1024/1024} ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved()/1024/1024}")
            torch.cuda.synchronize(self.device)
            t_start = pc() 
            # 2.执行当前层：取出刚刚生成的输入tensor，作为层的输入
            Y_tensors = vlayer(*[named_tensors[name] for name in X_names])
            torch.cuda.synchronize(self.device)
            t_end = pc() 
            # print(f"生成输出Y后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated()/1024/1024} ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved()/1024/1024}")
            # Result
            # 3.记录运行时间:
            # self.stats[FWD/BWD][ubatchsize][vlayer_id][device] += time
            TIME.add('FWD' if not requires_grad else 'BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
            # print("\t\t\tforward'ed trial:{}".format(tid))
            if not isinstance(Y_tensors, tuple):
                Y_tensors = (Y_tensors,)
            Y_tensors = list(Y_tensors)
            # Save {Y}
            # 4.将层输出的name和tensor加入到有序字典 named_tensors 中
            self._save_Y_tensors_to_named(Y_names, Y_tensors, named_tensors)
            # Out {Y} && {stashX}
            # 5.当前的输出即下一层的输入，为下一层的输入生成元数据并保存到XMETA中，📌下一层前向计算的时候就用这个元数据生成tensor
            # self.stats[ubatchsize][vlayer_id][names[0]] = TensorMeta (根据传入的Y_tensors实例化的元数据)
            XMETA.set(ubatchsize, vlayer_id+1, Y_names, Y_tensors)

        # 若到了最后一层
        else: # criterion
            assert Y_names == ["loss"]
            print("到最后一层了，vlayer为：", vlayer) # CrossEntropyLoss()
            print("X_names为: ", X_names) # ['out26']
            # In {X}
            # 1.对每一个输入tensor的name生成真实的tensor，返回一个有序字典：{name: 生成的tensor}，顺序即参数names中name的顺序
            named_tensors = realize_X(XMETA, ubatchsize, vlayer_id, X_names, requires_grad, self.device)
            print("named_tensors的形状为:", named_tensors[X_names[0]].shape)
            print(named_tensors[X_names[0]])
            # In {T}
            # 2.根据TMETA中保存的目标值tensor的元数据随机生成一个目标值tensor，放在字典里返回。{"label": 生成的tensor}
            named_targets = realize_T(TMETA, ubatchsize, self.device)
            print("named_targets的形状为:", named_targets[TMETA.target_names[0]].shape)
            # Forward on GPU
            torch.cuda.synchronize(self.device)
            t_start = pc()
            # 3.计算交叉熵损失
            if self.compute_loss is not None:
                # named_tensors：模型最后一层的输出
                # named_targets：目标值
                Y_tensors = self.compute_loss(vlayer, named_tensors, X_names, named_targets)
            else:
                Y_tensors = [vlayer(named_tensors[name],named_targets["target"]) for name in X_names]
                Y_tensors = [sum(Y_tensors)]
            torch.cuda.synchronize(self.device)
            t_end = pc() 
            # Result
            # 4.记录计算损失的时间:
            TIME.add('FWD' if not requires_grad else 'BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
            # print("\t\t\tforward'ed trial:{} loss = {}".format(tid, Y_tensors))
            # Save {Y}
            # 5.将层输出的name（["loss"]）和tensor加入到有序字典 named_tensors 中
            self._save_Y_tensors_to_named(Y_names, Y_tensors, named_tensors)
            del named_targets
        # Clean up
        del Y_tensors
        # return for backward pass
        # 若该项为true，说明执行的是反向传播，将该层的输入和输出（一个有序字典）传回来
        if requires_grad:
            return named_tensors
        else:
            del named_tensors

    # 1.准备backward函数的第二个参数
    # 2.对该层的输出tensor进行反向计算
    # 3.记录该层反向传播的时间
    def _vlayer_backward_an_ubatch(self, ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, named_tensors):
        # 1.准备backward函数的第二个参数
        # In {dY}
        # 若现在是计算损失的那一层，那么输出的loss是一个标量，backward函数不需要第二个参数，因此这里直接给了一个None
        if vlayer_id == len(self.model)-1: # criterion
            print("最后一层输出的loss长什么样：", named_tensors['loss']) # 最后一层输出的loss长什么样： tensor(10.8588, device='cuda:0', grad_fn=<NllLossBackward0>)
            #                              检查 named_tensors['loss'] 是否为 torch.Tensor 或 Variable 类型
            assert Y_names == ['loss'] and isinstance(named_tensors['loss'], (torch.Tensor,Variable))
            named_gradients = ODict({ 'loss': None })
            assert named_tensors['loss'].requires_grad
        else:
            # 使用后面那一层的输入的元数据生成tensor，即梯度的大小和当前层的输出一样大
            # ❓为什么梯度的大小和当前层的输出一样大？
            # 答：这不是梯度，而是作为backward()函数的第二个参数，用于执行雅可比向量积的，可以理解为设置了一个权重，
            # 用来调整各个因变量y对最终那个“标量梯度”的影响大小
            # 答：事实上使用当前层的输出也是一样的：vlayer_id, self.model[vlayer_id][2]
            named_gradients = realize_dX(XMETA, ubatchsize, vlayer_id+1, self.model[vlayer_id+1][1], self.device)
        # Backward on GPU
        Y_tensors = [] # 该层的输出tensor
        Y_gradients = [] 
        for name in Y_names: # only tensor & required_grad can run autograd
            # 拿到该层的输出tensor
            Y = named_tensors[name]
            if (type(Y) in [torch.Tensor, Variable]) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(named_gradients[name])
            elif type(Y) is list: # output tuple of bert pretrainheader
                for i, y in enumerate(Y):
                    if (type(y) in [torch.Tensor, Variable]) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(named_gradients[name][i])
        torch.cuda.synchronize(self.device)
        # print("该层输入计算梯度前的梯度为：", named_tensors[X_names[0]].grad) # None
        t_start = pc() 
        # 2.对该层的输出tensor进行梯度计算
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        torch.cuda.synchronize(self.device)
        t_end = pc() 
        # print("该层输入计算梯度后的梯度为：", named_tensors[X_names[0]].grad)
        # Result
        # 3.记录该层反向传播的时间
        TIME.add('BWD', ubatchsize, vlayer_id, 'GPU', t_end-t_start) 
        # print("\t\t\tbackward'ed trial:{}".format(tid))
        # Clean up {X,Y,dX,dY}
        del named_tensors; del named_gradients; del Y_tensors; del Y_gradients

    # 对每一个ubatchsize执行
    # profile每一层的执行时间和空间占用
    def profile_forward(self, ubatchsize_range, num_trials, TIME, MEMORY, XMETA, TMETA):
        # NOTE: no OoM allowed in this function
        if self.verbose: print("forward ...")
        for ubatchsize in range(*ubatchsize_range):
            print("\tubatchsize {} ...".format(ubatchsize))
            for vlayer_id, (vlayer, X_names, Y_names) in enumerate(self.model):
                if self.verbose: print("\t\tvlayer_id {}".format(vlayer_id))
                # Clean start
                gc.collect(); torch.cuda.empty_cache()
                # 等待设备上的所有流操作完成
                torch.cuda.synchronize(self.device)
                # memory_reserved：向CUDA申请的内存占用
                assert torch.cuda.memory_reserved()==0, "vlayer begin w/ alloc = {} B, resrv = {} B".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                # 将内存使用的峰值重置为当前值，这样在后续的代码执行过程中，可以准确地追踪内存的最高使用量
                torch.cuda.reset_peak_memory_stats() 
                # Swap-in model {W,B}
                # 用于将模型层的参数缓冲区加载到 CUDA 设备上，并根据函数的第二个参数requires_grad 设置参数是否需要梯度
                self._swapin_param_buf(vlayer, requires_grad=False)
                
                # total_params = 0
                # total_memory_bytes = 0
                # for param in vlayer.parameters():
                #     if param is not None:
                #         num_params = param.numel()
                #         dtype = param.dtype
                #         memory_bytes = num_params * param.element_size()
                        
                #         total_params += num_params
                #         total_memory_bytes += memory_bytes
                        
                #         # print(f"参数名称: {name}, 参数量: {num_params}, 数据类型: {dtype}, 空间占用: {memory_bytes / (1024 ** 2):.6f} MB")
                # print(f"layer{vlayer_id}, 总参数量: {total_params}, 总空间占用: {total_memory_bytes / (1024 ** 2):.6f} MB\n")

                # print("==========================")
                # print(f"加载该层模型后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated()/1024/1024} ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved()/1024/1024}")
                # First iteration for MEMORY
                # 这一次前向只记录内存占用，尽管时间也记录了，但会被清空
                with torch.no_grad():
                    # 分两种情况，所有层的前向/后向计算 和 损失函数的计算
                    # 1.根据元数据随机生成一个tensor作为层的输入，若是最后计算损失的层，还要实例化一个target tensor，即标签tensor
                    #   📌注意，无论ubatchsize是多大，一开始meta的shape都只是1维的，该函数内部会把第一个维度变为ubatchsize的大小
                    # 2.计算层的执行时间
                    # 3.将输出保存起来，因为若函数最后一个参数置为true(即进行的是反向计算的重计算)，这个输出需要返回
                    # 4.若当前层不是计算损失的那一层，则根据输出实例化一个元数据，因为该层的输出就是下一层的输入，下一层执行该函数时
                    #   需要根据这个元数据随机创造一个输入tensor
                    # 📌需要注意的是，requires_grad为True，代表该函数是在检测BWD的过程中被调用的，此时FWD计算的时间要加入到BWD中，
                    #   即最终的BWD执行时间包含了重计算当前层的时间
                    self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, requires_grad=False)
                    gc.collect()
                # torch.cuda.max_memory_allocated()：获取当前进程中张量分配的最大内存量
                # 将当前ubatchsize, vlayer_id对应的最大内存使用量存入MEMORY中
                # 📌分析：此内存占用包含输入、模型、输出的占用，即全部加载一起的值
                MEMORY.set('FWD', ubatchsize, vlayer_id, torch.cuda.max_memory_allocated())
                # print(f"该层最终的峰值内存占用为：{torch.cuda.max_memory_allocated()/1024/1024}")
                # print("==========================")
                # print("vlayer_id: ", vlayer_id," ,torch.cuda.max_memory_allocated():", torch.cuda.max_memory_allocated())
                # Then iterations for TIME
                # self.stats['FWD'][ubatchsize][vlayer_id][device] = 0.0
                TIME.reset('FWD', ubatchsize, vlayer_id, 'GPU')
                # 这次前向会记录层的执行时间
                for tid in range(0, num_trials): # each trial is one microbatch 
                    with torch.no_grad():
                        self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=False)
                        gc.collect()
                # Swap-out {W,B}
                # 将层放回到cpu上
                vlayer.cpu()            

    # 对每一个ubatchsize执行：
    # 对每一层，先执行一次前向计算，拿到该层的输入输出tensor，因为反向计算就是对该层的输出tensor计算梯度。
    # 而后执行该层的反向计算，记录该层gpu内存的占用和反向传播的时间
    def profile_backward(self, ubatchsize_range, num_trials, TIME, MEMORY, XMETA, TMETA):
        # NOTE: no OoM allowed in this function
        if self.verbose: print("backward (with recompute) ...")
        for ubatchsize in range(*ubatchsize_range):
            print("\tubatchsize {} ...".format(ubatchsize))
            # 反方向执行所有layer
            for vlayer_id, (vlayer, X_names, Y_names) in reversed(list(enumerate(self.model))): # reverse all vlayer (layer)
                if self.verbose: print("\t\tvlayer_id {}".format(vlayer_id))
                # Clean start
                gc.collect(); torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
                # memory_reserved：当前已经在 CUDA 设备上分配但尚未使用的内存量
                assert torch.cuda.memory_reserved()==0, "vlayer begin w/ alloc = {} B, resrv = {} B".format(torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
                torch.cuda.reset_peak_memory_stats() 
                # Swap-in model {W,B}
                # 用于将模型层的参数缓冲区加载到 CUDA 设备上，并根据函数的第二个参数requires_grad 设置参数是否需要梯度
                self._swapin_param_buf(vlayer, requires_grad=True)
                print(f"加载该层模型后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated() / 1024 / 1024} MB ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved() / 1024 / 1024} MB")
                # First iteration for MEMORY
                # 这次执行只记录gpu内存占用信息
                # 先执行一遍该层的前向计算以拿到该层的输入、输出tensor，📌因为要对该层的输出进行反向计算
                named_tensors = self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, requires_grad=True) # X/Y { name : tensor or const or [tensor,tensor] }
                size_bytes = named_tensors[Y_names[0]].element_size() * named_tensors[Y_names[0]].nelement()  # 总字节数
                size_mb = size_bytes / 1024 / 1024  # 转换为MB
                print(f"输出激活的显存占用: {size_mb:.2f} MB")
                
                print(f"前向计算后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated() / 1024 / 1024} MB ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved() / 1024 / 1024} MB")
                # 1.准备backward函数的第二个参数
                # 2.对该层的输出tensor进行反向计算
                # 3.记录该层反向传播的时间
                self._vlayer_backward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, 0, TIME, XMETA, TMETA, named_tensors)
                print(f"反向计算后，torch.cuda.memory_allocated():{torch.cuda.memory_allocated() / 1024 / 1024} MB ,torch.cuda.memory_reserved():{torch.cuda.memory_reserved() / 1024 / 1024} MB")
                del named_tensors # very important!
                gc.collect()
                # 记录gpu内存使用量：将当前ubatchsize, vlayer_id对应的最大内存使用量存入MEMORY中
                MEMORY.set('BWD', ubatchsize, vlayer_id, torch.cuda.max_memory_allocated())
                # print("vlayer_id: ", vlayer_id," ,torch.cuda.max_memory_allocated():", torch.cuda.max_memory_allocated())
                # Then iterations for TIME
                TIME.reset('BWD', ubatchsize, vlayer_id, 'GPU')
                # 记录该层反向传播执行的时间
                for tid in range(0, num_trials): # each trial is one microbatch 
                    named_tensors = self._vlayer_forward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, requires_grad=True) # X/Y { name : tensor or const or [tensor,tensor] }
                    self._vlayer_backward_an_ubatch(ubatchsize, vlayer_id, vlayer, X_names, Y_names, tid, TIME, XMETA, TMETA, named_tensors)
                    del named_tensors # very important!
                    gc.collect()
                # Swap-out model {dW,W,B}
                self._del_grad(vlayer)
                vlayer.cpu()

    # 在cpu上为每一层都执行 num_trials 次参数的更新，并记录参数更新的时间
    @torch.no_grad()
    def profile_update(self, num_trials, TIME):
        if self.offload_optim:
            for vlayer_id, ((vlayer, _, _), optim) in enumerate(zip(self.model, self.optimizer)):
                if optim is not None:
                    if self.verbose: print("\tvlayer_id {}".format(vlayer_id))
                    # Traverse all trials; each trial is one vlayer update
                    for tid in range(0, num_trials):
                        # 将层所有参数的梯度随机初始化，类型为fp32，位于cpu上
                        for param in vlayer.parameters():
                            param.requires_grad_(True)
                            param.grad = torch.rand(param.data.shape, dtype=torch.float32, device="cpu")
                        # compute updated weight
                        # 执行参数更新
                        t_start = pc() 
                        optim.step()
                        optim.zero_grad()
                        t_end = pc() 
                        TIME.add('UPD', None, vlayer_id, 'CPU', t_end-t_start) 
                        # print("\t\tupdated on trial:{}".format(tid))
        else:
            raise NotImplementedError("update on GPU")
        # print("update done")

    # 使用最大的micro batch size跑一下完整的前后向
    # data_names: ["Input0"]
    # data_tensors：[tensor([1,1,1, ...,1,1,1])]，tensor长度1024
    # target_names: ["labels"]
    # target_tensors: [tensor([1,1,1, ...,1,1,1])]，tensor长度1024
    def initial_iteration(self, umax, data_names, data_tensors, target_names, target_tensors):
        ubatchsize_range = [umax, umax + 1, 1]
        TIME = Time(ubatchsize_range, ubatchsize_range, len(self.model))
        MEMORY = Memory(ubatchsize_range, ubatchsize_range, len(self.model))
        XMETA = XMeta(ubatchsize_range, len(self.model))
        TMETA = TMeta(ubatchsize_range, len(self.model))
        XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
        TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
        self.profile_forward(ubatchsize_range, -1, TIME, MEMORY, XMETA, TMETA)
        self.profile_backward(ubatchsize_range, -1, TIME, MEMORY, XMETA, TMETA)
        print("initial iteration finished at batchsize {}".format(umax))

    # 通过倍增法探测最大microbatch的大小
    # args.probe_what：FWD
    # data_names: ["Input0"]
    # data_tensors：[tensor([1,1,1, ...,1,1,1])]，tensor长度1024
    # target_names: ["labels"]
    # target_tensors: [tensor([1,1,1, ...,1,1,1])]，tensor长度1024
    def probe_max_ubatchsize(self, type, data_names, data_tensors, target_names, target_tensors):
        """ 
        Probe max microbatch size by multiplicative-increase
        
        NOTE: additive-increase/decrease is not used in practice, as it causes repeated program rerun 
        (esp., model's initialization overhead). This is due to the limitation of PyTorch: 
        Each OoM causes memory leak (https://github.com/pytorch/pytorch/issues/27600), and 
        rerun is the only way to recover full GPU memory after OoM.
        
        NOTE: Forward probing, backward probing, normal profiling need three seperate runs of 
        the entire python program, due to the above limitation. 
        """
        assert type in ('FWD', 'BWD')
        
        print("\n----- probing {}'s max microbatch size -----".format(type))
        
        ubatchsize, umax = 1, -1
        while True:
            print("{}: try ubatchsize {} ...".format(type, ubatchsize))
            try:
                # 这里感觉只是为了以后好扩展代码，实际这里只会执行ubatchsize
                ubatchsize_range = [ubatchsize, ubatchsize + 1, 1]
                print(f"ubatchsize_range:{ubatchsize_range}")
                # 初始化时间的统计信息：{ 'FWD'/'BWD' : { ubatchsize: { vlayer_id: { 'device': 0.xxx sec } } } }
                TIME = Time(ubatchsize_range, ubatchsize_range, len(self.model))
                # 初始化内存占用的统计信息：{ 'FWD'/'BWD' : { ubatchsize: { vlayer_id: xxx bytes } } }
                MEMORY = Memory(ubatchsize_range, ubatchsize_range, len(self.model))
                # ubatchsize_range是一个有三个值的列表，代表 first, last, step, 会被解包传给range
                # 为每一个ubatchsize设置一个有序字典ODict，该字典中的key为vlayer_id（从0开始），遍历每一个id，将值初始化为None
                XMETA = XMeta(ubatchsize_range, len(self.model))
                # 1.同上，初始化一个XMeta，因为XMeta是TMeta的父类
                # 2.遍历每一个{ubatchsize：ODict}的键值对，删除ODict中除最后一层外的每个vlayer_id的值
                # TMeta用来存最后一层（计算损失层）的target tensor
                TMETA = TMeta(ubatchsize_range, len(self.model))
                # 给 XMETA 中成员变量有序字典stats中的每一个ubatchsize的第0层（vlayer_id=0）赋值，即为第0层生成一个有序字典，
                # 字典的name为输入名称，值为根据第二个参数 data_tensors 生成的元数据，后面会根据元数据生成相同形状和类型的随机tensor
                XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
                # 为stats中的每一个ubatchsize的最后一层（vlayer_id=len(self.model)-1）赋值
                # 为最后一层赋值，在stats中，每个vlayer_id还是一个ODict，key为name，即输入的名字，value为TensorMeta，即tensor的元信息
                TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
                if type == 'FWD':
                    # 对每一个ubatchsize执行
                    # profile每一层的执行时间和空间占用
                    self.profile_forward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                    print("....................FWD MEMORY..............................")
                    print(MEMORY)
                elif type == 'BWD':
                    # ❓为啥还要执行一次正向
                    # 答：暂时的理解：前向若是OOM了也不用算后向了，直接终止了
                    self.profile_forward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                    # 反向检测的时间
                    # 对每一个ubatchsize执行：
                    # 对每一层，先执行一次前向计算，拿到该层的输入输出tensor，因为反向计算就是对该层的输出tensor计算梯度。
                    # 而后执行该层的反向计算，记录该层gpu内存的占用和反向传播的时间
                    self.profile_backward(ubatchsize_range, 1, TIME, MEMORY, XMETA, TMETA)
                    print("....................BWD MEMORY..............................")
                    print(MEMORY)
                umax = ubatchsize
            except Exception as e: 
                if 'CUDA out of memory' in str(e):
                    del e
                    break
                elif 'an illegal memory access' in str(e):
                    print(e)
                    del e
                    break
                else:
                    raise e
            ubatchsize *= 2
        
        print("--- {}'s max microbatch size = {} ---\n".format(type, umax))
        return umax

# 在cuda:0上进行profile，将收集的信息保存到.pickle文件中
# 1.model、输入数据、label数据、Profiler类的初始化
# 2.通过倍增法探测最大 microbatch 的大小（1，2，4，8）
# 3.使用刚刚得到的最大micro batch执行探测：
#   3.1.FWDBWD：收集每一层的前后向(的平均)执行时间、内存占用(最大内存使用量)、输入该层的数据的元数据、label tensor的元数据
#   3.2.UDP：收集每一层在cpu上的(平均)参数更新时间、参数相关的元数据、buffer相关元数据、优化器状态相关元数据
def run(args, synthetic_data, create_model, create_optimizer, compute_loss=None):
    
    assert torch.cuda.is_available()
    print(f"CUDA_VISIBLE_DEVICES环境变量: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    torch.cuda.set_device(0) # control by CUDA_VISIBLE_DEVICES
    device = "cuda:0"
    print("device: ", device)

    """ Initialize model. """
    # 1.使用Importliob模块，把第1步拆分层的代码导入进来，赋值给module
    # 2.从参数 args 中指定的配置文件中加载 GPT-2 模型的配置信息，创建一个 GPT2Config 对象并赋值给 config
    # 3.创建了一个交叉熵损失函数 CrossEntropyLoss 的实例
    # 4.实例化已被拆分的model
    # 5.为args添加新的值，即模型的config实例
    # 返回model
    model = create_model(args)
    print("model created")

    """ Initialize data. """
    # 初始化数据
    # 根据input和label的形状与类型，分别为这俩创建一个值全为 1 的张量。返回这俩tensor的名字和创建的张量
    # 📌这个数据不是拿来用的，而是根据其生成元数据，用的时候根据元数据（shape，类型）随机生成一个tensor
    data_names, data_tensors, target_names, target_tensors = synthetic_data(args)
    # print("data_tensors的形状:", data_tensors[0].shape)
    # print(data_tensors)
    # exit(0)
    
    """ Initialize Harmony. """
    # 初始化Profiler类，并清空模型每个vlayer的梯度，并将参数从计算图中分离，并执行垃圾回收和清空cuda缓存
    # no_offload_optim 默认为false，not后置为true，即卸载优化器
    p = Profiler(model, compute_loss=compute_loss, offload_optim=not args.no_offload_optim, device=device, verbose=args.verbose)

    """ Modes to profile. """ 
    # 通过倍增法探测最大 microbatch 的大小（1，2，4，8）,在这个过程中，同样收集了时间、内存等信息，📌但是并不保存这些信息
    if args.mode == "probe":
        # 通过倍增法探测最大 microbatch 的大小（1，2，4，8）
        # args.probe_what：FWD/BWD
        # data_names: ["Input0"]
        # data_tensors：[tensor([1,1,1, ...,1,1,1])]，tensor长度1024
        # target_names: ["labels"]
        # target_tensors: [tensor([1,1,1, ...,1,1,1])]，tensor长度1024
        umax = p.probe_max_ubatchsize(args.probe_what, data_names, data_tensors, target_names, target_tensors)
        assert umax > 0, "[Error] Invalid {}'s max microbatch size = {}. Likely that even microabatch size = 1 explodes the GPU memory.".format(args.probe_what, umax)
        # 将探测到了最大micro-batch的大小写入json文件。（这里umax是一个整数，直接将其写入.json文件中）
        save_prof_data_struct(umax, args.output_dir, 'probe_{}_umax{}'.format(args.probe_what, args.outname_suffix))
    
    # 使用刚刚得到的最大micro batch执行探测：
    # 1.FWDBWD：收集每一层的前后向(的平均)执行时间、内存占用(最大内存使用量)、输入该层的数据的元数据、label tensor的元数据
    # 2.UDP：收集每一层在cpu上的(平均)参数更新时间、参数相关的元数据、buffer相关元数据、优化器状态相关元数据
    # 📌这块才真正保存这些信息，且每个microbatchsize都保存了 1,2,3,4...，默认情况下并没有microbatchsize的跳跃
    elif args.mode == "normal":
        
        # what里默认有这个，执行
        if 'FWDBWD' in args.what:
            
            # get probed ubatchsize
            # 1.从文件中拿出刚刚测得的前向最大的micro batch和后向最大的micro batch
            fwd_umax = load_prof_data_struct(args.output_dir, 'probe_{}_umax{}'.format('FWD', args.outname_suffix), base_dir="my_prof") if args.fwd_umax == -1 else args.fwd_umax
            bwd_umax = load_prof_data_struct(args.output_dir, 'probe_{}_umax{}'.format('BWD', args.outname_suffix), base_dir="my_prof") if args.bwd_umax == -1 else args.bwd_umax
            print(f"fwd_umax:{fwd_umax}")
            print(f"bwd_umax:{bwd_umax}")
            # 后向的占用更大，这也符合论文中说的
            assert fwd_umax >= bwd_umax, "fwd_umax:{} v.s. bwd_umax:{}".format(fwd_umax, bwd_umax)
            
            # run initial iteration for starting cuda context
            # 使用最大的micro batch size跑一下完整的前后向
            p.initial_iteration(bwd_umax, data_names, data_tensors, target_names, target_tensors)
            
            # set ubatchsize_range for FWD and BWD 
            if args.ubatchsize_step >= 1.0:
                ubatchsize_step = int(args.ubatchsize_step)
            else:
                ubatchsize_step = max(int(float(args.ubatchsize_step) * min(fwd_umax, bwd_umax)), 1)
            fwd_ubatchsize_range = [1, fwd_umax + 1, ubatchsize_step]
            bwd_ubatchsize_range = [1, bwd_umax + 1, ubatchsize_step]
            print("\n----- normal profiling -----")
            print("forward microbatch sizes: [{}, {}) with a step size {}".format(fwd_ubatchsize_range[0], fwd_ubatchsize_range[1], fwd_ubatchsize_range[2]))
            print("backward microbatch sizes: [{}, {}) with a step size {}".format(bwd_ubatchsize_range[0], bwd_ubatchsize_range[1], bwd_ubatchsize_range[2]))
            print("-------------------------------\n")

            # profile FWD and BWD
            TIME = Time(fwd_ubatchsize_range, bwd_ubatchsize_range, len(p.model))
            MEMORY = Memory(fwd_ubatchsize_range, bwd_ubatchsize_range, len(p.model))
            XMETA = XMeta(fwd_ubatchsize_range, len(p.model))
            TMETA = TMeta(fwd_ubatchsize_range, len(p.model))
            XMETA.init_data_meta_on_vlayer0(data_names, data_tensors)
            TMETA.init_target_meta_on_last_vlayer(target_names, target_tensors)
            
            # args.num_trials是跑的iteration的次数，这样的话TIME里记录的时间是4次执行时间的和
            print("\n----- profiling forward -----")
            # 📌注意，fwd_ubatchsize_range中的每个microbatchsize都被测了
            p.profile_forward(fwd_ubatchsize_range, args.num_trials, TIME, MEMORY, XMETA, TMETA)
            print("\n----- profiling backward -----")
            p.profile_backward(bwd_ubatchsize_range, args.num_trials, TIME, MEMORY, XMETA, TMETA)
            
            # save results
            # 将FWD/BWD每个ubatchsize每层记录的总的时间除上args.num_trials，即将总的时间替换为平均时间
            TIME.avg_trials(args.num_trials)
            
            if args.verbose:
                print(TIME)
                print(MEMORY)
                print(XMETA)
                print(TMETA)
            
            # 将几个存着采集信息的结构存到 .pickle 文件中
            print()
            save_prof_data_struct(TIME, args.output_dir, "prof_TIME_FWDBWD{}".format(args.outname_suffix))
            save_prof_data_struct(MEMORY, args.output_dir, "prof_MEMORY_FWDBWD{}".format(args.outname_suffix))
            save_prof_data_struct(XMETA, args.output_dir, "prof_XMETA{}".format(args.outname_suffix)) # NOTE: data shape is ubatched
            save_prof_data_struct(TMETA, args.output_dir, "prof_TMETA{}".format(args.outname_suffix)) # NOTE: target shape is ubatched
            print()

        # what里默认有这个，执行
        # 📌尽管只测了一次UDP，但UPD的时间与ubatch的大小无关，因为参数量永远不变，因此只在这测一次就够了
        if 'UPD' in args.what:
            if not args.no_offload_optim:

                """ Initialize optimizer. """
                # 初始化优化器：为model的每一层创建一个AdamW优化器，返回一个优化器列表
                p.optimizer = create_optimizer(p.model)
                print("optimizer created on CPU")

                # profile UPD
                TIME = Time(None, None, len(p.model))
                # 参数相关的元数据
                # 为model的每一层建立一个字典 {参数的名字：TensorMeta(参数的元数据)}
                # 为model的每一层建立一个字典 {vlayer_id: 参数的大小(bytes)}
                WMETA = WMeta(p.model)
                # buffer相关的元数据
                BMETA = BMeta(p.model)
                # 优化器状态相关的元数据
                KMETA = KMeta(p.model, p.optimizer)
                
                print("\n----- profiling update -----")
                # 在cpu上为每一层都执行 num_trials 次参数的更新，并记录参数更新的时间
                p.profile_update(args.num_trials, TIME)

                # save results
                # 平均参数更新的时间
                TIME.avg_trials(args.num_trials)

                if args.verbose:
                    print(TIME)
                    print(WMETA)
                    print(BMETA)
                    print(KMETA)

                # 将几个存着采集信息的结构存到 .pickle 文件中
                print()
                save_prof_data_struct(TIME, args.output_dir, "prof_TIME_UPD{}".format(args.outname_suffix))
                save_prof_data_struct(WMETA, args.output_dir, "prof_WMETA{}".format(args.outname_suffix))
                save_prof_data_struct(BMETA, args.output_dir, "prof_BMETA{}".format(args.outname_suffix))
                save_prof_data_struct(KMETA, args.output_dir, "prof_KMETA{}".format(args.outname_suffix))
                print()

            else:
                raise NotImplementedError("Update on GPU")
    else:
        raise ValueError
