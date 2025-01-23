# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import division, print_function
import argparse
import os
import sys
import json
import numpy as np
import gc
from collections import OrderedDict as ODict
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist
mp = torch.multiprocessing.get_context('spawn') # for GPU usage

def _assert_assumption(CONFIGS):
    # GPU only
    assert torch.cuda.is_available()
    # FP32 only
    torch.set_default_tensor_type('torch.FloatTensor')
    # Optimizer offload
    assert CONFIGS["opt_offld"]
    # double buffer need equal size
    if CONFIGS['mode'] == 'vPP':
        assert len(set(CONFIGS['ubatchszs_fwd'])) == 1
        assert len(set(CONFIGS['ubatchszs_bwd'])) == 1
    elif CONFIGS['mode'] == 'vDP': 
        for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
            assert len(set(ubatchszs_fwd)) == 1
            assert len(set(ubatchszs_bwd)) == 1
    else:
        raise ValueError
    # a single BWD task should have equal ubatchszs_fwd and ubatchszs_bwd per GPU
    if CONFIGS["pack_fwd"] == []: 
        assert CONFIGS["u_fwd"] == CONFIGS["u_bwd"]
        if CONFIGS['mode'] == 'vPP':
            assert CONFIGS['ubatchszs_fwd'] == CONFIGS['ubatchszs_bwd']
        elif CONFIGS['mode'] == 'vDP': 
            for ubatchszs_fwd, ubatchszs_bwd in zip(CONFIGS['ubatchszs_fwd'], CONFIGS['ubatchszs_bwd']):
                assert ubatchszs_fwd == ubatchszs_bwd
        else:
            raise ValueError

def worker_func(*pargs, **kwargs): # per process
    # 1.worker：harmony原版代码
    # 2.worker_moniter：监控内存使用情况
    from worker import Worker
    w = Worker(*pargs, **kwargs)
    # w.run_initial_iteration()
    w.run_training_loop()
    w.finish()
        
def run(args, real_dataset, create_model, create_optimizer, get_train_steps=None, get_lr_sched=None, compute_loss=None, save_model=None): # main process
    
    import seeding
    seeding.seed(args.seed, args.seed_cudnn)
    
    """ Initialize Harmony. """
    # 即装着切分成层的代码路径
    module_path = os.path.join(args.module_dir, args.module_name)
    assert os.path.exists(module_path)
    assert os.path.basename(module_path) not in ["prof", "sched"], "no base_dir in module_path"
    
    # read profiles
    from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta, load_prof_data_struct
    prof = ODict()
    print("........args.profile_fnames: ",args.profile_fnames)
    # ['prof_XMETA', 'prof_TMETA']
    for name in args.profile_fnames:
        key = name.split("prof_")[-1]
        # 加载保存在pickle文件中的数据结构
        prof[key] = load_prof_data_struct(module_path, name + args.suffix, base_dir="my_prof", verbose=True)
    
    # read schedule
    from task_data_struct import Medium, vTask, unserialize_scheduled
    # 在该例子中，这个参数就是空
    if args.schedule_dir == "":
        args.schedule_dir = module_path
    # 从指定路径的文件中反序列化（unserialize）调度（schedule）数据
    rTASKS, CONFIGS = unserialize_scheduled(args.schedule_dir, args.schedule_fname + args.suffix, base_dir="my_sched", verbose=False)
    _assert_assumption(CONFIGS)
    
    """ Initialize data. """
    if args.synthetic_data:
        args.num_epochs = 1
        assert args.num_iters is not None
        args.num_train_steps = args.num_iters
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))

    # 本例子会执行else的逻辑
    # 收集或计算训练的信息并打印出来，一次epoch的Iteration次数、epoch次数、总iteration次数
    else:
        # data_loader: 用于迭代数据集的DataLoader实例
        data_loader, examples, _, _, _, _, _ = real_dataset(args, CONFIGS["D"], data_workers=0)
        print(f"........len(data_loader):{len(data_loader)}")
        if get_train_steps is not None: # "bert_thomwolf"
            args.num_train_steps = get_train_steps(args, examples, CONFIGS["D"])
        else:
            # 训练的step数：minibatch的数量 × epoch次数
            args.num_train_steps = len(data_loader) * args.num_epochs
        # 设置一个epoch迭代的次数
        if args.num_iters is None:
            args.num_iters = len(data_loader) # num_minibatch
        else:
            # 若手动设置了一个epoch迭代的次数，这里要在手动设置的的次数和minibatch的数量间取最小值
            args.num_iters = min(args.num_iters, len(data_loader))
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num minibatches per epoch = %d" % len(data_loader))
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))
        del data_loader
    
    # 测试性能的工具
    if args.nvprof:
        assert args.num_epochs == 1, "num_epochs must be 1 during nvprof"
        if args.nvprof_iter == "first":
            args.nvprof_iter = { "start" : 0, "end" : 0 }
        elif args.nvprof_iter == "last":
            args.nvprof_iter = { "start" : args.num_iters - 1, "end" : args.num_iters - 1 }
        elif args.nvprof_iter == "all":
            args.nvprof_iter = { "start" : 0, "end" : args.num_iters - 1 } 
        else:
            raise ValueError

    """ Initialize model. """
    from utils import PrintCPUMem
    pcm = PrintCPUMem()
    pcm.print("before creating model")
    # 根据args动态地创建一个模型对象，并且可以选择加载预训练的模型参数
    model = create_model(args)
    pcm.print("model created")

    # 确保模型的所有参数都不在 GPU 上
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():
                # 如果参数在 GPU 上，断言将会触发异常，终止程序的执行
                assert not param.is_cuda

    for m in model[0][0].modules():
        for key, param in m._parameters.items():
            if param is not None:
                print(f"layer0 key,{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
        for key, buf in m._buffers.items():
            if buf is not None:
                print(f"layer0 buffer:{key}, {buf.numel()},{buf.shape}")

    total_size=0
    for name,param in model[0][0].named_parameters():
        print(name, param.numel())
        total_size+=param.numel()
    print(f"layer0: {total_size*4/1024/1024}")

    print()
    global_param_idx = 0
    for m in model[1][0].modules():
        print(f"xxxxxx: {m}")
        for key, param in m._parameters.items():
            if param is not None:
                print(f"key:{key},{global_param_idx}, {param},{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
                global_param_idx+=1
                
        for key, buf in m._buffers.items():
            if buf is not None:
                print(f"buffer:{key},{global_param_idx},{buf}, {buf.numel()},{buf.shape}")
                global_param_idx+=1
                

    total_size=0
    for name,param in model[1][0].named_parameters():
        print(name, param.numel())
        total_size+=param.numel()
    print(total_size*4/1024/1024)

    print("打印transformer layer的modules看看")
    for m in model[1][0].modules():
        print(f"xxxxxx: {m}")

    print("倒数第二层：")
    global_param_idx = 0
    for m in model[-3][0].modules():
        for key, param in m._parameters.items():
            if param is not None:
                print(f"倒数第二层 key,{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
                global_param_idx+=1
                print(global_param_idx)
        for key, buf in m._buffers.items():
            if buf is not None:
                print(f"倒数第二层 buffer:{key}, {buf.numel()},{buf.shape}")
                global_param_idx+=1
                print(global_param_idx)

    for name,param in model[-3][0].named_parameters():
        print(name, param.numel())

    print("最后一层：")
    global_param_idx = 0
    for m in model[-2][0].modules():
        for key, param in m._parameters.items():
            if param is not None:
                print(f"last key,{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
                global_param_idx+=1
                print(global_param_idx)
        for key, buf in m._buffers.items():
            if buf is not None:
                print(f"last buffer:{key}, {buf.numel()},{buf.shape}")
                global_param_idx+=1
                print(global_param_idx)

    total_size=0
    for name,param in model[-2][0].named_parameters():
        print(name, param.numel())
        total_size+=param.numel()
    print(f"last layer内存占用: {total_size*4/1024/1024}")

    # exit(0)
    
    # initialize empty model on CPU
    # 创建一个空model，📌这个model在后面就是GPU上的模型版本
    # 显然，删除了所有东西的model还在cpu上
    from local_model_gpu import delete_param_grad_buf
    empty_model = []
    for vlayer, _, _ in model:
        with torch.no_grad():
            vlayer_copy = deepcopy(vlayer)
        # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
        delete_param_grad_buf(vlayer_copy)
        empty_model.append(vlayer_copy)
    
    # initialize shared model on CPU  
    # model中的每一层都放进共享内存了
    for vlayer, _1, _2 in model:
        print("vlayer的类型是", type(vlayer),", 类名是:", vlayer.__class__.__name__)
        print(_1,_2)
        vlayer.share_memory() # move parameter into shared memory    
    pcm.print("shared model created")

    """ Initialize optimizer. """
    # 根据模型的每个层的参数情况创建相应的优化器，并返回一个装着所有层的优化器的列表
    optimizer = create_optimizer(args, model)
    pcm.print("optimizer created")
    
    # initialize shared optimizer on CPU
    from shared_optim_cpu import SharedOptimCPU
    shared_model = model # model is already shared
    shared_optimizer = [] # wrapper object for optimizer
    # 将所有层的优化器状态全部放入共享内存
    for id, ((vlayer, _, _), optim) in enumerate(zip(shared_model, optimizer)): 
        # 创建SharedOptimCPU的实例，其成员为放进共享内存的vlayer和其对应的优化器，且该优化器的优化器状态也全部在共享内存中。
        # 逻辑：将参数的梯度初始化为与参数数据形状相同的零张量，而后调用step()强制初始化优化器状态，而后把优化器状态全部放入共享内存
        # 1.将vlayer参数的梯度初始化为与参数数据形状相同的全零张量
        # 2.强制初始化优化器状态
        # 3.将优化器状态放入共享内存（甚至连step数也放进去了）
        # 4.将vlayer参数的梯度置为None
        shared_optimizer.append(SharedOptimCPU(vlayer, optim, id))
    pcm.print("shared optimizer created")

    """ Initialize distributed training. """ 
    gc.collect(); torch.cuda.empty_cache() 
    assert torch.cuda.memory_reserved() == 0, "fork process begins w/ alloc = {} B".format(torch.cuda.memory_reserved()) 
    
    processes = []
    # 若参数中设置了绑定gpu到cpu
    # 该参数默认为false，不执行
    print(f"是否使用参数 arg.numa_bind:{args.numa_bind}")
    if args.numa_bind:
        from utils import NumaBinder
        numa_binder = NumaBinder(args.numa_bind_config)
    for rank in range(CONFIGS["N"]):
        # target为该进程要运行的函数，args为target函数的输入参数
        p = mp.Process(target=worker_func, 
                        args=(args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, prof['XMETA'], prof['TMETA'], rTASKS, CONFIGS, rank),
                        name="rank%d"%rank)
        # NOTE: this moves parameter from pinned memory to shared memory
        p.start()
        processes.append(p)
        if args.numa_bind:
            numa_binder.bind(p, rank)
        
    if args.nvprof:
        from viewer.probe_cpu import ProbeCPU
        probe_cpu = ProbeCPU(pids=[p.pid for p in processes], 
                            ranks=[rank for rank in range(CONFIGS["N"])])
        probe_cpu.run(processes[0])
        print("--- rank -1: Done ---")
        print("--- all pids = (%s) ---"% " ".join("%d"%pid for pid in list([os.getpid()]+[p.pid for p in processes])) )

    for p in processes:
        p.join()
    print("--- all workers joined successfully. ---")
