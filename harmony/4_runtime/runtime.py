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


    from worker_no_gc_profile import Worker
    w = Worker(*pargs, **kwargs)
    # w.run_initial_iteration()
    w.run_training_loop()
    w.finish()

def run(args, real_dataset, create_model, create_optimizer, get_train_steps=None, get_lr_sched=None, compute_loss=None, save_model=None): # main process

    import seeding
    seeding.seed(args.seed, args.seed_cudnn)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    """ Initialize Harmony. """

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

        prof[key] = load_prof_data_struct(module_path, name + args.suffix, base_dir="my_prof", verbose=True)

    # read schedule
    from task_data_struct import Medium, vTask, unserialize_scheduled

    if args.schedule_dir == "":
        args.schedule_dir = module_path

    # args.suffix += "_bind_in_same_device"


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


    else:

        data_loader, examples, _, _, _, _, _ = real_dataset(args, CONFIGS["D"], data_workers=0)
        print(f"........len(data_loader):{len(data_loader)}")
        if get_train_steps is not None: # "bert_thomwolf"
            args.num_train_steps = get_train_steps(args, examples, CONFIGS["D"])
        else:

            args.num_train_steps = len(data_loader) * args.num_epochs

        if args.num_iters is None:
            args.num_iters = len(data_loader) # num_minibatch
        else:

            args.num_iters = min(args.num_iters, len(data_loader))
        print('----- Training Info -----')
        print("  num epoches = %d" % args.num_epochs)
        print("  num minibatches per epoch = %d" % len(data_loader))
        print("  num iterations per epoch = %d" % (args.num_iters))
        print("  num optimization steps = %d" % (args.num_train_steps))
        del data_loader


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

    model = create_model(args)
    pcm.print("model created")


    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) != 0:
            for param in vlayer.parameters():

                assert not param.is_cuda

    # for m in model[0][0].modules():
    #     for key, param in m._parameters.items():
    #         if param is not None:
    #             print(f"layer0 key,{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
    #     for key, buf in m._buffers.items():
    #         if buf is not None:
    #             print(f"layer0 buffer:{key}, {buf.numel()},{buf.shape}")

    layer0_size = 0
    print("layer0:")
    for name, param in model[0][0].named_parameters():
        print(name, param.numel())
        layer0_size += param.numel()

    # print()
    # global_param_idx = 0
    # for m in model[1][0].modules():
    #     print(f"xxxxxx: {m}")
    #     for key, param in m._parameters.items():
    #         if param is not None:
    #             print(f"key:{key},{global_param_idx}, {param},{param.numel()} ,{param.numel()*4/1024/1024}, {param.shape}")
    #             global_param_idx+=1

    #     for key, buf in m._buffers.items():
    #         if buf is not None:
    #             print(f"buffer:{key},{global_param_idx},{buf}, {buf.numel()},{buf.shape}")
    #             global_param_idx+=1


    total_size=0
    print("layer1:")
    for name,param in model[1][0].named_parameters():
        # print(name, param.numel())
        total_size+=param.numel()
    print(f"layer1的内存占用: {total_size*4/1024/1024} MB")

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

    last_layer_size = 0
    for name, param in model[-2][0].named_parameters():
        print(name, param.numel())
        last_layer_size += param.numel()
    print(f"layer0 与最后一层内存占用之和: {(layer0_size + last_layer_size)*4/1024**3:.2f} GB")


    total_model_params = 0
    print("统计整个模型的内存占用:")
    for layer_idx, (vlayer, _, _) in enumerate(model):
        layer_params = 0
        for param in vlayer.parameters():
            layer_params += param.numel()
        total_model_params += layer_params
        print(f"  层 {layer_idx}: {layer_params:,} 参数, {layer_params*4/1024/1024:.2f} MB")


    total_memory_gb = total_model_params * 4 / 1024 / 1024 / 1024
    print(f"整个模型总参数数量: {total_model_params:,}")
    print(f"整个模型内存占用: {total_memory_gb:.3f} GB")


    transformer_total_params = 0
    transformer_start_idx = 1
    transformer_end_idx = len(model) - 4
    for vlayer, _, _ in model[transformer_start_idx:transformer_end_idx]:
        for param in vlayer.parameters():
            transformer_total_params += param.numel()
    transformer_total_memory_mb = transformer_total_params * 4 / 1024 / 1024
    transformer_total_memory_gb = transformer_total_params * 4 / 1024 / 1024 / 1024
    print(f"所有 transformer layers 总参数数量: {transformer_total_params:,}")
    print(f"所有 transformer layers 内存占用之和: {transformer_total_memory_mb:.2f} MB ({transformer_total_memory_gb:.3f} GB)")

    # exit(0)

    # initialize empty model on CPU


    from local_model_gpu import delete_param_grad_buf
    empty_model = []
    for vlayer, _, _ in model:
        with torch.no_grad():
            vlayer_copy = deepcopy(vlayer)

        delete_param_grad_buf(vlayer_copy)
        empty_model.append(vlayer_copy)

    # initialize shared model on CPU

    for vlayer, _1, _2 in model:
        print("vlayer的类型是", type(vlayer),", 类名是:", vlayer.__class__.__name__)
        print(_1,_2)
        vlayer.share_memory() # move parameter into shared memory
    pcm.print("shared model created")

    """ Initialize optimizer. """

    optimizer = create_optimizer(args, model)
    pcm.print("optimizer created")

    # initialize shared optimizer on CPU
    from shared_optim_cpu import SharedOptimCPU
    shared_model = model # model is already shared
    shared_optimizer = [] # wrapper object for optimizer

    for id, ((vlayer, _, _), optim) in enumerate(zip(shared_model, optimizer)):


        shared_optimizer.append(SharedOptimCPU(vlayer, optim, id))
    pcm.print("shared optimizer created")

    """ Initialize distributed training. """
    gc.collect(); torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved() == 0, "fork process begins w/ alloc = {} B".format(torch.cuda.memory_reserved())

    processes = []


    print(f"是否使用参数 arg.numa_bind:{args.numa_bind}")
    if args.numa_bind:
        from utils import NumaBinder
        numa_binder = NumaBinder(args.numa_bind_config)
    for rank in range(CONFIGS["N"]):

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
