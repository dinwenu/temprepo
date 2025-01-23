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

import torch.cuda.profiler as cuda_profiler 
from torch.cuda.nvtx import mark as nvtx_mark 
from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 
from viewer.probe_cuda_mem import ProbeCudaMem 
from time import perf_counter as pc

import seeding, checker

from profiler import realize_X, realize_dX, realize_T
from task_data_struct import Medium, vTask
import shared_optim_cpu, local_model_gpu, msg_stash_x, ubatchsize_converter, swp_x, p2p
from decompose import decompose_minibatch
from tensor_helper import * 
from utils import *

import datetime
import time

class Worker(object): # each rank process
    def __init__(self, args, real_dataset, shared_model, shared_optimizer, empty_model, get_lr_sched, compute_loss, save_model, XMETA, TMETA, rTASKS, CONFIGS, rank):
        self.args = args
        self.shared_model = shared_model
        self.shared_optimizer = shared_optimizer
        self.compute_loss = compute_loss
        self.save_model = save_model
        self.XMETA, self.TMETA = XMETA, TMETA
        self.rTASKS, self.CONFIGS = rTASKS, CONFIGS
        self.rank, self.world_size = rank, CONFIGS["N"]
        self.verbose, self.nvprof = args.verbose, args.nvprof

        # worker process must re-seed
        seeding.seed(args.seed, args.seed_cudnn) 
        self.rand_state_train = seeding.RandState()

        # per-rank configs
        if CONFIGS['mode'] == 'vPP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd']
            print(f"ubatchszs_fwd_local:{CONFIGS['ubatchszs_fwd']}")
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd']
            print(f"ubatchszs_bwd_local:{CONFIGS['ubatchszs_bwd']}")
            self.minibatchsize_local = CONFIGS['D']
            print(f"minibatchsize_local:{CONFIGS['D']}")
        elif CONFIGS['mode'] == 'vDP':
            self.ubatchszs_fwd_local = CONFIGS['ubatchszs_fwd'][self.rank]
            self.ubatchszs_bwd_local = CONFIGS['ubatchszs_bwd'][self.rank]
            self.minibatchsize_local = sum(self.ubatchszs_fwd_local)
            assert self.minibatchsize_local == sum(self.ubatchszs_bwd_local)
        else:
            raise ValueError
        # 若前后向microbatch的list不一样，该项为true
        print("CONFIGS[\"u_fwd\"]:", CONFIGS["u_fwd"])
        print("CONFIGS[\"u_bwd\"]:", CONFIGS["u_bwd"])
        self.is_convert_ubs = True if CONFIGS["u_fwd"] != CONFIGS["u_bwd"] else False
        
        # Initialize the Gloo world first
        # 默认值为 "localhost"
        os.environ['MASTER_ADDR'] = args.master_addr
        # 默认值为 12345
        os.environ['MASTER_PORT'] = str(args.master_port)

        # 1.建立进程间的通信
        # backend：通信后端，gloo用于CPU
        # init_method：默认为 env://，表示使用读取环境变量的方式进行初始化。os.environ['MASTER_ADDR'] = 'localhost'和
        # os.environ['MASTER_PORT'] = '12345'，就是用来供给init_method使用的
        # 📌不同进程运行到dist.init_process_group(backend,  init_method,  rank,  world_size)的时候，会阻塞，直到确定所
        # 有进程都可以通信为止
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.world_size)      
        assert dist.get_rank() == self.rank and dist.get_world_size() == self.world_size
        print("rank%d (pid %d): initialized Gloo world. world_size %d" % (self.rank, os.getpid(), self.world_size))
        
        # Set up GPU
        torch.cuda.set_device(self.rank)
        
        # initialize dataset (must be local to be pinned)
        # 该参数默认为false，执行else
        if args.synthetic_data:
            self.data_loader = list(range(args.num_iters))
            self.data_ubatches, self.target_ubatches = synthesize_data(XMETA, TMETA, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, pin_memory=not args.no_pin_data)
        else:
            # self.bnames：{"is_data" = [True, False]， "name" = ["input0", "labels"]}
            self.data_loader, _, self.is_skip_minibatch, self.preprocess_minibatch, self.bnames, self.fdim, self.is_copy_minibatch = real_dataset(args, CONFIGS["D"], args.data_workers)
            self.data_ubatches, self.target_ubatches = None, None
        

        # 固定内存统一管理的逻辑现在这写一下
        # （1）获取当前rank上的所有层
        from task_data_struct import filter_tasks_by_attr_val
        bwd_tasks = filter_tasks_by_attr_val(self.rTASKS[self.rank], attr='type', value='BWD')
        local_layers = [layer for vt in bwd_tasks for layer in vt.layers]
        # （2）传给init_in_subproc方法。在下面的for循环中，是有id的，不过没用上。现在正好能用上了
        #      把id和local_layers一起传给这个方法, id只要在local_layers中，就深拷贝当前层作为pinned model
        print(f"rank:{self.rank}, 当前rank上的层为:{local_layers}")

        # initialize shared optimizer locally
        self.pcm = PrintCPUMem()
        self.pcm.print("rank%d: before initializing optimizer" % self.rank)
        lr_scheduler = []
        # 2.创建每一层的pinned memory版本（对每层来说，相当于复制出一个新的层）
        # 遍历shared_optimizer，深度拷贝其中的shared_memory变量（即共享内存中的layer），并将复制后的模型的参数和buffer
        # 移动到pinned memory中，作为一个新的成员变量，pinned_model
        for id, optim in enumerate(shared_optimizer):
            # args.no_pin_model：默认为false
            # args.no_pin_grad_buf：默认为false
            # 在当前process中初始化优化器
            # 1.保险操作，确保逻辑正确（确保参数和优化器状态都在shared memory上，另外确保param_groups[0]中除params这个key外其他的key的value不是tensor）
            # 2.深度拷贝model，其实就是一层（self.shared_model）
            # 3.📌将复制的层的参数和buffer移动到固定内存(第2步拷贝的model)中，以便更高效地将参数传输到 GPU，因为在传输时无需重新分配内存
            optim.init_in_subproc(self.rank, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf)
            # optim.init_in_subproc_2(id, local_layers, self.rank, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf)
            if get_lr_sched is not None: # "gpt2_huggingface"      
                # 若优化器存在，则创建一个带有 warm-up 的线性学习率调度器，否则往列表里添加一个None
                # 最后一个计算loss的层显然是None
                lr_scheduler.append(None if optim.shared_optimizer is None else 
                                    get_lr_sched(args, optim.shared_optimizer))
        self.pcm.print("rank%d: optimizer initialized" % self.rank)

        # initialize local model GPU 
        #
        # 3.创建每一层的GPU版本，参数和buffer都是空的（被替换为0张量）
        # 每一个layer都是独立的，有自己的优化器，对每一层执行：
        # 3.1.初始化一个类，过程为：将存放在这一层optimizer中的pinned_model，就是一个layer，即放在固定内存中的vlayer，
        #   复制到GPU上，然后赋给empty model对应的layer上。最后删除GPU上当前layer（empty_vlayer）的所有参数、梯度和缓冲区
        # 3.2.调用类中model成员（empty_vlayer）的train函数，即启用 batch normalization 和 dropout 
        # 3.3.将该类加入到list中
        # 最后，local_model包含了所有包含GPU上vlayer的类
        self.local_model = []
        for vlayer_id, (optim, (_,X_names,Y_names), empty_vlayer) in enumerate(zip(shared_optimizer, shared_model, empty_model)):
            # 1.保险操作：确保pinned_model（就是一个vlayer）的参数和buffer都在固定内存中
            # 2.将存放在cpu固定内存的vlayer的参数和buffer的data，使用.cuda()方法复制到GPU上，并赋给empty_vlayer(empty_model 对应的layer)的data上
            #   即现在empty_model的该层layer（empty_vlayer）存在于GPU上
            # 3.删除GPU上当前layer（empty_vlayer）的所有参数、梯度和缓冲区
            local_vlayer = local_model_gpu.LocalModelGPU(optim.pinned_model, optim.shared_model, empty_vlayer, vlayer_id, X_names, Y_names, self.rank, self.world_size, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf) 
            # local_vlayer = local_model_gpu.LocalModelGPU_2(optim.pinned_model, optim.shared_model, empty_vlayer, vlayer_id, X_names, Y_names, self.rank, self.world_size, no_pin_model=args.no_pin_model, no_pin_grad_buf=args.no_pin_grad_buf) 
            # 调用类中model成员（empty_vlayer）的train函数，即启用 batch normalization 和 dropout 
            local_vlayer.train() # shared_model/pinned_train.train() not necessary
            # 将该类加入到list中
            self.local_model.append(local_vlayer)
        self.pcm.print("rank%d: local model initialized" % self.rank)
        
        # initialize MSG stashing X on CPU
        # 4.初始化MSGX线程实例
        layer_X_names = ODict()
        for vlayer_id, (_,X_names,_) in enumerate(shared_model):
            layer_X_names[vlayer_id] = X_names
        # pin_memory：args.no_pin_x默认为false，即不会不pin x，而参数名字的语义本身与arg中的参数相反，因此取反为true，即pin x
        #
        # Handles gloo send/recv of stashing X between cpu processes. 
        # 提取MSGX的任务的信息，为当前rank上发送、接收MSGX的线程提前准备数据结构，重点为创建线程安全的发送/接收字典。而后
        # 创建并启动线程：创建一个发送线程，还要为所有向当前rank发送X的src rank创建接收线程，
        msg_stashx = msg_stash_x.MSGStashX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        
        # 将layer_id 和 named_tensors 加入到MSGStashX类的 send_dict 字典中（线程安全字典）
        swapout_stashx_output_fn = msg_stashx.isend

        # 若前后向microbatch的list不一样，该项为true
        if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
            # 
            # self.minibatchsize_local：minibatch size
            # CONFIGS['u_fwd']：fwd micro batch的大小
            # self.ubatchszs_fwd_local：fwd microbatchsize的列表
            # CONFIGS['u_bwd']：bwd micro batch的大小
            # self.ubatchszs_bwd_local：bwd microbatchsize的列表
            # msg_stashx.isend：上面那个方法（MSGStashX类的）
            # pack_ordering=False
            # pin_memory=not args.no_pin_x（true）
            # 
            # 实际调用的还是MSGstashX的线程发送tensor，即调用MSGstashX的isend方法，该类只相当于在正常的中间环节中添加一个额外
            # 的步骤，用于将更大的FWD用microbatch拆分为BWD用microbatch。
            # 分析：前向microbatch size默认是不能超过bwd microbatchsize的，道理很简单。若fwd小于bwd，该microbatch压根不能用于BWD的流程
            # 只能被当作剩余数据等待下一个iteration的新数据进来，组合成一个足够大的tensor，在此之前BWD无法进行，根本就无法进行正常训练
            stashx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_stashx.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
            # 将layer_id和input2：cpu_named_tensors加入到 input_queue 队列中，这意味着UBatchSizeConverter实例的线程
            # 将开始执行tensor大小的转换，而后将convert好的tensor列表加入到MSGstashX的send_ditc字典中，这也意味着
            # MSGstashX实例的线程将开始执行向目标rank的发送任务
            swapout_stashx_output_fn = stashx_ubs_converter.isend
        
        # initialize SWP locally
        # 多卡的vPP不用管这个
        local_x = None
        if (CONFIGS['mode'] == 'vPP' and CONFIGS['N'] == 1) or (CONFIGS['mode'] == 'vDP'):
            local_x = msg_stash_x.LocalX(self.rank, list(range(CONFIGS['R'])))
            swapout_localx_output_fn = local_x.isend
            if self.is_convert_ubs: # initialize Optional UBatchSize Converter on CPU
                localx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, local_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_localx_output_fn = localx_ubs_converter.isend
        
        # initialize P2P
        # 实例化一个P2P类，用于P2P通信
        self.p2px_handler, self.p2pm_handler = None, None
        if CONFIGS['mode'] == 'vPP' and CONFIGS['N'] > 1:
            # CONFIGS['reverse_bwd']：默认为false，该参数已被废弃，不用管
            # 
            # 1.访问每一个rank，在每个rank和其下一个rank间建立一个NCCL通信组。若当前rank包含在正在建立的通信组中，
            #   就为字典 self.groups 添加一个值：{ "r1->r2": dist.group_obj }
            # 2.
            #   2.1.创建一个包含单个元素的张量，用于初始化 NCCL 通信器
            #   2.2.将当前rank所在的通信组取出，进行一次r1->r2的点对点通信
            self.p2px_handler = p2p.P2PX(self.rank, self.world_size, CONFIGS['reverse_bwd'], verbose=self.verbose, nvprof=self.nvprof)
        elif CONFIGS['mode'] == 'vDP' and CONFIGS['N'] > 1:
            self.p2pm_handler = p2p.P2PModel(self.rank, self.world_size, verbose=self.verbose)

        # Get default cuda stream (already initialized by local_model_gpu)
        # 获取当前 CUDA 设备上的默认 CUDA 流
        # 📌在PyTorch中，默认情况下，GPU上的操作是在默认流（default stream）中执行的。默认流是一个序列化的流，
        # 其中的操作按照它们出现的顺序逐个执行。这意味着在没有显式指定其他流的情况下，所有的操作都会在默认流中执行。
        self.default_stream = torch.cuda.default_stream(self.rank)

        # initialize Update in Background thread
        # 开启一个update线程
        # 不断尝试从put_queue队列中拿取任务，对每个任务执行：
        # 1.等待当前stream中的所有操作全部完成，然后才会继续执行后续的代码。即等待dW,B'被swap到pinned memory中
        # 2.对给定vtask中所有layer执行更新buffer操作：将cpu上pinned memory中的数据复制到在共享内存的vlayer的 buffer 中
        # 3.更新shared memory上vt中每一层的参数，如果配置了学习率调度器，还要调用相应的学习率调度器来更新学习率
        # 4.将完成的任务的idx加入到get_queue队列中
        self.update_handler = shared_optim_cpu.UpdateInBkgd(self.default_stream, shared_optimizer, lr_scheduler, self.rank, nvprof=self.nvprof)

        # initialize Prefetch Model background thread
        # SyncPinModelInBkgd实例 用来辅助 PrefetchLocalModelGPU实例，即并行的完成共享到固定的复制

        # 开启一个同步pinned model的线程，即将shared memory中模型的参数和梯度复制到pinned memory的model中
        # 不断尝试从put_queue中拿取任务（vt），对拿到的vt执行：
        # 1.若W,B的媒介是SHM，将共享模型中的参数和缓冲区同步到本地的固定内存模型中(在pinned memory上)
        #   --若W和B已经被固定在rank上了，则什么都不用做
        # 2.将同步完成的vt的idx加入get_queue队列
        syncpin_handler = shared_optim_cpu.SyncPinModelInBkgd(shared_optimizer, self.rank, nvprof=self.nvprof)

        # 新建一个 swapin_stream ，专门用来 swap in model
        # 不断尝试从 put_queue 中拿取(vt, ev_compute（准备工作的事件，即在GPU上先初始化W和B的tensor）)，执行（若队列是空的会被阻塞）：
        # 1.从 put_queue 队列中弹出 (vt, ev_compute（准备工作的事件，即在GPU上先初始化W和Btensor）)，若队列没有元素会被阻塞在这
        # 2.返回syncpin_handler实例的get_queue的首个元素（任务的idx），并把其从队列中删除。即返回已经完成从共享模型到固定内存模型复制W,B的任务（即同步）
        # 3.等待事件 ev_compute 在 CUDA 流 self.swapin_stream 上完成，然后再继续执行后续的操作（后续提交给该stram的所有work都得等）
        #   即等待在GPU上初始化vt所有层的 W和B(的tensor) 的完成
        # 4.在swapin_stream中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
        #   显示的设置参数 param 的 requires_grad 属性为 True
        # 5.在当前 CUDA 流上记录一个事件 ev_swapin
        # 6.将 (idx(当前任务的id),ev_swapin) 加入到 get_queue 中
        self.prefetch_model_handler = local_model_gpu.PrefetchLocalModelGPU(syncpin_handler, self.local_model, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        
        # initialize SwapIn background thread
        # 新建一个swapin_stream，专门用来 swap in stashX
        # 不断从 put_queue 中拿取 layer_id, cuda_named_tensors, ev_compute 并执行：
        # 1.如果 ev_compute 不为 None，则在当前 CUDA 流上等待该事件的完成。这是为了确保在执行后续操作之前，
        #   必须等待先前的计算完成
        # 2.调用 msg_stashx.recv 方法，即拿到从src_rank穿来的 cpu_tensor，若没有tensor会被阻塞住
        #   2.1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
        #   2.2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
        #       即一个 （name, tensor）
        #       若内部没有tensor显然会被阻塞住(wait)
        # 3.将cpu上的tesnor拷贝到gpu上的tensor。该函数会返回一个记录的事件，用于让compute stream等待
        # 4.将 (cuda_named_tensors, ev_swapin) 放入 get_queue 队列中
        self.swapin_stashx_handler = swp_x.SwapIn(msg_stashx.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) 

        # 暂略，vDP专用：新建一个swapin_stream，专门用来 swap in LocalX(X/dx)
        self.swapin_localx_handler = swp_x.SwapIn(local_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof) if local_x is not None else None
        
        # initialize SwapOut background thread
        # 新建一个stream，专门用于 swap_out
        swapout_stream = torch.cuda.Stream(device=self.rank)
        # 用于offload vPP 中的 stashX
        self.swapout_stashx_handler = swp_x.SwapOut(swapout_stashx_output_fn, self.rank,    
                                    swapout_stream=swapout_stream,
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_stashx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof)
        
        # 暂略，vDP专用
        self.swapout_localx_handler = swp_x.SwapOut(swapout_localx_output_fn, self.rank, 
                                    swapout_stream=swapout_stream, 
                                    compute_stream=self.default_stream, 
                                    blocking=True if args.no_offload_localx else False,
                                    pin_memory=not args.no_pin_x,
                                    nvprof=self.nvprof) \
                                    if local_x is not None else None
        
        # initialize MSG X on CPU # NOTE: tentatively only for last FWD to first BWD
        # 📌这个MSGX实例专门用来管理最后一个前向任务向第一个BWD任务发送Y 和 第一个BWD任务接收最后一个FWD任务发送来的Y
        #    注意是在cpu上发送接收数据，非GPU通信
        #
        # Handles gloo send/recv of Y/dX between cpu processes. 
        # MSGStashX的子类
        # 1.创建一个发送数据的辅助线程 _send_helper_thread
        #   不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要接收最后一个fwd任务输出的Y的layer id。要发送的tensor都保存在
        #   send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
        #   要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
        #   新的tesnor
        # 2.为每个 src_rank，即给当前rank上第一个bwd任务发送Y的最后一个fwd任务所在的rank，创建一个接收数据的辅助线程
        #  _recv_helper_thread。不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
        #  即不断尝试从src_rank接收对应layer l的microbatch大小为u的输入X。📌若没有收到数据会阻塞住
        #  将读取到的 named_tensors 连同其 layer_id 构成一个元组加入到 self.recv_dicts[src_rank] 这个线程安全字典中
        msg_x = msg_stash_x.MSGX(self.rank, rTASKS, layer_X_names, XMETA, self.ubatchszs_bwd_local, 'pack-by-pack', pin_memory=not args.no_pin_x, nvprof=self.nvprof)
        
        # Call by upstream thread. Nonblocking send. 
        # 向MSGX实例的 send_dict 这个线程安全字典中的 odict[layer_id] 这个list添加：
        # self.odict[layer_id].append(named_tensors)
        swapout_msgx_output_fn = msg_x.isend
        self.swapout_msgx_handler, self.swapin_msgx_handler = None, None

        # 若当前rank上 不存在 最后一个FWD任务且输出Y的媒介为MSG，也不存在第一个BWD任务要接收发向自己的X（同样媒介为MSG），删除msg_x
        # 📌分析：vPP应该执行这个，因为vPP最后一个FWD发送Y的媒介，和第一个BWD任务接收X的媒介都为P2P
        if msg_x.has_no_send() and msg_x.has_no_recv():
            del msg_x; msg_x = None
        # 若当前rank上存在最后一个FWD任务要发送Y，且输出Y的媒介为MSG，但不存在第一个BWD任务要接收发向自己的X（同样媒介为MSG）
        elif not msg_x.has_no_send() and msg_x.has_no_recv(): # sender only
            if self.is_convert_ubs:
                msgx_ubs_converter = ubatchsize_converter.UBatchSizeConverter(self.rank, self.minibatchsize_local, CONFIGS['u_fwd'], self.ubatchszs_fwd_local, CONFIGS['u_bwd'], self.ubatchszs_bwd_local, msg_x.isend, pack_ordering=False, pin_memory=not args.no_pin_x, nvprof=self.nvprof)
                swapout_msgx_output_fn = msgx_ubs_converter.isend
            # 📌这个SwapOut相当于MSGX的上游线程，该线程等待swapout的事件完成再把tensor放到MSGX的send_dict中后，MSGX
            #   线程才会发送Y给其他rank
            # 创建并开启SwapOut线程
            # 确保tensor在完全offload到cpu后，再添加到MSGX线程的send_dict中，发送出去
            # 不断尝试从 put_queue 这个线程安全队列中拿取 layer_id, cpu_named_tensors, ev_swapout, flag，执行：
            # 1.等待ev_swapout事件执行完成
            # 2.调用 output_fn 函数，即 MSGX 实例的isend方法，向 send_dict 这个线程安全字典中的 odict[layer_id] 这个
            #   list添加：self.odict[layer_id].append(named_tensors)
            self.swapout_msgx_handler = swp_x.SwapOut(swapout_msgx_output_fn, self.rank, 
                                        swapout_stream=swapout_stream, 
                                        compute_stream=self.default_stream, 
                                        blocking=True if args.no_offload_msgx else False,
                                        pin_memory=not args.no_pin_x,
                                        nvprof=self.nvprof)
        # 若当前rank上不存在最后一个FWD任务，但存在第一个BWD任务，要接收其他rank向当前rank发送的X
        elif msg_x.has_no_send() and not msg_x.has_no_recv(): # recver only
            # 📌这个SwapIn线程相当于MSGX的下游线程，MSGX收到了传来的Y，SwapIn再把传来的tensor发到GPU上
            # 创建并开启SwapIn线程
            # 不断从 put_queue 中拿取 layer_id, cuda_named_tensors, ev_compute 并执行：
            # 1.如果 ev_compute 不为 None，则在当前 CUDA 流上等待该事件的完成。这是为了确保在执行后续操作之前，
            #   必须等待先前的计算完成
            # 2.调用 MSGX.recv 方法，即拿到从src_rank传来的 cpu_tensor，若没有tensor会被阻塞住
            #   2.1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
            #   2.2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
            #       即一个 （name, tensor）
            #       若内部没有tensor显然会被阻塞住(wait)
            # 3.将cpu上的tesnor拷贝到gpu上的tensor。该函数会返回一个记录的事件，用于让compute stream等待
            # 4.将 (cuda_named_tensors, ev_swapin) 放入 get_queue 队列中
            self.swapin_msgx_handler = swp_x.SwapIn(msg_x.recv, self.rank, swapin_stream=None, compute_stream=self.default_stream, nvprof=self.nvprof)
        else:
            raise NotImplementedError
        
        # initialize succesor info for all prefetch
        self.sucinfo = SucInfoForPrefetch(self.rank, rTASKS, XMETA)

        # exit(0)

    ################### Initial Iteration ###################
    # 对不包含计算loss层的vt执行:
    # 1.根据ubatch_size, l(vlayer_id), X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor
    # 2.用随机生成的tensor作为输入，执行一次vt的前向传播
    # 3.把 Y_names，Y_tensors 装到ODict中返回，若names只有一个，ODict中显然只有一个键值对，不然就是多个键值对
    # 4.FWD任务删除所有tesnor，若是BWD任务调用的该函数，返回随机生成的输入tensor字典、输出tensor字典
    # 对包含计算loss层的vt，区别在于：
    # 1.还需生成label tensor
    # 2.多了一步计算loss
    # 3.无论FWD还是BWD，最后都返回输入tensor字典、输出tensor字典
    # 4.输出的tensor字典是输出的loss
    def _initial_a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size, requires_grad=False, verbose=False, nvprof=False):
        # 对不包含计算loss层的vt执行
        if not vt.has_criterion:
            ### In {X}
            ### 1.根据ubatch_size, l(vlayer_id), X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            # 根据ubatch_size, l(vlayer_id), X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor，
            # 返回一个有序字典：{name: 生成的tensor}，顺序即参数names中name的顺序
            # 1.返回self.stats[ubatchsize][vlayer_id][name]，name就是该层的输入名，这个返回的东西是一个 TensorMeta，即tensor的元数据
            # 2.根据meta（元数据，TensorMeta）生成真实的tensor
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Compute forward pass on GPU
            ### 2.用随机生成的tensor作为输入，执行一次vt的前向传播
            if nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            # 为什么采用这种很绕的命名方式，我理解为节省空间，复用同一个变量
            Y_tensors = [X_named_tensors[name] for name in X_names]
            for l in vt.layers:
                Y_tensors = self.local_model[l](*Y_tensors)
                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if verbose: print("\trank{}: task{}({}) {}(#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            ### 3.把 Y_names，Y_tensors 装到ODict中返回，若names只有一个，ODict中显然只有一个键值对，不然就是多个键值对
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            ### Clean up
            ### 4.FWD任务删除所有tesnor，BWD任务返回随机生成的输入tensor字典、输出tensor字典
            del Y_tensors
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else:
                return X_named_tensors, Y_named_tensors
        else: # criterion pack
            assert requires_grad
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names
            X_named_tensors = realize_X(self.XMETA, ubatch_size, l, X_names, requires_grad, "cuda:%d"%self.rank, use_rand=False)
            ### Recompute on GPU
            # 该函数是forward时调用的，这里注释为recompute不合理
            if nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    Y_tensors = self.local_model[l](*Y_tensors)
                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors
            ### In {T}
            # 根据TMETA中保存的target tesnor的元数据随机生成一个tensor，放在字典里返回。{"label": 生成的tensor}
            T_named_tensors = realize_T(self.TMETA, ubatch_size, "cuda:%d"%self.rank, use_rand=False)
            ### Compute loss on GPU
            # 
            assert vt.layers[-1] == self.CONFIGS['R']-1
            # 
            last_vlayer = self.local_model[self.CONFIGS['R']-1]
            if self.compute_loss is not None: 
                # last_vlayer：最后一层，即计算loss的层
                # Y_named_tensors：倒数第二层输出的tensor字典，{名字：tensor}
                # Y_names：倒数第二层输出值的名称
                # T_named_tensors：{"label": tensor}
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if nvprof: nvtx_range_pop()
            ### Save Y
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    # 1.为backward函数的第二个参数准备一个值全为1的tensor，若是第一个BWD任务，即对loss求偏导，则无需准该tensor
    # 2.执行反向传播
    # 3.删除所有用到的tensor
    def _initial_a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size, X_named_tensors, Y_named_tensors, verbose=False, nvprof=False):
        ### In {dY}
        # 若是包含计算loss层的第一个BWD任务，用于backward第二个参数的tesnor直接设置为空，同时将loss tensor设置为requires_grad
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad

        # 若vt不是第一个BWD任务，为backward的第二个参数准备一个值全为1的tensor
        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            # 使用后面那一层的输入的元数据生成tensor，即梯度的大小和当前层的输出一样大
            # ❓为什么梯度的大小和当前层的输出一样大？
            # 答：这不是梯度，而是作为backward()函数的第二个参数，用于执行雅可比向量积的，可以理解为设置了一个权重，
            # 用来调整各个因变量y对最终那个“标量梯度”的影响大小
            # 答：事实上使用当前层的输出也是一样的：vlayer_id, self.model[vlayer_id][2]
            dY_named_tensors = realize_dX(self.XMETA, ubatch_size, l+1, self.local_model[l+1].X_names, device="cuda:%d"%self.rank, use_rand=False)
        ### Compute backward pass
        if nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        # 准备好反向传播要用的tesnor
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names:
            # 取出该vt最终输出的tensor
            Y = Y_named_tensors[name]
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): 
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])
        # 执行反向传播
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        if verbose: print("\trank{}: task{}({}) BWD(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        if nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        # 删除所有用到的tensor
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    # ❓为什么要先用假数据跑一次minibatch（1个iteration）
    def run_initial_iteration(self, verbose=False, nvprof=False):
        # 该参数默认为false，不执行
        if self.args.no_initial_iter:
            print("rank%d: --- No Initial Iteration ---" % self.rank)
            return

        print("rank%d: initial iteration starts"%(self.rank))
        assert dist.get_rank() == self.rank and torch.cuda.current_device() == self.rank
        # clean memory before start
        # torch.cuda.synchronize(self.rank)：等待当前GPU上所有的核函数执行完毕
        # dist.barrier()：在分布式环境中，同步所有进程，确保在执行后续操作之前，所有进程都已经完成了前面的工作
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        # 断言当前 GPU 设备上没有任何内存被保留
        assert torch.cuda.memory_reserved(self.rank)==0 # 查看总共占用的显存
        dist.barrier()
        # task starts 
        if nvprof:
            probe_cuda_mem = ProbeCudaMem(self.rank)
            probe_cuda_mem.start()  
            cuda_profiler.start()
            nvtx_mark("cudaProfilerStart") 
            print("rank%d: cuda profiler starts" % self.rank)    

        time_start = pc()    
        #     
        for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,task5,...] }
            if verbose: print("\trank{}: executing {}".format(self.rank, vt))
            if vt.type == 'FWD' and vt.is_gpu:
                # -----------------------------------------------      
                with torch.no_grad():
                    ### Swap-in model {W,B}
                    if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                    # 当前这个prefetch线程接收一个vt，开始这个vt的swap_in工作，即把vt所有层在cpu上的W、B拷贝到gpu上。相当于触发了当前线程的工作
                    # 若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)
                    # 分析：并行的做两件事，1）调用同步线程将vt中的这些在cpu共享内存中的模型复制到pinned memory上；2）在GPU上为vt中的这些层分配一个空tensor
                    #      第2件事不执行，_thread_func会阻塞在remove上。在以上两件事全部执行完后，才会执行 swap_in 操作
                    cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                    assert cur_vt_idx == vt.idx
                    if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                    if nvprof: nvtx_range_pop() 
                    ### Run through each microbatch in a data batch
                    # 对每一个 microbatch size 执行一次
                    for i, u in enumerate(vt.ubatchszs):# [u1, u2, u3] 
                        # 对不包含计算loss层的vt执行:
                        # 1.根据ubatch_size, l(vlayer_id), X_names从XMETA中把元数据拿出来，根据元数据生成一个随机的tensor
                        # 2.用随机生成的tensor作为输入，执行一次vt的前向传播
                        # 3.把 Y_names，Y_tensors 装到ODict中返回，若names只有一个，ODict中显然只有一个键值对，不然就是多个键值对
                        # 4.FWD任务删除所有tesnor，若是BWD任务调用的该函数，返回随机生成的输入tensor字典、输出tensor字典
                        # 对包含计算loss层的vt，区别在于：
                        # 1.还需生成label tensor
                        # 2.多了一步计算loss
                        # 3.无论FWD还是BWD，最后都返回输入tensor字典、输出tensor字典
                        # 4.输出的tensor字典是输出的loss tensor
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=False, verbose=verbose, nvprof=nvprof)
                        gc.collect()
                        if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                    ### Delete model {W,B}
                    self.default_stream.synchronize() # CPU wait Compute
                    if nvprof: nvtx_range_push("task{}({}) Del(W,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers:
                        # 若当前任务不需要输出W和B，
                        if not (l in vt.Out['W']) and not (l in vt.Out['B']):
                            # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
                            self.local_model[l].del_param_grad_buf()
                        elif vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN':
                            pass
                        else: # P2P
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                # -----------------------------------------------
            elif vt.type == 'BWD' and vt.is_gpu:
                # -----------------------------------------------
                ### Swap-in model {W,B}
                if nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                cur_vt_idx = self.prefetch_model_handler.get(vt, None)
                assert cur_vt_idx == vt.idx 
                if verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                if nvprof: nvtx_range_pop() 
                ### Run through each microbatch in a data batch. 
                for i, u in enumerate(vt.ubatchszs):
                    ### Recompute to create pytorch graph
                    # requires_grad=True，即要保存 X_named_tensors, Y_named_tensors
                    X_named_tensors, Y_named_tensors = \
                        self._initial_a_pack_forward_an_ubatch(vt, i, u, requires_grad=True, verbose=verbose, nvprof=nvprof) 
                    ### Backward pass on recomputed graph
                    # 1.为backward函数的第二个参数准备一个值全为1的tensor，若是第一个BWD任务，即对loss求偏导，则无需准该tensor
                    # 2.执行反向传播
                    # 3.删除所有用到的tensor
                    self._initial_a_pack_backward_an_ubatch(vt, i, u, X_named_tensors, Y_named_tensors, verbose=verbose, nvprof=nvprof)
                    ### Clean up
                    del X_named_tensors; del Y_named_tensors # very important!
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                ### Swap-out model {W,dW,B}
                if self.CONFIGS["opt_offld"]:                
                    ### Delete model {W,dW,B}
                    self.default_stream.synchronize() # CPU wait for SwapOut
                    if nvprof: nvtx_range_push("task{}({}) Del(W,dW,B)".format(vt.idx, vt.show_layers())) 
                    for l in vt.layers: 
                        # 若dW的输出媒介是LOC，B的输出媒介是SHM
                        if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                            # 递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
                            self.local_model[l].del_param_grad_buf()
                        else: # 'B' == PIN
                            raise ValueError("Underdevelopment")
                    gc.collect()
                    if verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                    if nvprof: nvtx_range_pop() 
                else:
                    raise ValueError("GPU Optimizer Underdevelopment.")
                # -----------------------------------------------
            elif vt.type == 'UPD' and not vt.is_gpu:
                # -----------------------------------------------
                pass
                # -----------------------------------------------
            else:
                raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
        # tasks ends
        torch.cuda.synchronize(self.rank); dist.barrier()
        time_end = pc() 
        if nvprof:
            nvtx_mark("cudaProfilerStop") 
            cuda_profiler.stop()
            probe_cuda_mem.stop()
            print("rank%d: cuda profiler stops" % self.rank) 
        print("rank%d: initial iteration ends. time %.3f s"%(self.rank, time_end-time_start))
        # clean memory
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        dist.barrier()
        
        if self.args.initial_iter_only:
            print("rank%d: --- Initial Iteration Only ---" % self.rank)
            exit(0) 

    ################### Regular Training Loop ###################
    #
    def _a_pack_forward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                data_ubatches, target_ubatches, 
                                requires_grad=False, 
                                prefetch_model_handler=None,
                                swapin_stashx_handler=None,
                                swapin_localx_handler=None,
                                swapin_msgx_handler=None,
                                swapout_stashx_handler=None,
                                swapout_localx_handler=None,
                                swapout_msgx_handler=None,
                                sucinfo=None):
        """ requires_grad == False: FWD (non-criterion)
            requires_grad == True: Recompute (for all) """
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs)-1

        # 大部分FWD、BWD都执行这个, 只有第一个BWD vt包含计算loss层不会执行这个
        # 📌分析：对BWD的冲计算任务（即BWD过程中的FWD任务），会通过medium=='MSG'拿取暂存的stashX，而非
        # 仅仅针对FWD任务。比较绕的点就在于BWD拿取stashX的任务是在FWD（重计算）的逻辑中执行的。
        if not vt.has_criterion: # not last pack yet
            ### In {X}
            # 1.接受整个vt的输入X，分几种情况：
            #   --整个模型的第一层，输入的就是输入数据microbatch，直接将该microbatch复制到GPU上
            #   --中间的vt，输入肯定是前面的vt输出的结果，通过P2P通信广播过来
            #   --...
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            # 当前vt的输入名称
            X_names = self.local_model[l].X_names
            if m.medium == "DAT": # Get one microbatch data
                # Data as X
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                # 将字典中的tensor替换为GPU版本，即移动到GPU上
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                # 该参数默认为false，执行else
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    # ❓？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            
            # 📌BWD的重计算任务会在此处拿取stashX
            elif m.medium == "MSG": # message pass stashed input
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}StashX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                # 该参数默认为false，执行else
                if self.args.no_prefetch_stashx:
                    X_named_tensors = swapin_stashx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:

                    X_named_tensors = swapin_stashx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) SwapIn(#{}StashX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 

            # print(f"rank:{self.rank}, vt:{vt.idx}({vt.show_layers()}), 接收的tensor为:{X_named_tensors} ({ubatch_idx})")

            ### Prefetch point @ FWD/Recompute (non-criterion)'s ULast 
            # 2.若当前执行到最后一个micro batch，为当前rank上下一个vt预接收其需要的输入数据
            # 2.1.若当前任务是FWD任务（not requires_grad），而非BWD时执行的重计算，则为后继vt预取 X/dY
            # 2.2.若该rank上的下一个任务是BWD任务，就把stashX拿进来。具体来说，在sucinfo.stashx()函数中，若后继任务是FWD任务，
            #     直接返回None。prefetch_suc函数在收到None后也会直接返回，不做任何处理
            if is_last_ubatch:
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 

                # 2.1
                # 若当前任务是FWD任务（not requires_grad），而非BWD时执行的重计算，则为后继vt预取 X/dY
                # 分析：若是重计算任务，则不会预取输入X，原因很简单，重计算任务要拿的是暂存的stashX，不存在前面的任务给你发X，拿stashX
                #      的代码就在拿X的下面
                #      对于BWD任务的后继BWD任务需要的dY的问题，这个逻辑肯定是在 _a_pack_backward_an_ubatch 函数中执行，
                #      这里由于 not requires_grad 才会执行，显然直接不需要考虑这种情况了
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv and not requires_grad:
                    # sucinfo.p2pin()：若后继任务接收X/dY的媒介为P2P，返回其元数据和src_rank，用于prerecv_suc方法
                    #                  提前使用P2P方法接收后继FWD或BWD任务的输入X/dY
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())
                # if swapin_msgx_handler is not None and \
                #     not self.args.no_prefetch_msgx and not requires_grad:
                #     swapin_msgx_handler.prefetch_suc(sucinfo.msgx())
                # if prefetch_model_handler is not None and \
                #     not self.args.no_prefetch_model and not requires_grad:
                #     prefetch_model_handler.iput(sucinfo.model()) 

                # 2.2
                # 若该rank上的下一个任务是BWD任务，就把stashX拿进来。具体来说，在sucinfo.stashx()函数中，若后继任务是FWD任务，
                # 直接返回None。prefetch_suc函数在收到None后也会直接返回，不做任何处理
                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    # sucinfo.stashx()：
                    # 若后继任务是BWD（非第一个BWD），且输入媒介是MSG，返回 (l(后继任务的首层id), 后继任务输入X的元数据) 。非MSG直接返回None
                    # 其他情况直接返回None
                    # 📌分析：只有后继任务为BWD任务，stashx()才有返回值，也就是说若后继任务为FWD，这里根本不会预取
                    #
                    # 在gpu上按照sucinfo.stashx()返回的元数据在gpu上生成一个空tensor，随后将
                    # (suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
                    # 📌这意味着SwapIn线程会开始cpu到gpu的拷贝操作
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                # if swapin_stashx_handler is not None and \
                #     not self.args.no_prefetch_stashx and requires_grad:
                #     swapin_stashx_handler.prefetch_suc(sucinfo.stashx())

                # swapin_localx_handler 为 vDP   专用，暂略
                if swapin_localx_handler is not None and \
                    not self.args.no_prefetch_localx and not requires_grad:
                    # sucinfo.localx()：
                    # 为后继任务准备输入信息(元数据)，后继为FWD/首个BWD则准备输入X，后继为BWD则准备输入dY
                    # 两种情况
                    # 1.FWD->FWD、最后一个FWD->第一个BWD(包含计算层)
                    #   若suc_vt首层接收X的媒介不为SWP，返回None，否则返回 (l(suc_vt的首层)，l这层接收X的元数据)
                    # 2.首个BWD->BWD、BWD(非首个)->BWD(非首个)
                    #   若suc_vt最后一层接收dY的媒介不为SWP，返回None，
                    #   否则返回 (l+1(suc_vt的最后一层+1即为当前vt的首层)，l+1这层接收X的元数据)
                    swapin_localx_handler.prefetch_suc(sucinfo.localx())
                if self.nvprof: nvtx_range_pop() 

            ### Compute forward pass on GPU
            # 3.若当前是BWD任务调用的FWD任务，即重计算
            #   启用tensor的梯度计算，并设置为保留梯度。若输入的tensor是复用之前的tensor，还需进行detach_()以及梯度清零操作
            if requires_grad:
                # 启用tensor的梯度计算，并设置为保留梯度。若输入的tensor是复用之前的tensor，还需进行detach_()以及梯度清零操作
                turn_on_X_grad(X_named_tensors) 
            if self.nvprof: nvtx_range_push("task{}({}) {}(#{})".format(vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx)) 
            # 当前vt的输入tensor
            Y_tensors = [X_named_tensors[name] for name in X_names]

            # 4.Swap X: 对vt中的每一层跑一次前向，对vt中的首层还需进行额外的处理，即swap掉其输入X，并传给目标rank
            for l in vt.layers:
                # 若当前层l，其输入X需要swap out掉，且输出X的媒介为MSG，则调用SwapOut实例的线程，将tensor卸载到cpu的pinned memory上
                # 📌该swapout线程还会调用 MSGstashX 的方法，触发 MSGstashX 发送线程的运行，即将swapout的tensor 发送到目标rank上
                if not requires_grad and l in vt.Out['X']: ### Out {stashX}
                    if self.nvprof: nvtx_range_push("task{}(L{}) SwapOut(#{}StashX)".format(vt.idx, l, ubatch_idx)) 
                    if vt.Out['X'][l].medium == "MSG": # message pass stashed X
                        # 1.记录一个默认计算流上的事件 ev_compute
                        # 2.在swapout stream上等待 ev_compute 事件完成，即确保计算完成后才能swapout
                        # 3.在swapout_stream上:
                        #   3.1.在cpu上的pinned memory上创建一个空tensor
                        #   3.2.异步的将gpu上的tensor拷贝到刚刚分配的空tensor上
                        #   返回cpu_named_tensors
                        # 4.将 (layer_id, cpu_named_tensors, ev_swapout, flag) 添加到 put_queue 队列中。📌这意味着当前实例的线程会将已经卸载
                        #   到cpu上的tensor放到 MSGstashX 实例的 send_dict 中。
                        #   📌这也意味着 MSGstashX 的发送线程将向 dst_rank 发送此 tensor
                        swapout_stashx_handler.offload(l, 
                                make_tensors_named(self.local_model[l].X_names, Y_tensors))
                        # print("\trank{}: task{}(L{}) SwapOut(#{}StashX)".format(self.rank, vt.idx, l, ubatch_idx))
                    else:
                        raise NotImplementedError
                    if self.nvprof: nvtx_range_pop() 
                # print("\trank{}: task{}(L{}) {}".format(self.rank, vt.idx, l, "FWD" if not requires_grad else "Recompute"))
                # total_params = 0
                # total_memory_bytes = 0
                # for name, param in self.local_model[l].model.named_parameters():
                #     num_params = param.numel()
                #     dtype = param.dtype
                #     memory_bytes = num_params * param.element_size()
                    
                #     total_params += num_params
                #     total_memory_bytes += memory_bytes
                    
                #     # print(f"参数名称: {name}, 参数量: {num_params}, 数据类型: {dtype}, 空间占用: {memory_bytes / (1024 ** 2):.6f} MB")
                # print(f"rank:{self.rank}, layer{l}, 总参数量: {total_params}, 总空间占用: {total_memory_bytes / (1024 ** 2):.6f} MB\n")
                Y_tensors = self.local_model[l](*Y_tensors)
                if not isinstance(Y_tensors, tuple):
                    Y_tensors = (Y_tensors,)
                Y_tensors = list(Y_tensors)
            if self.verbose: print("\trank{}: task{}({}) {} (#{})".format(self.rank, vt.idx, vt.show_layers(), "FWD" if not requires_grad else "Recompute", ubatch_idx))
            if self.nvprof: nvtx_range_pop() 
            ### Save Y
            l = vt.layers[-1]
            Y_names = self.local_model[l].Y_names
            Y_named_tensors = make_tensors_named(Y_names, Y_tensors)

            # 5.若不需要梯度，即当前不是BWD中的重计算过程，需要异步的将vt的输出发送到目标rank上
            if not requires_grad:

                num_elements = Y_tensors[0].numel()
                element_size = Y_tensors[0].element_size()
                memory_usage = num_elements * element_size / (1024 ** 2)
                print(f"rank:{self.rank}, 激活的参数量:{num_elements}, 激活的数据类型:{Y_tensors[0].dtype}, 激活的内存占用: {memory_usage} mb")

                # print(f"rank:{self.rank}, vt:{vt.idx}({vt.show_layers()}), FWD完成, 发送的tensor:{Y_named_tensors} ({ubatch_idx})")

                ### Out {Y}
                m = vt.Out['Y'][l]
                # 非阻塞的将 tensor 发送到目标rank的GPU上，返回一个异步work句柄
                if m.medium == "P2P":
                    if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    self.p2px_handler.isend(Y_named_tensors, dst=m.rank)
                    # print("\trank{}: task{}({}) P2POut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "MSG": # last FWD convert to first BWD
                    if self.nvprof: nvtx_range_push("task{}({}) MSGOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    swapout_msgx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: task{}({}) MSGOut(#{}Y)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
                elif m.medium == "SWP": # swap locally for vDP
                    if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}Y)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                    if self.is_convert_ubs:
                        flag_is_convert = True if vt.is_last_fwd else False
                        swapout_localx_handler.offload(l+1, Y_named_tensors, flag_is_convert)
                    else:
                        swapout_localx_handler.offload(l+1, Y_named_tensors)
                    # print("\trank{}: swp_send'ed L{}-Y".format(self.rank, l))
                else:
                    raise NotImplementedError
                if self.nvprof: nvtx_range_pop() 
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del Y_tensors
            # 对FWD任务，无需保存输入和输出
            if not requires_grad:
                del X_named_tensors; del Y_named_tensors
            else: # for backward pass
                return X_named_tensors, Y_named_tensors

        # 对于包含计算loss层的, 此处实际上是在BWD的执行过程中执行,即首个BWD任务才是实际上的最后一个
        # FWD任务, 要先执行FWD产生一个loss
        else: # criterion pack
            assert requires_grad # fused forward and backward
            ### In {X}
            l, m = vt.layers[0], vt.In['X'][vt.layers[0]]
            X_names = self.local_model[l].X_names

            # 1.接受整个vt的输入X，分几种情况：
            #   --整个模型的第一层，输入的就是输入数据microbatch，直接将该microbatch复制到GPU上
            #   --中间的vt，输入肯定是前面的vt输出的结果，通过P2P通信广播过来
            #   --...
            if m.medium == "DAT": # a single BWD task
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}Data)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                X_named_tensors = swp_x.swapin(data_ubatches[ubatch_idx])
                # print("\trank{}: task{}({}) SwapIn(#{}Data)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "P2P": # the same above
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    X_named_tensors = self.p2px_handler.recv(self.XMETA.get(ubatch_size,l), src=m.rank)
                else:
                    X_named_tensors = self.p2px_handler.prerecv(self.XMETA.get(ubatch_size,l), src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "MSG": # last FWD convert to first BWD
                if self.nvprof: nvtx_range_push("task{}({}) MSGIn(#{}X)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_msgx:
                    X_named_tensors = swapin_msgx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_msgx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: task{}({}) MSGIn(#{}X)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}LocalX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    X_named_tensors = swapin_localx_handler.fetch(l, self.XMETA.get(ubatch_size,l))
                else:
                    X_named_tensors = swapin_localx_handler.prefetch(l, self.XMETA.get(ubatch_size,l), is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-X".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop() 

            # print(f"rank:{self.rank}, vt:{vt.idx}({vt.show_layers()}), 在执行loss vt之前, 接收到的输入为{X_named_tensors} ({ubatch_idx})")

            ### Prefetch point @ Recompute(criterion) ULast
            # 2.若当前执行到最后一个micro batch，为当前rank上下一个vt预取其需要的输入数据
            # 2.1.为后继vt预取 dY
            # 2.2.若该rank上的下一个任务是BWD任务，就把stashX拿进来。具体来说，在sucinfo.stashx()函数中，若后继任务是FWD任务，
            #     直接返回None。prefetch_suc函数在收到None后也会直接返回，不做任何处理
            if is_last_ubatch:
                if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                
                # 2.1.为后继vt预取 dY
                if self.p2px_handler is not None and \
                    not self.args.no_p2p_prerecv:
                    self.p2px_handler.prerecv_suc(sucinfo.p2pin())

                # 若该rank上的下一个任务是BWD任务，这个就会执行，把stashX拿进来
                if swapin_stashx_handler is not None and \
                    not self.args.no_prefetch_stashx:
                    # sucinfo.stashx()：
                    # 若后继任务是BWD（非第一个BWD），且输入媒介是MSG，返回 (l(后继任务的首层id), 后级任务输入X的元数据) 。非MSG直接返回None
                    # 其他情况直接返回None
                    #
                    # 在gpu上按照sucinfo.stashx()返回的元数据在gpu上生成一个空tensor，随后将
                    # (suc_layer_id, cuda_named_tensors, ev_compute)添加到 put_queue 中，
                    # 📌这意味着SwapIn线程会开始cpu到gpu的拷贝操作
                    swapin_stashx_handler.prefetch_suc(sucinfo.stashx())
                if self.nvprof: nvtx_range_pop() 

            ### Recompute on GPU
            # 3.启用tensor的梯度计算，并设置为保留梯度。若输入的tensor是复用之前的tensor，还需进行detach_()以及梯度清零操作
            turn_on_X_grad(X_named_tensors)
            if self.nvprof: nvtx_range_push("task{}({}) Recompute(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
           
            # 4.执行前向计算并保留输出tensor
            if len(vt.layers) > 1: # packed
                Y_tensors = [X_named_tensors[name] for name in X_names]
                for l in vt.layers[:-1]:
                    Y_tensors = self.local_model[l](*Y_tensors)
                    if not isinstance(Y_tensors, tuple):
                        Y_tensors = (Y_tensors,)
                    Y_tensors = list(Y_tensors)
                Y_names = self.local_model[vt.layers[-2]].Y_names
                Y_named_tensors = make_tensors_named(Y_names, Y_tensors)
            else: # only last vlayer
                Y_names = X_names
                Y_named_tensors = X_named_tensors

            ### In {T}
            # 5.将target tensor拿到GPU上
            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}T)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            T_named_tensors = swp_x.swapin(target_ubatches[ubatch_idx])
            if self.nvprof: nvtx_range_pop() 
            
            ### Compute loss on GPU
            # 6.计算loss
            assert vt.layers[-1] == self.CONFIGS['R']-1
            last_vlayer = self.local_model[self.CONFIGS['R']-1]        
            if self.compute_loss is not None: # "bert_thomwolf", "gpt2_2bw", "gpt2_huggingface"
                Y_tensors = self.compute_loss(last_vlayer, Y_named_tensors, Y_names, T_named_tensors)
            else:
                Y_tensors = [last_vlayer(Y_named_tensors[name],T_named_tensors["target"]) for name in Y_names]
                Y_tensors = [sum(Y_tensors)]
            if self.verbose: print("\trank{}: task{}({}) Recompute(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            if self.nvprof: nvtx_range_pop() 
            
            ### Save Y
            # 7.保存loss
            Y_named_tensors = make_tensors_named(['loss'], Y_tensors)
            ### Clean up
            if self.nvprof: nvtx_range_push("task{}({}) FWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
            # print("\trank{}: task{}({}) FWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            del T_named_tensors; del Y_tensors; 
            return X_named_tensors, Y_named_tensors

    def _a_pack_backward_an_ubatch(self, vt, ubatch_idx, ubatch_size,
                                X_named_tensors, Y_named_tensors,
                                swapin_localx_handler=None,
                                swapout_localx_handler=None,
                                sucinfo=None,
                                iteration_num=None):
        is_last_ubatch = ubatch_idx == len(vt.ubatchszs) - 1
        ### In {dY}
        # 1.接受后一个BWD任务发来的dY
        # 1.1.对首个BWD任务,显然不需要
        if vt.has_criterion:
            dY_named_tensors = ODict({ 'loss': None })
            assert Y_named_tensors['loss'].requires_grad

        # 1.2.非第一个BWD任务，须接收前一个BWD任务发来的dY
        else:
            l, m = vt.layers[-1], vt.In['dY'][vt.layers[-1]]
            # 返回当前vt输出X的元数据
            dY_named_metas = make_dY_named_metas(self.XMETA, ubatch_size, l)
            # 同步的接收发来的 dY（异步接收，但会阻塞至接收完毕），非第一个BWD任务，接收dY的步骤在上一个vt已经执行
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2PIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_p2p_prerecv:
                    dY_named_tensors = self.p2px_handler.recv(dY_named_metas, src=m.rank)
                else:
                    now = datetime.datetime.now()
                    print(f"rank:{self.rank}, vt{vt.idx}({vt.show_layers()})({ubatch_idx})接收时间:{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    dY_named_tensors = self.p2px_handler.prerecv(dY_named_metas, src=m.rank, is_end=is_last_ubatch) 
                # print("\trank{}: task{}({}) P2PIn(#{}dY)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapIn(#{}dY)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.args.no_prefetch_localx:
                    dY_named_tensors = swapin_localx_handler.fetch(l+1, dY_named_metas)
                else:
                    dY_named_tensors = swapin_localx_handler.prefetch(l+1, dY_named_metas, is_last_ubatch)
                # print("\trank{}: swp_recv'ed L{}-dY".format(self.rank, l))
            else:
                raise NotImplementedError
            if self.nvprof: nvtx_range_pop()         

        # print(f"rank:{self.rank}, vt:{vt.idx}({vt.show_layers()})({vt.type}), 接受的dY为:{dY_named_tensors} (it:{iteration_num})({ubatch_idx})")
       
        ### Prefetch point @ BWD's ULast
        # 若当前是最后一个micro batch，且当前不是第一个BWD任务，则为后继BWD任务 预接收dY。仅仅是开始接收，要发送的dX还没算出来呢
        if is_last_ubatch:
            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
            # 若当前不是第一个BWD任务，
            if self.p2px_handler is not None and \
                not self.args.no_p2p_prerecv and not vt.has_criterion:
                # sucinfo.p2pin()：若后继BWD任务接收dY的媒介为P2P，返回当前任务输入X的元数据作为后继BWD任务输入dY的元数据，用于prerecv_suc方法
                # 返回 ( { name:TensorMeta }, 来源rank )
                # 提前使用P2P方法接收后继BWD任务的输入dY
                self.p2px_handler.prerecv_suc(sucinfo.p2pin())
            if swapin_localx_handler is not None and \
                not self.args.no_prefetch_localx:
                swapin_localx_handler.prefetch_suc(sucinfo.localx())
            if self.nvprof: nvtx_range_pop() 

        ### Compute backward pass
        # 3.执行反向传播，计算梯度
        if self.nvprof: nvtx_range_push("task{}({}) BWD(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        Y_tensors = []
        Y_gradients = [] 
        for name in self.local_model[vt.layers[-1]].Y_names: # only tensor & required_grad can run autograd
            Y = Y_named_tensors[name]
            # print(f"rank:{self.rank}, Y.grad:{Y.grad}")
            if isinstance(Y,(torch.Tensor, Variable)) and (Y.requires_grad):
                Y_tensors.append(Y)
                Y_gradients.append(dY_named_tensors[name])
            elif isinstance(Y, list): # output tuple of bert pretrainheader
                for i, y in enumerate(Y):
                    if isinstance(y,(torch.Tensor, Variable)) and (y.requires_grad):
                        Y_tensors.append(y)
                        Y_gradients.append(dY_named_tensors[name][i])
        #
        # print(f"rank:{self.rank}, 反向前:Y.grad:{Y_named_tensors[self.local_model[vt.layers[-1]].Y_names[0]].grad}")
        torch.autograd.backward(tuple(Y_tensors), grad_tensors=tuple(Y_gradients))
        # print(f"rank:{self.rank}, 反向后:Y.grad:{Y_named_tensors[self.local_model[vt.layers[-1]].Y_names[0]].grad}")
        if self.verbose: print("\trank{}: task{}({}) BWD(#{},{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx, ubatch_size))
        if self.nvprof: nvtx_range_pop() 

        ### Out {dX}
        # 4.若当前任务需要输出dX（显然最后一个BWD任务不需要），取出输入X的grad tensor(dX)，使用异步通信将dX点对点的发送到目标rank上
        if vt.Out['dX']:
            ### Save dX
            # 取出给定tensor的grad tensor，装在named_tensor字典中返回，使用异步通信将dX点对点的发送到目标rank上
            dX_named_tensors = make_dX_from_X(X_named_tensors) # ref to .grad

            # for name,tensor in dX_named_tensors.items():
            #     num_elements = tensor.numel()
            #     element_size = tensor.element_size()
            #     memory_usage = num_elements * element_size / (1024 ** 2)
            #     print(f"rank:{self.rank}, BWD激活的参数量:{num_elements}, 激活的数据类型:{tensor.dtype}, 激活的内存占用: {memory_usage} mb")

            l, m = vt.layers[0], vt.Out['dX'][vt.layers[0]] 
            if m.medium == "P2P":
                if self.nvprof: nvtx_range_push("task{}({}) P2POut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                now = datetime.datetime.now()
                print(f"rank:{self.rank}, vt{vt.idx}({vt.show_layers()})({ubatch_idx})发送时间:{now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                self.p2px_handler.isend(dX_named_tensors, dst=m.rank)
                # print("\trank{}: task{}({}) P2POut(#{}dX)".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
            elif m.medium == "SWP": # swap locally for vDP
                if self.nvprof: nvtx_range_push("task{}({}) SwapOut(#{}dX)".format(vt.idx, vt.show_layers(), ubatch_idx)) 
                if self.is_convert_ubs:
                    swapout_localx_handler.offload(l, dX_named_tensors, False)
                else:
                    swapout_localx_handler.offload(l, dX_named_tensors)
                # print("\trank{}: swp_send'ed L{}-dX".format(self.rank,l))
            else:
                raise NotImplementedError
            del dX_named_tensors; 
            if self.nvprof: nvtx_range_pop() 
        ### Clean up {X,Y,dX,dY}
        if self.nvprof: nvtx_range_push("task{}({}) BWDClean(#{})".format(vt.idx, vt.show_layers(), ubatch_idx)) 
        # print("\trank{}: task{}({}) BWDClean(#{})".format(self.rank, vt.idx, vt.show_layers(), ubatch_idx))
        del X_named_tensors; del Y_named_tensors
        del dY_named_tensors; del Y_tensors; del Y_gradients

    # 
    def run_training_loop(self):
        local_losses = [] # per-minibatch
        global_losses = [] # per-minibatch
        if self.args.no_update: 
            grad_sums = [] # per-minibatch
        self.update_cnt = 0
        self.time_iters = []

        self.delete_time = []
        self.get_time = []
        self.swapout_time = []
        self.gc_time = []
        self.compute_time = []
        self.update_time = []
        self.offload_time = []

        # num_iters：一个epoch迭代（iteration）的次数，手动设置或dataloader minibatch的数量
        # num_epochs：epoch的数量，手动设置
        # 即整个迭代次数的一半
        self.avg_it = int(self.args.num_iters * self.args.num_epochs /2.) # from this iter to average time
        
        ### clean memory before start
        # torch.cuda.synchronize(self.rank)：等待当前GPU上所有的核函数执行完毕
        torch.cuda.synchronize(self.rank); dist.barrier()
        gc.collect(); torch.cuda.empty_cache() 
        torch.cuda.synchronize(self.rank)
        assert torch.cuda.memory_reserved(self.rank)==0
        self.cgm = CheckGPUMem(self.rank)
        dist.barrier()
        print("rank%d: --- training starts ---" % self.rank)
        
        ### start
        self.rand_state_train.set()
        for epoch in range(self.args.num_epochs): # traverse epoches
            for it, minibatch in enumerate(self.data_loader): # traverse each minibatch
                if it >= self.args.num_iters:
                    break
                ### clean start
                gc.collect() 
                # 该参数默认为 false
                if self.args.empty_cache: torch.cuda.empty_cache() 
                torch.cuda.synchronize(self.rank)
                assert torch.cuda.memory_allocated(self.rank)==0, "iteration with rank = {} begins w/ alloc = {} B".format(self.rank, torch.cuda.memory_allocated(self.rank)) 
                dist.barrier()
                if self.nvprof and it == self.args.nvprof_iter["start"]:
                    probe_cuda_mem = ProbeCudaMem(self.rank)
                    probe_cuda_mem.start()  
                    cuda_profiler.start()
                    nvtx_mark("cudaProfilerStart") 
                    print("rank%d: cuda profiler starts"%self.rank)
                else:
                    # 重置CUDA设备上的内存峰值统计信息
                    torch.cuda.reset_peak_memory_stats(self.rank) 
                time_start = pc() 
                ### data minibatch
                # 该参数默认为false，执行else
                if self.args.synthetic_data:
                    data_ubatches, target_ubatches = self.data_ubatches, self.target_ubatches
                else:
                    # 在gpt2中，这个参数为true，📌现在minibatch是一个有两个元素的元组，第二个元素是自己的复制
                    #
                    # 1.复制一份minibatch，将两个minibatch放进元组里，第一个minibatch是前向用的minibatch，第二个
                    #   minibatch是后向用的minibatch，两者将按照前后向不同的microbatchsize进行切分，切分成microbatch
                    if self.is_copy_minibatch: # "gpt2_huggingface"
                        minibatch = (minibatch, deepcopy(minibatch))

                    # 2.忽略sample数量不够和sample维度不对的minibatch，直接跳过此次循环
                    # 若当前minibatch的第0维数量和定义的Minibatchsize大小不同，返回True，即sample的数量不对。最后一个minibatch可能数量不够
                    # 若minibatch第1维数量和defined_seq_len不同，返回True
                    if self.is_skip_minibatch(minibatch, self.CONFIGS['D'], self.fdim, verbose=self.verbose): # skip fractional minibatch
                        assert (not self.nvprof) or (self.nvprof and it != self.args.nvprof_iter["end"]), "Unstoped Profiling"
                        continue
                    # preprocess_minibatch 函数实际上不做任何事情
                    minibatch = self.preprocess_minibatch(minibatch) # preprocess as if single GPU

                    # self.bnames：{"is_data" = [True, False]， "name" = ["input0", "labels"]}
                    # self.ubatchszs_fwd_local：fwd minibatchsize的列表
                    # self.ubatchszs_bwd_local：bwd minibatchsize的列表
                    #
                    # 3.将传进来的两个相同的minibatch分别按照前向microbatch size列表和后向Microbatch列表进行拆分，拆分成micorbatch。
                    #   返回前向和后向的microbatch tensor列表，每个元素都是一个named_tensor字典
                    data_ubatches, target_ubatches = decompose_minibatch(minibatch, self.bnames, self.ubatchszs_fwd_local, self.ubatchszs_bwd_local, self.XMETA, self.TMETA, self.CONFIGS, self.rank, pin_memory=not self.args.no_pin_data) # make microbatches
                    # print(f"data_ubatches:{data_ubatches}")
                ### task starts    
                # 对当前rank的每一个vt
                if self.nvprof: nvtx_range_push("rank{}, iteration:{}".format(self.rank,it))
                delete_time = []
                get_time = []
                swapout_time = []
                gc_time = []
                compute_time = []
                update_time = []
                offload_time = []
                for j, vt in enumerate(self.rTASKS[self.rank]): # { rank0: [task0,task2,...] }
                    if self.verbose: print("\trank{}: executing {}".format(self.rank, vt))
                    # 设置 SucInfoForPrefetch 实例的 self.vt = vt；self.rank_vt_idx = rank_vt_idx
                    self.sucinfo.set(vt, j)
                    if self.nvprof: nvtx_range_push("task{}({})({})".format(vt.idx, vt.show_layers(), vt.type)) 
                    if vt.type == 'FWD' and vt.is_gpu:
                        # -----------------------------------------------      
                        with torch.no_grad():
                            ### Swap-in model {W,B}
                            if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                            # no_prefetch_model：该参数默认为false，执行else
                            # 返回后续任务（FWD或BWD任务），即当前rank上第j+1或j+2个任务
                            suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()

                            # 1.将vt中所有layer的W和B复制GPU(的layer)上（GPU也有一份所有的layer，但一开始都是空的）
                            # 1.1.准备工作1：调用syncpin_handler实例的线程将vt中的这些在cpu共享内存中的layer复制到pinned memory上；
                            # 1.2.准备工作2：在默认计算流上，按照Pinned memory中layer的W,B的大小、类型，为给定vt在GPU(的对应层)上的所有layer初始化W和B（值是随机的）
                            #   同时也是当前PrefetchLocalModelGPU实例的线程的触发工作，将东西放进put_queue，这意味着线程开始执行3
                            # 1.3.在 swapin_stream 中将W和B从cpu memory（默认在固定内存上）上拷贝到gpu的model上，若vt的类型为BWD，还需要
                            #   显示的设置参数 param 的 requires_grad 属性为 True
                            # 1.4.调用_wait将is_running 置为false，返回get_queue 的首个 (vt.idx，ev_swapin)
                            # 1.5.self.compute_stream.wait_event(ev_swapin)
                            # 1.6.若suc_vt参数不为空，意味着该函数会为提前执行一部分后继任务，即调用self.syncpin_handler.iput(suc_vt)，与1.1相同
                            start_time = time.time()
                            cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, 取模型执行了:{execution_time:.6f}秒")
                            get_time.append(execution_time)
                            assert cur_vt_idx == vt.idx
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                            if self.nvprof: nvtx_range_pop() 
                            ### Run through each microbatch in a data batch
                            # 2.有多少个microbatch就跑多少次前向
                            #   从函数的运行结果来看, 就是不断的向目标rank发送计算出来的Y
                            for i, u in enumerate(vt.ubatchszs):
                                start_time = time.time()
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                        data_ubatches, target_ubatches, 
                                        requires_grad=False, 
                                        prefetch_model_handler=self.prefetch_model_handler,
                                        swapin_stashx_handler=self.swapin_stashx_handler,# 新建一个swapin_stream，专门用来 swap in stashX
                                        swapin_localx_handler=self.swapin_localx_handler,
                                        swapin_msgx_handler=self.swapin_msgx_handler,
                                        swapout_stashx_handler=self.swapout_stashx_handler,
                                        swapout_localx_handler=self.swapout_localx_handler,
                                        swapout_msgx_handler=self.swapout_msgx_handler,
                                        sucinfo=self.sucinfo)
                                end_time = time.time()
                                execution_time = end_time - start_time
                                print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, microbatch{i}, 前向执行了:{execution_time:.6f}秒")
                                compute_time.append(execution_time)
                                if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                                if self.nvprof: nvtx_range_pop() 
                                start_time = time.time()
                                gc.collect()
                                end_time = time.time()
                                execution_time = end_time - start_time
                                print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, microbatch{i}, gc.collect执行了:{execution_time:.6f}秒")
                                gc_time.append(execution_time)
                            ### Prefetch point @ FWD Del
                            # 3.阻塞，直到该流中的任务全部完成
                            self.default_stream.synchronize() # CPU wait Compute
                            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                            # 该参数默认为false，执行
                            # 4.预取suc_vt的模型，将其W和B拿到GPU上
                            if not self.args.no_prefetch_model:
                                # input：即把vt的所有layer的W和B input 到GPU上
                                # 分析：没有指定流，整个过程应是在默认流上执行
                                # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
                                # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
                                # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
                                # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
                                self.prefetch_model_handler.iput(suc_vt)

                            # 在P2P情况下，这个应该不执行
                            # 📌24/10/10：实际上就是给vPP准备的。最后一个FWDvt→第一个BWDvt在前后向microbatch大小不一致时，通信媒介被自动设置为MSG
                            #   此时这里就会执行
                            if self.swapin_msgx_handler is not None and not self.args.no_prefetch_msgx:
                                # self.sucinfo.msgx()：
                                # 若当前vt和后继vt的情况为：FWD -> 首个BWD，且suc_vt的输入X的媒介为MSG，返回suc_vt首层的层号、输入X的元数据
                                #
                                # 
                                self.swapin_msgx_handler.prefetch_suc(self.sucinfo.msgx())
                            # if self.swapin_stashx_handler is not None and not self.args.no_prefetch_stashx:
                            #     self.swapin_stashx_handler.prefetch_suc(self.sucinfo.stashx())
                            if self.nvprof: nvtx_range_pop() 
                            ### Delete model {W,B}
                            if self.nvprof: nvtx_range_push("task{}({}) Del(W,B)".format(vt.idx, vt.show_layers())) 
                            start_time = time.time()
                            for l in vt.layers:
                                # 若不需要输出W和B，递归地删除模块（包括子模块）中的所有参数、梯度和缓冲区
                                # 大部分前向任务都不需要输出W和B
                                if not (l in vt.Out['W']) and not (l in vt.Out['B']):
                                    self.local_model[l].del_param_grad_buf()
                                # 若W和B的输出媒介为PIN，即BWD任务也在当前rank上，W和B需要保存在GPU上，避免再次拿取
                                elif vt.Out['W'][l].medium=='PIN' and vt.Out['B'][l].medium=='PIN':
                                    pass
                                else: # P2P
                                    raise NotImplementedError
                            end_time = time.time()
                            execution_time = end_time - start_time
                            delete_time.append(execution_time)
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, 删除执行了:{execution_time:.6f}秒")
                            start_time = time.time()
                            gc.collect()
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, gc.collect执行了:{execution_time:.6f}秒")
                            gc_time.append(execution_time)
                            if self.verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                            if self.nvprof: nvtx_range_pop() 
                        # -----------------------------------------------
                    elif vt.type == 'BWD' and vt.is_gpu:
                        # -----------------------------------------------
                        ### Swap-in model {W,B}
                        if self.nvprof: nvtx_range_push("task{}({}) SwapIn(W,B)".format(vt.idx, vt.show_layers())) 
                        if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) start")
                        # 该参数为false，执行else
                        # 返回后续任务（FWD或BWD任务），即当前rank上第j+1或j+2个任务
                        suc_vt = None if self.args.no_prefetch_model else self.sucinfo.model()

                        # 1.拿取模型W和B至GPU
                        # 等待当前vt的W和B的预取事件的完成，同时开始suc_vt的部分任务，即共享内存到固定内存的模型复制
                        # 拿取get_queue（逻辑上完成预取的任务队列）中的首个元素，即等待一个之前就触发的预取模型（swapin）事件完成。
                        # 拿取只代表逻辑上执行完，实际上可能没执行完，因此需要等待事件的完成。最后返回拿取完成的首个元素中的vt_idx 
                        start_time = time.time()
                        cur_vt_idx = self.prefetch_model_handler.get(vt, suc_vt)
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, 取模型执行了:{execution_time:.6f}秒")
                        get_time.append(execution_time)
                        # 确保之前预取模型的vt就是当前正在执行的vt
                        assert cur_vt_idx == vt.idx
                        if self.verbose: print_gpu_mem(self.rank, vt, "SwapIn(W,B) end")
                        if self.nvprof: nvtx_range_pop() 

                        ### Run through each microbatch in a data batch. 
                        # 2.对每个microbatch执行一次前向得到输出结果，再对输出结果进行反向传播。若当前是第一个BWD任务还需将loss存起来
                        #   从BWD函数的运行角度来说, 就是不断地(每执行一次micorbatch)执行BWD并向前一个BWD任务发送dX
                        m_loss = 0. # loss averaged across examples in this minibatch
                        for i, u in enumerate(vt.ubatchszs):
                            ### Recompute to create pytorch graph
                            # 重计算，即重新执行前向计算，得到该vt的输入和输出
                            start_time = time.time()
                            X_named_tensors, Y_named_tensors = \
                                self._a_pack_forward_an_ubatch(vt, i, u,
                                                            data_ubatches, target_ubatches, 
                                                            requires_grad=True,
                                                            swapin_stashx_handler=self.swapin_stashx_handler,
                                                            swapin_localx_handler=self.swapin_localx_handler,# vDP专用，vPP为None
                                                            swapin_msgx_handler=self.swapin_msgx_handler,
                                                            sucinfo=self.sucinfo)
                            if self.nvprof: nvtx_range_pop() 
                            # 若当前vt是第一个BWD任务, 则返回的Y_named_tensors保存的是loss
                            # 将返回的loss除以microbatch的数量得到一个平均值, 再将该平均值累加到m_loss
                            if 'loss' in Y_named_tensors:
                                # 每个micorbatch计算出来的loss都要除以microbatch的数量
                                Y_named_tensors['loss'] /= len(vt.ubatchszs) # NOTE: ubatches need to be equal
                                m_loss += Y_named_tensors['loss'].item()
                            ### Backward pass on recomputed graph
                            self._a_pack_backward_an_ubatch(vt, i, u,
                                                        X_named_tensors, Y_named_tensors,
                                                        swapin_localx_handler=self.swapin_localx_handler,# vDP专用，vPP为None
                                                        swapout_localx_handler=self.swapout_localx_handler,# vDP专用，vPP为None
                                                        sucinfo=self.sucinfo,
                                                        iteration_num=it)
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, microbatch{i}, 反向执行了:{execution_time:.6f}秒")
                            compute_time.append(execution_time)
                            if self.verbose: print_gpu_mem(self.rank, vt, "End(#%d)" % i)
                            if self.nvprof: nvtx_range_pop()
                            ### Clean up
                            # 每个microbatch结束后，删除掉保存的整个vt的输入输出
                            del X_named_tensors; del Y_named_tensors # very important!
                            start_time = time.time()
                            gc.collect()
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, microbatch{i}, gc.collect执行了:{execution_time:.6f}秒")
                            gc_time.append(execution_time)

                        ### Prefetch point @ AllReduce
                        # 3.预取suc_vt的模型，将其W和B拿到GPU上
                        if not self.args.no_prefetch_model:
                            if self.nvprof: nvtx_range_push("task{}({}) PrefetchPt".format(vt.idx, vt.show_layers() )) 
                            # input：即把vt的所有layer的W和B input 到GPU上
                            # 分析：没有指定流，整个过程应是在默认流上执行
                            # 1.断言当前没有其他的 iput 操作正在执行，确保只有一个 iput 操作可以进行
                            # 2.在默认流上记录一个事件（分配一个新的event），即ev_compute，用于后续在 swapin 流上等待
                            # 3.若给定vt上的所有layer的W和B的媒介为SHM，则按照cpu上参数和buffer的大小和类型，为gpu上的data分配一个大小和类型相同的空tensor
                            # 4.将(vt, ev_compute)添加到 put_queue 中，📌这也意味着该函数所在实例(PrefetchLocalModelGPU)的线程将开始执行swap in
                            self.prefetch_model_handler.iput(suc_vt) 
                            if self.nvprof: nvtx_range_pop() 
                        ### Optional dW aggregation (and B sync)
                        if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                            self.default_stream.synchronize() # TODO: wait Compute by cuda event 
                            if self.nvprof: nvtx_range_push("task{}({}) AllReduce(dW,B)".format(vt.idx, vt.show_layers())) 
                            for l in vt.layers:
                                self.local_model[l].average_grad(self.p2pm_handler)
                                if self.args.average_buffer:
                                    self.local_model[l].average_buf(self.p2pm_handler) # optional: all rank average buffers (can comment out to only use rank0's buf)
                            # TODO: wait AllReduce finish by cuda event
                            if self.nvprof: nvtx_range_pop() 
                        # 保存计算出来的loss
                        if m_loss != 0.:
                            local_losses.append(m_loss)
                            global_losses.append(m_loss)

                        ### Swap-out model {W,dW,B}
                        # 4.直接删除W，卸载dW、B至cpu的shared_memory版本模型（每层自己就是一个模型）上（实际存储于pinned memory）
                        # 该代码没实现不offload的代码，必须执行
                        if self.CONFIGS["opt_offld"]:                   
                            # ### Clip dW for "gpt2_huggingface"
                            #     self.default_stream.synchronize()
                            #     for l in vt.layers:
                            #         torch.nn.utils.clip_grad_norm_(self.local_model[l].model.parameters(), self.args.max_grad_norm) 
                            ### Out {W,dW,B}
                            # 阻塞，直到该流中的任务全部完成
                            self.default_stream.synchronize() # CPU wait
                            if self.nvprof: nvtx_range_push("task{}({}) SwapOut(dW,B)".format(vt.idx, vt.show_layers())) 
                            # 对vt中的所有layer，若dW的输出媒介为LOC，B的输出媒介为SHM，将gpu模型的grad和buffer卸载到cpu的shared_model上
                            # 📌这俩东西实际上存储在pinned memory上，虽然挂在shared_model上（buffer以pinned_buf成员变量挂在shared_model上）
                            start_time = time.time()
                            for l in vt.layers:
                                if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                                    if self.CONFIGS["mode"]=='vPP' or (self.CONFIGS["mode"]=='vDP' and self.rank==0):
                                        # 1.在cpu上按照param的shape和类型分配一个零值tensor
                                        # 2.将刚分配的tensor拷贝到pinned memory中
                                        # 3.将刚刚分配的tensor赋给 shared memory 版本模型（一层就是一个模型）的grad属性
                                        # 4.📌以非阻塞的方式将gpu参数的 grad.data 拷贝到刚刚分配的tensor上，即拷贝到shared_model上
                                        #   shared_model的param.grad实际上在pinned memory上
                                        self.local_model[l].swapout_grad() # Swap-out dW (accumulated)
                                        # 1.若模型存在buffer，在cpu上的固定内存按照buffer的shape和类型分配一个零值tensor
                                        # 2.将gpu上的buffer tensor📌以非阻塞的方式拷贝到cpu固定内存上刚刚分配的零值tensor
                                        # 最终这个pinned_buf会以成员变量（pinned_buf）的形式挂在shared_model上
                                        self.local_model[l].swapout_buf() # Swap-out B (updated)
                                else:
                                    raise NotImplementedError
                            end_time = time.time()
                            execution_time = end_time - start_time
                            swapout_time.append(execution_time)
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, microbatch{i}, 卸载执行了:{execution_time:.6f}秒")
                            if self.verbose: print_gpu_mem(self.rank, vt, "SwapOut'ed(dW,B)")    
                            if self.nvprof: nvtx_range_pop() 
                            ### Delete model {W,dW,B} 
                            # 在默认流上阻塞，等待swapout完成
                            self.default_stream.synchronize() # CPU wait for SwapOut
                            if self.nvprof: nvtx_range_push("task{}({}) Del(W,dW,B)".format(vt.idx, vt.show_layers())) 
                            # 递归地删除每一层（包括子模块）中的所有参数、梯度和缓冲区
                            start_time = time.time()
                            for l in vt.layers:
                                if vt.Out['dW'][l].medium == "LOC" and vt.Out['B'][l].medium == "SHM":
                                    self.local_model[l].del_param_grad_buf() # also del gradient
                                else: # 'B' == PIN
                                    raise NotImplementedError
                            end_time = time.time()
                            execution_time = end_time - start_time
                            delete_time.append(execution_time)
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, 删除执行了:{execution_time:.6f}秒")
                            start_time = time.time()
                            gc.collect()
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print(f"rank{self.rank}, vt.idx:{vt.idx}, {vt.type}, gc.collect执行了:{execution_time:.6f}秒")
                            gc_time.append(execution_time)

                            if self.verbose: print_gpu_mem(self.rank, vt, "Deleted(W,B)")
                            if self.nvprof: nvtx_range_pop() 
                        else:
                            raise ValueError("GPU Optimizer Underdevelopment.")
                        # -----------------------------------------------
                    elif vt.type == 'UPD' and not vt.is_gpu:
                        # -----------------------------------------------
                        ### In {dW,W,K} Out {W,K}
                        # UpdateInBkgd实例的线程在后台进行vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
                        if self.nvprof: nvtx_range_push("task{}({}) UPD(W)".format(vt.idx, vt.show_layers())) 
                        if self.CONFIGS["mode"]=='vPP' or (self.CONFIGS["mode"]=='vDP' and self.rank==0):
                            if not self.args.no_update:
                                # 将给定的vt放入到put_queue队列中，同时将self.the_last_put 设置为 vt.idx。📌这意味着UpdateInBkgd实例的线程会开始执行
                                # vt上的layer的buffer从固定内存到共享内存的复制，以及在shared_memory上进行参数的更新
                                self.update_handler.iput(vt)
                        if self.nvprof: nvtx_range_pop() 
                        # -----------------------------------------------
                    else:
                        raise ValueError("Unknown vTask.type {} with .device {} !".format(vt.type,vt.device))
                    if self.nvprof: nvtx_range_pop() 
                if self.nvprof: nvtx_range_pop() 
                ### tasks iteration ends
                # 等待最后一个放进put_queue中的任务从get_queue中拿出来，即等待update执行完成
                if not self.args.no_update:
                    self.update_handler.synchronize()
                    # 执行完的minibatch数量
                    self.update_cnt += 1
                # torch.cuda.synchronize(self.rank)：等待当前GPU上所有的核函数执行完毕
                torch.cuda.synchronize(self.rank)
                dist.barrier()
                ### statistics
                # 将在当前一个minibatch上执行完当前rank上所有任务花费的时间存起来
                self.time_iters.append(pc()-time_start) 
                if self.nvprof and it == self.args.nvprof_iter["end"]:
                    nvtx_mark("cudaProfilerStop") 
                    cuda_profiler.stop()
                    probe_cuda_mem.stop()
                    print("rank%d: cuda profiler stops"%self.rank)
                ## if it % self.args.display_period == 0:
                ps = "rank%d: Epoch%d/%d Iter%d/%d %.3f sec, %.3f/%.3f GB" % ( 
                    self.rank, epoch, self.args.num_epochs, it, self.args.num_iters, 
                    self.time_iters[-1],
                    float(torch.cuda.memory_allocated()) / 1024**3,# 查看当前GPU的内存占用
                    float(torch.cuda.memory_reserved()) / 1024**3)# 查看向CUDA申请的内存占用
                # 
                if local_losses != []:
                    np.save(os.path.join(self.args.output_dir, "local_losses_rank%d.npy"%self.rank), local_losses)
                # vDP，略
                if self.CONFIGS["mode"] == 'vDP' and self.CONFIGS['N'] > 1:
                    global_losses[-1] = allreduce_cpu_loss(global_losses[-1], averaging=True)
                # 若当前rank是执行第一个BWD任务，即包含计算层的BWD任务，为字符串ps添加loss信息
                if self.rank == self.CONFIGS['loss_rank']:
                    ps += ", Loss %.3f"% global_losses[-1]
                    np.save(os.path.join(self.args.output_dir, "train_losses.npy"), global_losses)
                print(ps)

                self.delete_time.append(sum(delete_time))
                self.get_time.append(sum(get_time))
                self.swapout_time.append(sum(swapout_time))
                self.gc_time.append(sum(gc_time))
                self.compute_time.append(sum(compute_time))

                print(f"rank{self.rank}, 取模型共花费:{self.get_time[-1]}'s, 计算共花费:{self.compute_time[-1]}'s, 删除模型共花费:{self.delete_time[-1]}'s, 卸载梯度buffer共花费:{self.swapout_time[-1]}'s, gc.collect共花费:{self.gc_time[-1]}'s")
                if self.args.no_update:
                    assert self.CONFIGS["mode"] !='vPP'
                    gs = checker.check_grad_sum_harmony(self.shared_model)
                    grad_sums.append(gs)
                    np.save(os.path.join(self.args.output_dir, "grad_sums_rank%d.npy"%self.rank), grad_sums)
                # check GPU OoM & cudaFree & cudaMalloc
                self.cgm.check(it, is_check_malloc=not self.args.empty_cache and len(self.time_iters)-1 >= self.avg_it)
        ### end training
        torch.cuda.synchronize(self.rank)
        dist.barrier()
        print("rank%d: --- done ---" % self.rank)

    # 
    def finish(self): 
        ### statistics
        if self.verbose:
            # vPP p2pm_handler 为None
            print_p2p_bytes(self.rank, self.p2px_handler, self.p2pm_handler, self.update_cnt)

        # self.avg_it：即整个迭代次数的一半
        # 取后一半所有iteration（跑一个minibatch）所需的时间，取平均值，得到平均的一次迭代时间
        avg_iter_time = np.mean(self.time_iters[self.avg_it:]) # sec
        avg_compute_time = np.mean(self.compute_time[self.avg_it:])
        avg_delete_time = np.mean(self.delete_time[self.avg_it:])
        avg_collect_time = np.mean(self.gc_time[self.avg_it:])
        avg_get_time = np.mean(self.get_time[self.avg_it:])
        avg_swapout_time = np.mean(self.swapout_time[self.avg_it:])

        
        # CONFIGS["D"]：minibatchsize
        # 一个minibatch sample的数量/一次迭代的平均时间 = 吞吐量
        avg_throughput = self.CONFIGS['D'] / avg_iter_time # samples/sec
        # 从不同进程中收集 各个rank向CUDA申请的内存占用，目标进程（rank0）会收集所有进程发送的整数，并返回一个整数列表
        gpu_reserved = gather_integer(torch.cuda.memory_reserved(), self.rank) # bytes
        if self.rank == 0:
            gpu_reserved = " ".join("%.1f"%(float(byte)/1024**3) for byte in gpu_reserved) # GB
            # 返回 ["occupied"：已使用的物理内存量]
            cpu_occupied = self.pcm.system_cpu_memory(["occupied"])
            # self.avg_it：所有迭代中间的那一次
            # len(self.time_iters)：迭代次数
            # 
            print("[Global] Iter[%d,%d) Avg Iter Time: %.3f sec, Avg Throughput: %.3f sample/s, GPU: (%s) GB, CPU: %s, Num Updates: %d\n" % (self.avg_it, len(self.time_iters), avg_iter_time, avg_throughput, gpu_reserved, cpu_occupied, self.update_cnt))
            print(f"[Global] 平均计算时间:{avg_compute_time}, 平均删除时间:{avg_delete_time}, 平均垃圾回收时间:{avg_collect_time}, 平均拿取时间:{avg_get_time}, 平均卸载梯度buffer时间:{avg_swapout_time}")
            print(f"[Global] 拿取时间占总体时间的比例:{avg_get_time / avg_iter_time}")
            self.pcm.print("rank%d: eventually" % self.rank)
        ### save model
        # 暂略，默认不保存模型
        # self.args.save_final_model：默认为false
        if self.args.save_final_model and self.rank == 0 and self.save_model is not None:
            self.save_model(self.args, self.shared_model, self.update_cnt)
