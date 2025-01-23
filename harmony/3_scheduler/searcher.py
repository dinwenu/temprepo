# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
import json
import numpy as np
from time import perf_counter as pc
from collections import OrderedDict as ODict
import sys

from layer_packing import greedy_memory_packing, balanced_memory_packing, reuse_memory_packing, balanced_time_packing, balanced_time_packing_2, manual_pack
from task_graph_composor import compose
from task_graph_composor_2 import compose2
from simulator import simulate, sim_res_str

class TopK(object):
    def __init__(self, k=1, hash_fn=None, print_fn=None):
        """ hash_fn: if None, use counter as id (can have duplicate configs); 
                     else, use hash value as id (de-duplicated configs) """
        self.k = k if k > 0 else float('inf')
        # 候选集字典
        self.roster = ODict() # { id: { "time" : 1., "config" : ODict } }
        self.cnt = 0
        self.hash_fn = hash_fn
        self.print_fn = print_fn 
        if self.print_fn is None: 
            self.print_fn = lambda r, t, c: print("Top{}: time {}, config {}".format(r + 1, t, c))

    # 维护一个候选集，即self.roster
    # 若候选集未满(候选集大小为self.k)，直接将time和config添加到候选集
    # 否则，若给定的time小于候选集中的最大时间，删掉候选集中具有最大时间的候选者，将当前time和config添加到候选集
    def add(self, time, config):
        """ rank topK based on this argument time """
        id = self.cnt if self.hash_fn is None else self.hash_fn(config)
        # 若候选集未满，直接将当前time和config添加到候选集中
        if self.cnt < self.k: # topK not full yet
            if id not in self.roster:
                self.roster[id] = ODict({"time":time, "config":config})
                self.cnt += 1      

        # 如果候选集已满，则执行以下逻辑来更新候选集
        else: # topK is full
            # find the worst one
            # 获取候选集中所有元素的key，即id
            topk_id = [key for key in self.roster.keys()]
            topk_time = [v["time"] for v in self.roster.values()]
            # 找到候选集中时间最大的元素的下标
            worst_id = topk_id[ topk_time.index(max(topk_time)) ]
            # if in topK and unique, replace the worst one
            # 若当前给定的time比候选集中最差的候选者的time小，且id不存在于候选集中，删掉候选集中具有最大时间的
            # 候选者，将当前的time和config添加到候选集中
            if time < self.roster[worst_id]['time'] and (id not in self.roster):
                del self.roster[worst_id]
                self.roster[id] = ODict({"time":time, "config":config})
                self.cnt += 1
    
    # 按照执行时间对候选者进行排序，并按照排序建立候选者的config list和相关信息字符串list
    # 返回这两个list
    def summary(self, title=""):
        print("%s" % title)
        
        topk_time = [v["time"] for v in self.roster.values()]
        topk_config = [v["config"] for v in self.roster.values()]
        # 使用 numpy 库中的 argsort 函数对 topk_time 进行排序，并返回排序后的索引
        # 每个元素，即索引的下标代表了排序后的位置，索引则代表了topk_time中对应的元素(time)的位置
        indices = np.argsort(topk_time) # returning index of a sorted list (ascending)
        # print(f"indices:{indices}")

        sorted_config, sorted_print = [], []
        for r, i in enumerate(indices):
            # 按照排好序的顺序取出topk_time中的元素
            time = topk_time[i]
            config = topk_config[i]
            # 打印并返回字符串，描述了候选者的各类信息
            s = self.print_fn(r, time, config)
            sorted_config.append(config)
            sorted_print.append(s)
        
        # 返回按时间从小到大排序的候选者的config，和相关信息字符串
        return sorted_config, sorted_print

def hash_fn(config):
    u_fwd = int(config['CONFIGS']['u_fwd'])
    pack_fwd = config['CONFIGS']['pack_fwd']
    u_bwd = int(config['CONFIGS']['u_bwd'])
    pack_bwd = config['CONFIGS']['pack_bwd']
    # e.g., pack_fwd = []
    # e.g., pack_fwd = [[0, 1, 2, 3, 4...12, 13, 14, 15, 16, 17, 18]]
    # e.g., pack_fwd = [[0, 1, 2, 3], ..., [16, 17, 18, 19, 20]]
    pack_fwd = tuple( tuple(p) for p in pack_fwd )
    pack_bwd = tuple( tuple(p) for p in pack_bwd )
    # hash the immutables
    
    return hash((u_fwd,pack_fwd,u_bwd,pack_bwd))

def print_fn(r, time, config):
    # print_str1 = print_global(config["end2end_times"], config["end2end_memories"], title="Top%d"%(r+1))
    print_str1 = sim_res_str(config["res"], title="Top%d"%(r+1))
    print(print_str1)
    
    u_fwd = config['CONFIGS']['u_fwd']
    pack_fwd = config['CONFIGS']['pack_fwd']
    u_bwd = config['CONFIGS']['u_bwd']
    pack_bwd = config['CONFIGS']['pack_bwd']
    print_str2  = "\tu_fwd   : {}\n".format(u_fwd)
    print_str2 += "\tpack_fwd: {} =\t{}\n".format(len(pack_fwd), pack_fwd)
    print_str2 += "\t{}\n".format(config["packing_method_fwd"])
    print_str2 += "\tu_bwd   : {}\n".format(u_bwd)
    print_str2 += "\tpack_bwd: {} =\t{}\n".format(len(pack_bwd), pack_bwd)
    print_str2 += "\t{}".format(config["packing_method_bwd"])
    print(print_str2)
    
    return print_str1 + "\n" + print_str2
    # print("\tu_fwd   : {}".format(u_fwd))
    # print("\tpack_fwd: {}x =\t{}".format(len(pack_fwd), pack_fwd))
    # print("\t{}".format(config["packing_method_fwd"]))
    # print("\tu_bwd   : {}".format(u_bwd))
    # print("\tpack_bwd: {}x =\t{}".format(len(pack_bwd), pack_bwd))
    # print("\t{}".format(config["packing_method_bwd"]))  

def is_equal_ubatchsize(args, ubatchsize):
    """ for both Ufwd and Ubwd """
    D, N = args.minibatchsize, args.num_gpus
    if args.mode == 'vDP':
        is_equal = True
        for n in range(N):
            # ----- find per-GPU microbatch sizes -----
            DD = int(float(D)/N)
            if D%N != 0: # uneven batch size across GPUs
                if n < D%N:
                    DD += 1
            assert DD >= ubatchsize
            if DD % ubatchsize != 0:
                is_equal = False
                break
        return is_equal
    
    # 对vPP来说，只有minibatchsize能整除microbatchsize，返回的才是true
    elif args.mode == 'vPP':
        assert D >= ubatchsize
        return D % ubatchsize == 0
    else:
        raise ValueError

# 像是一个检查完整性的函数，即
# 1.确保microbatch的大小不超过minibatch的大小
# 2.筛选出能被minibatchsize整除的ubatchsize
# 返回：sorted(ubatchsizes_fwd), sorted(ubatchsizes_bwd)
def find_ubatchsizes(args):
    """ find valid ubatch sizes to search """
    # ubatchsize = 1 ~ min(Umax,min(DD)) for vDP both fwd and bwd
    # ubatchsize = 1 ~ min(Umax,D) for vPP both fwd and bwd
    Umin_fwd, Umax_fwd = min(args.fwd_ubatchsizes), max(args.fwd_ubatchsizes)
    Umin_bwd, Umax_bwd = min(args.bwd_ubatchsizes), max(args.bwd_ubatchsizes)
    D, N = args.minibatchsize, args.num_gpus    
    if args.mode == 'vDP':
        D = int(float(D)/N) # namely, min(DD)
    elif args.mode == 'vPP':
        D = D
    else:
        raise ValueError
    # 最大的ubatchsize为，fwd_ubatchsizes的最大值，和minibatchsize之间的最小值。即microbatch的大小不允许
    # 超过minibatch的大小
    ubatchsizes_fwd = list(range(Umin_fwd, min(Umax_fwd, D)+1, args.ubatchsize_step))
    ubatchsizes_bwd = list(range(Umin_bwd, min(Umax_bwd, D)+1, args.ubatchsize_step))
    
    # then select equal ubatchsize if needed (DD/D % ubatchsize == 0)
    # 该参数默认为false，这里执行: 筛选出能被minibatchsize整除的ubatchsize
    if not args.inequal_ubatchsize:
        #                                                1.assert D >= ubatchsize，确保ubatchsize不超过minibatchsize
        #                                                2.对vPP来说，只有minibatchsize能整除u，返回的才是true
        #                                                  📌因此得到的microbatchsize一定是和minibatchsize的次幂表达式的底数相同
        ubatchsizes_fwd = [u for u in ubatchsizes_fwd if is_equal_ubatchsize(args, u)]
        ubatchsizes_bwd = [u for u in ubatchsizes_bwd if is_equal_ubatchsize(args, u)]
    else:
        print("[WARNING] allow inequal microbatchsize will disable double buffering. Although we can still search, runtime is not supported currently.")
    
    return sorted(ubatchsizes_fwd), sorted(ubatchsizes_bwd)

# 根据参数指定的方式，对所有层进行打包
# blanced_time：按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
# 在首个符合GPU容量的打包完成后，不会再进行后续的打包
# 返回一个字典：{ "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs） }
def find_layer_packs(args, ubatchsize, num_layers, type, reuse_packs=None, verbose=True):
    """ for an ubatchsize, find valid layer packs to search """
    assert type in ["FWD","BWD"]
    
    # build per_layer_memories list
    memory_list = []; x_list = []
    # 为每一层layer，分别为两个列表添加一个值，分别为层的占用大小，和该层输入的大小
    for l in range(num_layers):
        mem  = args.prof['MEMORY_FWDBWD'].get(type, ubatchsize, l, interp=True) # int
        xmem = args.prof["XMETA"].get_bytes(ubatchsize, l, interp=True)# int
        # MEMORY_FWDBWD本身就包含了这一层的输入和输出，减xmem即减去这一层输入的大小
        # 📌每层的空间占用为，参数+输出
        memory_list.append(mem - xmem) # bytes 
        # 每层的输入大小
        x_list.append(xmem) # bytes
    
    # different packing methods
    layer_packs = ODict() # { "greedy" : [[0,1,2], [3,4,5]] }
    packing_method = args.packing_method_fwd if type == 'FWD' else \
                     args.packing_method_bwd
    print(f"packing_method:{packing_method}")

    tab = "\t\t\t" if type == 'FWD' else "\t"
    # 按照给定的参数，执行对应的packing策略
    for method in packing_method:
        if method == "greedy":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "greedy_addx":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="greedy_memory_packing (addx)", tab=tab)
        elif method == "greedy_reverse":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, verbose=verbose,title="greedy_memory_packing (reverse)", tab=tab)
        elif method == "greedy_reverse_addx":
            # 使用贪婪方法打包层，即只要当前pack中所有的层+首层的输入不超过GPU的容量，就一直往pack中添加新的layer。否则，开始下一个pack
            # 其内部会逆序的进行打包，但最终在返回前会对每个list(layer pack)进行翻转
            # 返回：layer_pack（list），即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, per_layer_x=x_list, verbose=verbose,title="greedy_memory_packing (reverse,addx)", tab=tab)
        elif method == "balanced":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_addx":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="balanced_memory_packing (addx)", tab=tab)
        elif method == "reuse":
            # 对packs中的所有pack计算，该pack所有层的大小+该pack首层的输入大小。选出其中的最大值
            # 若最大值 < GPU容量，返回该packs
            # 若最大的pack > GPU容量，返回None，表示不能reuse packing
            layer_packs[method] = reuse_memory_packing(reuse_packs, memory_list, x_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_time":
            # 所有层BWD的执行时间的list
            time_list = [ args.prof['TIME_FWDBWD'].get(type, ubatchsize, l, "GPU",interp=True) for l in range(num_layers) ] # sec (float)
            # 按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
            # 在首个符合GPU容量的打包完成后，不会再进行后续的打包
            # 即从将整个model切成两份开始，检测按时间均匀方法切割的layer pack有没有超过GPU的容量。没超过直接返回，超过了继续测试切成
            # 3份，以此类推
            # 返回layer_packs，该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
            layer_packs[method] = balanced_time_packing(time_list, memory_list, x_list, args.memory_cap, verbose=verbose, tab=tab)
        # skip invalid packing
        # 删掉字典中没有值的键值对
        if layer_packs[method] is None:
            del layer_packs[method]
    
    # { "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs） }
    return layer_packs

# 已废弃
def find_layer_packs_2(args, ubatchsize, num_layers, type, reuse_packs=None, verbose=True):
    """ for an ubatchsize, find valid layer packs to search """
    assert type in ["FWD","BWD"]
    
    # build per_layer_memories list
    memory_list = []; x_list = []
    # 为每一层layer，分别为两个列表添加一个值，分别为层的占用大小，和该层输入的大小
    for l in range(num_layers):
        mem  = args.prof['MEMORY_FWDBWD'].get(type, ubatchsize, l, interp=True) # int
        xmem = args.prof["XMETA"].get_bytes(ubatchsize, l, interp=True)# int
        # MEMORY_FWDBWD本身就包含了这一层的输入和输出，减xmem即减去这一层输入的大小
        # 📌每层的空间占用为，参数+输出
        memory_list.append(mem - xmem) # bytes 
        # 每层的输入大小
        x_list.append(xmem) # bytes
    
    # different packing methods
    layer_packs = ODict() # { "greedy" : [[0,1,2], [3,4,5]] }
    packing_method = args.packing_method_fwd if type == 'FWD' else \
                     args.packing_method_bwd
    print(f"packing_method:{packing_method}")

    tab = "\t\t\t" if type == 'FWD' else "\t"
    # 按照给定的参数，执行对应的packing策略
    for method in packing_method:
        if method == "greedy":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "greedy_addx":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="greedy_memory_packing (addx)", tab=tab)
        elif method == "greedy_reverse":
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, verbose=verbose,title="greedy_memory_packing (reverse)", tab=tab)
        elif method == "greedy_reverse_addx":
            # 使用贪婪方法打包层，即只要当前pack中所有的层+首层的输入不超过GPU的容量，就一直往pack中添加新的layer。否则，开始下一个pack
            # 其内部会逆序的进行打包，但最终在返回前会对每个list(layer pack)进行翻转
            # 返回：layer_pack（list），即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
            layer_packs[method] = greedy_memory_packing(memory_list, args.memory_cap, reverse=True, per_layer_x=x_list, verbose=verbose,title="greedy_memory_packing (reverse,addx)", tab=tab)
        elif method == "balanced":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_addx":
            layer_packs[method] = balanced_memory_packing(memory_list, args.memory_cap, per_layer_x=x_list, verbose=verbose, title="balanced_memory_packing (addx)", tab=tab)
        elif method == "reuse":
            # 对packs中的所有pack计算，该pack所有层的大小+该pack首层的输入大小。选出其中的最大值
            # 若最大值 < GPU容量，返回该packs
            # 若最大的pack > GPU容量，返回None，表示不能reuse packing
            layer_packs[method] = reuse_memory_packing(reuse_packs, memory_list, x_list, args.memory_cap, verbose=verbose, tab=tab)
        elif method == "balanced_time":
            # 所有层BWD的执行时间的list
            time_list = [ args.prof['TIME_FWDBWD'].get(type, ubatchsize, l, "GPU",interp=True) for l in range(num_layers) ] # sec (float)
            # 按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
            # 在首个符合GPU容量的打包完成后，不会再进行后续的打包
            # 即从将整个model切成两份开始，检测按时间均匀方法切割的layer pack有没有超过GPU的容量。没超过直接返回，超过了继续测试切成
            # 3份，以此类推
            # 返回layer_packs，该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
            layer_packs[method] = balanced_time_packing_2(time_list, memory_list, x_list, args.memory_cap, args.num_gpus, verbose=verbose, tab=tab)
        # skip invalid packing
        # 删掉字典中没有值的键值对
        if layer_packs[method] is None:
            del layer_packs[method]
    
    # { "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs） }
    return layer_packs

# 
def search(args):
    """ top-level function """
    """ search for the best configuration (Ufwd, Pfwd, Ubwd, Pbwd) for min estimated runtime under memory capacity constraints. """
    
    ### find microbatch sizes to search
    # 像是一个检查完整性的函数，即对FWD的microbatchsize和BWD的microbatchsize列表中的每个microbatchsize执行：
    # 1.确保microbatch的大小不超过minibatch的大小
    # 2.📌筛选出能被minibatchsize整除的ubatchsize (所以说不是每个ubatchsize都能被用)
    # 返回：sorted(ubatchsizes_fwd), sorted(ubatchsizes_bwd)
    ubatchsizes_fwd, ubatchsizes_bwd = find_ubatchsizes(args)
    if args.verbose: 
        print("searchable ubatchsizes_fwd: {}".format(ubatchsizes_fwd)) 
        print("searchable ubatchsizes_bwd: {}".format(ubatchsizes_bwd)) 
    ubatchsizes_fwd = np.array(ubatchsizes_fwd, dtype=np.uint64)
    
    ### find valid ubatch size and layer packs
    valid_size_pack = [] # [(u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd)]
    t_start = pc()
    ## search BWD first
    # 
    for u_bwd in ubatchsizes_bwd: 
        if args.verbose: print("\nFor u_bwd: %d, find pack_bwd..."%u_bwd)

        # 根据参数指定的方式，对所有层进行打包
        # blanced_time：按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
        # 在首个符合GPU容量的打包完成后，不会再进行后续的打包
        # 返回一个字典：{ "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs） }
        method_packs_bwd = find_layer_packs(args, u_bwd, args.num_layers, "BWD", verbose=args.verbose)
        # under each BWD packing, search FWD
        for method_bwd, pack_bwd in method_packs_bwd.items():
            if args.verbose: print("\tFor %s, find FWD..."%method_bwd)
            assert len(pack_bwd) > 0
            # 若只打了一个包，则代表该任务就是一个单个BWD任务，不用找FWD了
            if len(pack_bwd) == 1: # Empty (single BWD pack)
                if args.verbose: print("\t\tFWD is empty")
                u_fwd = u_bwd
                pack_fwd = []
                valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, "", method_bwd))

            # 
            else: # search FWD
                # 取pack_bwd中倒数第二个pack的最后一个layer_id，并+1
                # 即除最后一个pack外，前面的pack总共有多少层
                num_layers_fwd = pack_bwd[:-1][-1][-1] + 1
                print(f"\t去除BWD的最后一个pack, 前面的pack总共有多少层:{num_layers_fwd}")

                # assert num_layers_fwd == sum(len(p) for p in pack_bwd[:-1])
                # 该参数默认为false，不执行
                if args.smaller_ufwd: # allow u_fwd < u_bwd
                    idx = 0
                    print("[WARNING] allow microbatch size of forward be smaller than backward is still bleeding.")
                else: # make u_fwd starts from u_bwd
                    # 找到 u_bwd 应该在 ubatchsizes_fwd 中插入的位置，即fwd中对应u_bwd的位置
                    idx = np.searchsorted(ubatchsizes_fwd,u_bwd,side='left') # works for non-valid u_fwd, valid u_fwd, even out of range u_fwd
                    print(idx)

                # 从fwd micro batch size的起始位置开始遍历（起始位置就是从当前u_bwd的大小开始）
                for u_fwd in ubatchsizes_fwd[idx:]: 
                    u_fwd = int(u_fwd)
                    if args.verbose: print("\t\tFor u_fwd: %d, find pack_fwd..."%u_fwd)
                    # 返回一个字典：{ "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs），"reuse"：pack_bwd[:-1] }
                    method_packs_fwd = find_layer_packs(args, u_fwd, num_layers_fwd, "FWD", reuse_packs=pack_bwd[:-1], verbose=args.verbose)
                    print(f"method_packs_fwd:{method_packs_fwd}")
                    for method_fwd, pack_fwd in method_packs_fwd.items():
                        valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd))
    
    ### evaluate each valid one
    print("\nEvalute valid size and packs (%d points) ..."%len(valid_size_pack))
    if args.verbose: 
        print("< u_fwd, num_pack_fwd, u_bwd, num_pack_bwd, packing_method_fwd, packing_method_bwd >")
    # 
    top_k = TopK(args.topk, hash_fn if args.dedup else None, print_fn)
    # 
    for u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd in valid_size_pack:
        ## compose task graph
        if args.mode == 'vPP' and args.num_gpus > 1 and u_fwd != u_bwd:
            args.last_fwd_msg = True
        else:
            args.last_fwd_msg = False
        
        CONFIGS, TASKS, rTASKS = compose(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose=True)
        ## estimate runtime 
        res = simulate(args, rTASKS, CONFIGS, TASKS=TASKS, prefetch_offload=args.prefetch_offload, verbose=False, view=False)
        if args.verbose: 
            print(sim_res_str(res, 
                              title="< %d, %d, %d, %d, %s, %s >"%
                              (u_fwd, len(pack_fwd), u_bwd, len(pack_bwd), 
                               method_fwd[:6].ljust(6,' '), 
                               method_bwd[:6].ljust(6,' ') ) ) )
        ## compare for the best
        global_time = res['global_endtime']
        # 若反向的pack只有一个，说明整个model能放进一个GPU中，打印一条警告，建议不使用Harmony方法
        if len(pack_bwd) == 1:
            # 该参数默认为false，执行
            if not args.rank_fit_normally:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is now put as Top1.")
                global_time /= 1000.
            else:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is still ranked normally.")
        ## add to top k best
        # 维护一个候选集，即self.roster
        # 若候选集未满(候选集大小为self.k)，直接将time和config添加到候选集
        # 否则，若给定的time小于候选集中的最大时间，删掉候选集中具有最大时间的候选者，将当前time和config添加到候选集
        top_k.add(  global_time, 
                    ODict({ 'res': res, 
                            'CONFIGS': CONFIGS,
                            'rTASKS': rTASKS,
                            'packing_method_fwd': method_fwd,
                            'packing_method_bwd': method_bwd })) 
    
    t_end = pc()
    print("\n--- Search done: %.3f sec ---"%(t_end-t_start))
    
    return top_k

def search2(args):
    """ top-level function """
    """ search for the best configuration (Ufwd, Pfwd, Ubwd, Pbwd) for min estimated runtime under memory capacity constraints. """
    
    ### find microbatch sizes to search
    # 像是一个检查完整性的函数，即对FWD的microbatchsize和BWD的microbatchsize列表中的每个microbatchsize执行：
    # 1.确保microbatch的大小不超过minibatch的大小
    # 2.📌筛选出能被minibatchsize整除的ubatchsize (所以说不是每个ubatchsize都能被用)
    # 返回：sorted(ubatchsizes_fwd), sorted(ubatchsizes_bwd)
    ubatchsizes_fwd, ubatchsizes_bwd = find_ubatchsizes(args)
    if args.verbose: 
        print("searchable ubatchsizes_fwd: {}".format(ubatchsizes_fwd)) 
        print("searchable ubatchsizes_bwd: {}".format(ubatchsizes_bwd)) 
    ubatchsizes_fwd = np.array(ubatchsizes_fwd, dtype=np.uint64)
    
    ### find valid ubatch size and layer packs
    valid_size_pack = [] # [(u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd)]
    t_start = pc()
    ## search BWD first
    # 
    for u_bwd in ubatchsizes_bwd: 
        if args.verbose: print("\nFor u_bwd: %d, find pack_bwd..."%u_bwd)

        # 根据参数指定的方式，对所有层进行打包
        # blanced_time：按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
        # 在首个符合GPU容量的打包完成后，不会再进行后续的打包
        # 返回一个字典：{ "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs） }
        method_packs_bwd = find_layer_packs(args, u_bwd, args.num_layers, "BWD", verbose=args.verbose)
        # under each BWD packing, search FWD
        for method_bwd, pack_bwd in method_packs_bwd.items():
            if args.verbose: print("\tFor %s, find FWD..."%method_bwd)
            assert len(pack_bwd) > 0
            # 若只打了一个包，则代表该任务就是一个单个BWD任务，不用找FWD了
            if len(pack_bwd) == 1: # Empty (single BWD pack)
                if args.verbose: print("\t\tFWD is empty")
                u_fwd = u_bwd
                pack_fwd = []
                valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, "", method_bwd))

            # 
            else: # search FWD
                # 取pack_bwd中倒数第二个pack的最后一个layer_id，并+1
                # 即除最后一个pack外，前面的pack总共有多少层
                num_layers_fwd = pack_bwd[:-1][-1][-1] + 1
                print(f"\t去除BWD的最后一个pack, 前面的pack总共有多少层:{num_layers_fwd}")

                # assert num_layers_fwd == sum(len(p) for p in pack_bwd[:-1])
                # 该参数默认为false，不执行
                if args.smaller_ufwd: # allow u_fwd < u_bwd
                    idx = 0
                    print("[WARNING] allow microbatch size of forward be smaller than backward is still bleeding.")
                else: # make u_fwd starts from u_bwd
                    # 找到 u_bwd 应该在 ubatchsizes_fwd 中插入的位置，即fwd中对应u_bwd的位置
                    idx = np.searchsorted(ubatchsizes_fwd,u_bwd,side='left') # works for non-valid u_fwd, valid u_fwd, even out of range u_fwd
                    print(idx)

                # 从fwd micro batch size的起始位置开始遍历（起始位置就是从当前u_bwd的大小开始）
                for u_fwd in ubatchsizes_fwd[idx:]: 
                    u_fwd = int(u_fwd)
                    if args.verbose: print("\t\tFor u_fwd: %d, find pack_fwd..."%u_fwd)
                    # 返回一个字典：{ "balanced_time"(打包方法)：[[0,1,2], [3,4,5]]（layer_packs），"reuse"：pack_bwd[:-1] }
                    method_packs_fwd = find_layer_packs(args, u_fwd, num_layers_fwd, "FWD", reuse_packs=pack_bwd[:-1], verbose=args.verbose)
                    print(f"method_packs_fwd:{method_packs_fwd}")
                    for method_fwd, pack_fwd in method_packs_fwd.items():
                        valid_size_pack.append((u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd))
    
    ### evaluate each valid one
    print("\nEvalute valid size and packs (%d points) ..."%len(valid_size_pack))
    if args.verbose: 
        print("< u_fwd, num_pack_fwd, u_bwd, num_pack_bwd, packing_method_fwd, packing_method_bwd >")
    # 
    top_k = TopK(args.topk, hash_fn if args.dedup else None, print_fn)
    # 
    for u_fwd, pack_fwd, u_bwd, pack_bwd, method_fwd, method_bwd in valid_size_pack:
        ## compose task graph
        if args.mode == 'vPP' and args.num_gpus > 1 and u_fwd != u_bwd:
            args.last_fwd_msg = True
        else:
            args.last_fwd_msg = False
        
        CONFIGS, TASKS, rTASKS = compose2(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose=True)
        ## estimate runtime 
        res = simulate(args, rTASKS, CONFIGS, TASKS=TASKS, prefetch_offload=args.prefetch_offload, verbose=False, view=False)
        if args.verbose: 
            print(sim_res_str(res, 
                              title="< %d, %d, %d, %d, %s, %s >"%
                              (u_fwd, len(pack_fwd), u_bwd, len(pack_bwd), 
                               method_fwd[:6].ljust(6,' '), 
                               method_bwd[:6].ljust(6,' ') ) ) )
        ## compare for the best
        global_time = res['global_endtime']
        # 若反向的pack只有一个，说明整个model能放进一个GPU中，打印一条警告，建议不使用Harmony方法
        if len(pack_bwd) == 1:
            # 该参数默认为false，执行
            if not args.rank_fit_normally:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is now put as Top1.")
                global_time /= 1000.
            else:
                print("[WARNING] The entire model can fit onto a single GPU. Not recommend to use Harmony. The fit config is still ranked normally.")
        ## add to top k best
        # 维护一个候选集，即self.roster
        # 若候选集未满(候选集大小为self.k)，直接将time和config添加到候选集
        # 否则，若给定的time小于候选集中的最大时间，删掉候选集中具有最大时间的候选者，将当前time和config添加到候选集
        top_k.add(  global_time, 
                    ODict({ 'res': res, 
                            'CONFIGS': CONFIGS,
                            'rTASKS': rTASKS,
                            'packing_method_fwd': method_fwd,
                            'packing_method_bwd': method_bwd })) 
    
    t_end = pc()
    print("\n--- Search done: %.3f sec ---"%(t_end-t_start))
    
    return top_k

    
