# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse
from collections import OrderedDict as ODict
from copy import deepcopy
import pickle

import sys; sys.path.append("../2_profiler")
from prof_data_struct import load_prof_data_struct

from sched_args import initialize_args
from layer_packing import manual_pack
from task_graph_composor import compose
from searcher import search
from simulator import simulate, sim_res_str
from task_data_struct import serialize_scheduled

# 使用手工配置的超参数（前后向Micro batch的大小、GPU数量）
def manual_schedule(args):
    ### add num of layers    
    xmeta = load_prof_data_struct(args.module_path, "prof_XMETA" + args.suffix, base_dir="my_prof")
    # 模型的层数
    args.num_layers = len(xmeta.get_vlayer_ids())
    print("模型的层数为：", args.num_layers)
    ### manual size data and pack layers
    if args.manual_packsize != -1:
        # pack_size：一个pack有多少层
        # 返回正向的打包方案和反向的打包方案
        # ❓为什么正向没有最后一组layer？
        # 正向:[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20]]
        # 反向：[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
        args.manual_pack_fwd, args.manual_pack_bwd = manual_pack(args.num_layers, args.manual_packsize)
    else: # DEBUG
        if args.module_name in ("bert_large", "bert_thomwolf", "bert_seq", "bert_2bw"):
            assert args.num_layers == 28 # 0~26 regular Bert + 27th criterion
            if args.manual_numpacks == 4:
                args.manual_pack_fwd = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
                args.manual_pack_bwd = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27]]
            elif args.manual_numpacks == 5:
                args.manual_pack_fwd = [list(range(0,14))]
                args.manual_pack_bwd = [list(range(0,4)), list(range(4,8)), list(range(8,14)), list(range(14,28))]
            elif args.manual_numpacks == 9:
                args.manual_pack_fwd = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]] 
                args.manual_pack_bwd = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25, 26, 27]]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    # ----- generate task graph -----
    if args.verbose:
        print("\n< Ufwd {}, Pfwd {}, Ubwd {}, Pbwd {} >\n".format(
            args.manual_ufwd, args.manual_pack_fwd, args.manual_ubwd, args.manual_pack_bwd))  
    if args.mode == 'vPP' and args.num_gpus > 1 and args.manual_ufwd != args.manual_ubwd:
        args.last_fwd_msg = True
    else:
        args.last_fwd_msg = False
    # 
    CONFIGS, TASKS, rTASKS = compose(args, args.manual_ufwd, args.manual_pack_fwd, args.manual_ubwd, args.manual_pack_bwd, verbose=args.verbose)
    #------ estimate runtime --------
    if args.simulation:
        ### read all profiles
        prof = ODict()
        # "prof_TIME_FWDBWD","prof_MEMORY_FWDBWD","prof_XMETA","prof_TMETA","prof_TIME_UPD","prof_WMETA","prof_BMETA","prof_KMETA"
        for name in args.profile_fnames:
            key = name.split("prof_")[-1]
            prof[key] = load_prof_data_struct(args.module_path, name + args.suffix)
        args.prof = prof
        ### simulation
        print(f"args.prefetch_offload:{args.prefetch_offload}")
        res = simulate(args, rTASKS, CONFIGS, TASKS=TASKS, prefetch_offload=args.prefetch_offload, verbose=args.verbose, view=args.view)
        print(sim_res_str(res, 
                title="< %d, %d, %d, %d >"%
                (args.manual_ufwd, len(args.manual_pack_fwd), args.manual_ubwd, len(args.manual_pack_bwd)) ) )
    # ----- serialize into pickle -----
    fname = "D%d_%s_N%d_Ufwd%d_Ubwd%d"% (CONFIGS["D"], CONFIGS["mode"],CONFIGS["N"],CONFIGS["u_fwd"], CONFIGS["u_bwd"])
    if args.manual_packsize != -1:
        fname += "_P%d" % args.manual_packsize
    elif args.manual_numpacks != -1:
        fname += "_numP%d" % args.manual_numpacks
    print()
    # 把按rank放好的task字典 和 配置字典 保存为文件
    serialize_scheduled(rTASKS, CONFIGS, args.output_dir, fname + args.suffix, base_dir="my_sched")

# 自动搜索最优的配置
def search_schedule(args):
    ### read profiles
    prof = ODict()
    # 从文件中读取2阶段的检测结果
    for name in args.profile_fnames:
        key = name.split("prof_")[-1]
        prof[key] = load_prof_data_struct(args.module_path, name + args.suffix, base_dir="my_prof")
    args.prof = prof
    args.num_layers = len(prof['XMETA'].get_vlayer_ids())
    # 返回在检测最大的microbatch size的过程中，所有FWD(≤最大的fwd_ubsize)的microbatchsize的列表。[1,2,3,4,...]
    args.fwd_ubatchsizes = prof['TIME_FWDBWD'].get_ubatchsizes('FWD')
    # 返回在检测最大的microbatch size的过程中，所有BWD(≤最大的bwd_ubsize)的microbatchsize的列表。[1,2,3,4,...]
    args.bwd_ubatchsizes = prof['TIME_FWDBWD'].get_ubatchsizes('BWD')
    if args.verbose: 
        print("number of layers: %d" % args.num_layers)
        print("fwd ubatchsizes: {}".format(args.fwd_ubatchsizes))
        print("bwd_ubatchsizes: {}".format(args.bwd_ubatchsizes))
    ### start search
    top_k = search(args)
    ### save TopK schedules
    # 按照执行时间从小打到大候选者进行排序，并按照排序建立候选者的config list和相关信息字符串list
    # 返回这两个list
    sorted_ods, sorted_print = top_k.summary()

    ## schedule.pickle
    print()
    fname = "D%d_%s_N%d"%(args.minibatchsize, args.mode, args.num_gpus)
    
    for r, od in enumerate(sorted_ods):
        print("==============================")
        print(od["CONFIGS"]['ubatchszs_fwd'])
        print(od["CONFIGS"]['ubatchszs_bwd'])
        print("==============================")

    for rank, vts in od["rTASKS"].items():
        print(f"rank:{rank}:")
        for vt in vts:
            print(vt)

    # exit(0)

    # 把tok_k个候选者的rTAKS和相关的config写入到文件中
    for r, od in enumerate(sorted_ods):
        # "_ufwd%d_ubwd%d" % (od["CONFIGS"]["u_fwd"], od["CONFIGS"]["u_bwd"])
        serialize_scheduled(od["rTASKS"], od["CONFIGS"], args.output_dir, fname + "_Top%d" % (r + 1) + args.suffix, base_dir="my_sched")
    ## summary.txt
    summary_path = os.path.join(args.output_dir, "my_sched", fname + "_Summary" + args.suffix + ".txt")
    # 将TopK个候选者的相关字符串信息写入到txt文件中
    with open(summary_path, "wt") as f:
        f.write("\n".join(sorted_print))
    print("searched summary saved: %s" % summary_path)
    ## global time list
    # 得到每个候选者的执行时间的list，从小到大排序
    sorted_global_time = [ od['res']['global_endtime'] for od in sorted_ods ]
    # 
    gt_path = os.path.join(args.output_dir, "my_sched", fname + "_GlobalTime" + args.suffix + ".pickle")
    # 将排好序的每个候选者的执行时间写入到.pickle文件中
    with open(gt_path,'wb') as f:
        pickle.dump(sorted_global_time, f)
    print("global time list saved: %s" % gt_path)

if __name__ == "__main__":
    
    args = initialize_args()
    
    if args.manual:
        manual_schedule(args)
    else:
        search_schedule(args)
    
