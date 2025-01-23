# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict as ODict
from copy import deepcopy

from task_data_struct import *

def compose_task_configs(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose=False):
    CONFIGS = ODict()
    CONFIGS["R"] = args.num_layers
    CONFIGS["mode"] = args.mode
    CONFIGS["N"] = args.num_gpus
    CONFIGS["D"] = args.minibatchsize
    CONFIGS["u_fwd"] = u_fwd # microbatch size of forward
    CONFIGS["ubatchszs_fwd"] = None # add from TASKS
    CONFIGS["pack_fwd"] = pack_fwd # 正向的打包方案
    CONFIGS["u_bwd"] = u_bwd
    CONFIGS["ubatchszs_bwd"] = None # add from TASKS
    CONFIGS["pack_bwd"] = pack_bwd
    CONFIGS["reverse_bwd"] = args.reverse_bwd
    CONFIGS["opt_offld"] = args.offload_optim if hasattr(args, 'offload_optim') else not args.no_offload_optim
    CONFIGS["last_fwd_msg"] = args.last_fwd_msg
    CONFIGS['loss_rank'] = None # add from TASKS
    
    return CONFIGS

# 根据设置的minibatchsize参数，生成一个列表，值为manual_ufwd/ubwd，长度为minibatchsize/manual_ufwd
# D：minibatchsize
# U：microbatch size of forward/backward
#                      (32, 2)
def split_minibatchsize(D, U):
    assert isinstance(D, int) and isinstance(U, int)
    assert D >= U
    if D % U == 0:
        ubatchszs = [U] * int(D/U)
    else:
        ubatchszs = [U] * int(D/U) + [ D%U ]
    assert sum(ubatchszs) == D
    return ubatchszs

# 1.为fwd layer packs的每个layer pack生成一个类型为FWD的 vTask 的实例；为bwd layer packs的每个pack生成一个类型
#   为BWD的 vTask实例，和一个类型为UDP的实例
# 2.将反向任务和其对应的更新任务放在一起，即JIT(just-in-time)更新。而后更新所有任务的idx，即任务的执行顺序
# 3.使用round-robin方法，将task分配到GPU上。换句话说，就是给每个task的device属性赋值
# 4.设置每一个task的数据依赖，即根据task和其前后task是否在一个GPU上，设置其输入输出该通过什么途径传递。
#   换句话说，就是为task的In和Out属性赋值，这俩都是字典，形式为{'X' : {l:Medium()}, ...}。
#   前向：每个task的输入(传进来的输入X、参数W、Buffer B)，输出(输出的Y、X、、W、B)
#   后向：输入(dY, InputX（首个BWD任务）/StashX W, B, T )，输出(dX, dW, W, B, L1)
#   更新：输入(dW，W，K)，输出(W，K)
# 5.若某些层的前向后向任务在一个GPU上，则这俩任务为可立即执行的pack。将这俩任务 W 和 B 连接的媒介设置为PIN(fwd的Out，
#   bwd的In)，即前向后 W B 不移动，反向直接用
def compose_task_graph(CONFIGS, verbose=False):
    """ use (u_fwd, pack_fwd, u_bwd, pack_bwd) to compose a task graph """
    R = CONFIGS["R"] # 模型层数
    mode = CONFIGS["mode"]
    N = CONFIGS["N"] # GPU数量
    D = CONFIGS["D"] # minibatch size
    u_fwd = CONFIGS["u_fwd"] # microbatch size of forward
    pack_fwd = CONFIGS["pack_fwd"] # 正向的打包方案
    u_bwd = CONFIGS["u_bwd"]
    pack_bwd = CONFIGS["pack_bwd"] # 反向的打包方案
    reverse_bwd = CONFIGS["reverse_bwd"]
    opt_offld = CONFIGS["opt_offld"]
    last_fwd_msg = CONFIGS["last_fwd_msg"] # 脚本中未设置该参数

    print("GPU数量为: ", N)
    print(f"The global minibatch size: {D}")
    
    TASKS = []
    if mode == 'vPP':
        # ----- find microbatch sizes -----
        # 根据设置的minibatchsize参数，生成一个列表，值为manual_ufwd/ubwd，长度为 minibatchsize/manual_ufwd
        # ubatchszs_fwd=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        # 代表16个的microbatch，列表中的每个值代表都是一个micorbatch size，加起来即为 minibatch
        ubatchszs_fwd = split_minibatchsize(D, u_fwd)
        ubatchszs_bwd = split_minibatchsize(D, u_bwd)
        print(f"ubatchszs_fwd的长度为: {len(ubatchszs_bwd)}")
        if verbose: print("ubatchszs_fwd={}, ubatchszs_bwd={}".format(ubatchszs_fwd, ubatchszs_bwd))

        # ----- create tasks from sized data and packed layers -----
        # 1.为fwd layer packs的每个layer pack生成一个类型为FWD的 vTask 的实例；为bwd layer packs的每个pack生成一个类型
        #   为BWD的 vTask实例，和一个类型为UDP的实例
        for a_pack in pack_fwd: # [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20]]
            vt = vTask( layers = a_pack, 
                        type = 'FWD', 
                        ubatchszs = ubatchszs_fwd )
            TASKS.append(vt)
        for a_pack in pack_bwd: # [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
            vt = vTask( layers=a_pack, 
                        type='BWD',
                        ubatchszs = ubatchszs_bwd )
            TASKS.append(vt)
        for a_pack in pack_bwd:
            vt = vTask( layers=a_pack, 
                        type='UPD' )
            TASKS.append(vt)
        if verbose: print_tasks(TASKS, "created tasks (sized data & packed layers)")

        # ----- order tasks with jit update -----    jit即立即执行更新操作
        # 2.将反向任务和其对应的更新任务放在一起，即JIT(just-in-time)更新。而后更新所有任务的idx，即任务的执行顺序
        # 从 TASK 中筛选出所有 type='FWD' 的task
        fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='FWD')
        # 根据task的layers属性进行排序，原理为取layers这个列表的第一个值
        fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='layers')
        bwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='BWD')
        # 对BWD vt，按照layers[0]即第一层进行降序排序
        bwd_tasks = sort_tasks_by_attr(bwd_tasks, attr='layers', reverse=True)
        upd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='UPD')
        upd_tasks = sort_tasks_by_attr(upd_tasks, attr='layers', reverse=True)
        TASKS = []
        for vt in fwd_tasks:
            TASKS.append(vt)
        # 将反向任务和其对应的更新任务放在一起
        for vt1, vt2 in zip(bwd_tasks, upd_tasks):
            TASKS.append(vt1) 
            TASKS.append(vt2)
        # TASKS中各vtask的id即其自身在TASKS中的排序
        for i, vt in enumerate(TASKS):
            vt.idx = i
        if verbose: print_tasks(TASKS, "ordered tasks (jit update)")

        # ----- place/bind tasks in round-robin -----
        # 3.使用round-robin方法，将task分配到GPU上。换句话说，就是给每个task的device属性赋值

        # ascending round-robin for both fwd and bwd
        # 按照round-robin的方法，将task绑定到GPU上
        nfwd = len(fwd_tasks)
        print("\n.........nfwd:", nfwd)
        for vt in TASKS: # ordered by idx
            if vt.type == 'FWD':
                # 0
                # 1
                # 2
                vt.device = "GPU:%d" % ( vt.idx % N )
            elif vt.type == 'BWD':
                # GPU的id会接着FWD的编号继续
                # 📌除2的意思：以第3行为例，7-3=4，即FWD后已有4个任务。一个BWD和一个UDP是绑定在一起的，因此4/2=2，
                # 即看作执行了2个任务，即已分配了2个GPU。那么idx=7的这个任务，顺序来看就是第5个任务
                # (3+(3-3)/2)%4 = 3
                # (3+(5-3)/2)%4 = 0
                # (3+(7-3)/2)%4 = 1
                vt.device = "GPU:%d" % ( int(nfwd + (vt.idx-nfwd)/2) % N )
            elif vt.type == 'UPD':
                if opt_offld:
                    # 即跟前一个BWD所在的GPU相同
                    # (3+(4-1-3)/2)%4 = 3，-1即将该任务看作是前一个BWD任务，以获得相同的CPU序号
                    # (3+(6-1-3)/2)%4 = 0
                    vt.device = "CPU:%d" % ( int(nfwd + (vt.idx-1-nfwd)/2) % N )
                else:
                    vt.device = "GPU:%d" % ( int(nfwd + (vt.idx-1-nfwd)/2) % N )
        if verbose: print_tasks(TASKS, "placed/bind tasks (round-robin fwd and bwd)")
        # 已被废弃，不用管
        if reverse_bwd: # fwd: round-robin + bwd: reverse round-robin for jit bwd 
            for vt in TASKS: # ordered by idx
                if vt.type == 'BWD':
                    vt.set_new_rank( int(nfwd - (vt.idx-nfwd)/2) % N )
                    assert 0 <= vt.rank and vt.rank < N
                elif vt.type == 'UPD':
                    vt.set_new_rank( int(nfwd - (vt.idx-1-nfwd)/2) % N )
                    assert 0 <= vt.rank and vt.rank < N
            if verbose: print_tasks(TASKS, "placed/bind tasks (with reverse_bwd)")

        # ----- setup tasks' data dependency with p2p -----
        # 4.设置每一个task的数据依赖，即根据task和其前后task是否在一个GPU上，设置其输入输出该通过什么途径传递。
        #   换句话说，就是为task的In和Out属性赋值，这俩都是字典，形式为{'X' : {l:Medium()}, ...}。
        #   前向：每个task的输入(传进来的输入X、参数W、Buffer B)，输出(输出的Y、X、、W、B)
        #   后向：输入(dY, InputX（首个BWD任务）/StashX W, B, T )，输出(dX, dW, W, B, L1)
        #   更新：输入(dW，W，K)，输出(W，K)
        fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='FWD')
        bwd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='BWD')
        # 深拷贝一份tasks，但所有task的vlayers属性只保留第一个值（该Pack的首层）
        bwd_tasks_1st_layer = leave_first_layer_in_each_task(bwd_tasks)
        for task in bwd_tasks_1st_layer:
            print(f"............bwd_tasks_1st_layer:{task.layers[0]}")
        upd_tasks = filter_tasks_by_attr_val(TASKS, attr='type', value='UPD')
        # 为layers属性中包含第0层的vt的has_data属性赋值为true
        # 为layers属性中包含最后一层的vt的has_criterion属性赋值为true
        set_has_data_criterion(TASKS, R) # R：模型层数
        # 设置最后一个fwd任务的is_last_fwd属性为true
        set_the_last_fwd(TASKS)
        for vt in TASKS: # ordered by idx
            print(f"\t.......vt.idx:{vt.idx}, vt.layers{vt.layers}.......")
            vt.layers = sorted(vt.layers)
            # 配置前向任务的输入输出依赖
            # 📌总结：
            # 1.BWD与FWD任务的共性：W和B的输入媒介都为SHM；
            #   BWD与FWD任务的区别：FWD任务的W、B输出媒介为空，但BWD任务的 B 的输出媒介是SHM
            # 2.BWD dW的输出媒介为LOC，UDP dW的输入媒介为LOC
            # 3.对于包含计算层的BWD任务(同时也是第1个BWD任务)，其第一层的输入X的媒介为P2P，即其前一个FWD任务最终的输出
            #   而对于其他BWD任务，其第一层输入X的媒介为MSG，对应了该BWD任务的FWD任务 第一层的X的输出媒介
            #   📌分析：task第一层的输入 X 即整个task的输入，这个输入在反向时还会用到一次，因此在FWD的Out中和BWD的In中呈现
            #           出了一种对应关系
            # 4.对于包含计算层的BWD任务(同时也是第1个BWD任务)，其In中，dY为空。
            #   对于其他BWD任务。其In中最后一层dY的输入媒介为P2P，即前一个BWD任务的第一层的输出。对应了其前一个BWD任务Out中，
            #   最后一层dX的输出媒介为P2P。
            # 5.
            if vt.type == 'FWD':
                # In { 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() } }
                # 若当前任务包含第0层，则vt的输入媒介为DAT
                if vt.has_data: # vt.idx == 0:
                    vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                    # # for DAT's dst T tasks
                    # T_tasks = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)
                    # vt.In['X'][vt.layers[0]].set_for_T(T_tasks)

                # 若不包含第0层（即不是第一个FWD任务），且当前task和前一个task不在同一个rank上，则输入媒介为P2P
                elif TASKS[vt.idx-1].rank != vt.rank:
                    # 
                    vt.In['X'] = ODict({vt.layers[0]:Medium('P2P',vt.idx-1,TASKS)})

                # 若既不是第一个FWD任务，同时还和前一个task在同一个rank上，则输入媒介为SWP
                else: # swap locally
                    vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                vt.In['W'] = ODict()
                vt.In['B'] = ODict()

                # 对当前任务layers属性中的每一层，其参数和buffer的 In 媒介为SHM
                for l in vt.layers:
                    vt.In['W'][l] = Medium("SHM")
                    vt.In['B'][l] = Medium("SHM")

                # Out { 'Y': { L1:Medium() }, 'X': { L1:Medium() }, 'W': {}, 'B': {} }

                # 若下一个task和当前task不在一个rank(GPU)上，则vt最后一层的输出媒介为P2P
                if TASKS[vt.idx+1].rank != vt.rank:
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('P2P',vt.idx+1,TASKS)})
                # 否则，说明下一个task和当前task在一个rank上，输出媒介为SWP
                else: # swap locally
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx+1)})
                vt.Out['X'] = ODict()
                # 遍历fwd vt的每一层，若有BWD vt的首层和该层相同，把这个BWD vt加入到found字典中
                found = find_dependent_tasks_for_layers(bwd_tasks_1st_layer, vt.layers)
                print(f"...FWD... found bwd task id:{found}")
                # l:该vt的首层，dst_idx:bwd task的idx属性
                # 该FWD vt的首层输出媒介设置为MSG，Medium的rank(就是GPU序号)属性设置为目标任务（dst_idx）的rank
                # ❓为何第一层的输入X要保存到Out中呢
                # 答：见上面第3点的分析
                for l, dst_idx in found.items():
                    vt.Out['X'][l] = Medium("MSG",dst_idx,TASKS)
                vt.Out['W'] = ODict()
                vt.Out['B'] = ODict()
            elif vt.type == 'BWD':
                # 若当前task包含计算loss层
                if vt.has_criterion:
                    # In { 'dY':{}, 'InputX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {LLast:Medium()} }
                    vt.In['dY'] = ODict()
                    if vt.has_data: # a single BWD task
                        vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                        # # for DAT's dst T tasks
                        # vt.In['X'][vt.layers[0]].set_for_T([vt])
                    # 若当前vt的前一个任务和自己不在一个rank上，vt的输入媒介为P2P
                    elif TASKS[vt.idx-1].rank != vt.rank:
                        vt.In['X'] = ODict({vt.layers[0]:Medium('P2P',vt.idx-1,TASKS)})
                    # 否则，说明在一个rank上，输入媒介为SWP
                    else: # swap locally
                        vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})

                # 若当前task不是第一个BWD任务，即包含最后一层的BWD任务
                else:
                    # In { 'dY': { L1:Medium() }, 'StashX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {} }
                    # 若当前task相邻的前一个BWD任务(-1是前一个UDP任务，-2就是前一个BWD任务)和自己不在一个rank上，输入媒体为P2P
                    if TASKS[vt.idx-2].rank != vt.rank:
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('P2P',vt.idx-2,TASKS)})
                    # 否则，说明相邻任务和自己在一个rank上，输入媒介为SWP
                    else: # swap locally
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx-2)})
                    vt.In['X'] = ODict()
                    # 就是把对应当前bwd task的fwd task取出来
                    # 对vt.layers中的每一层执行：尝试从fwd tasks中取出layers属性含有当前这一层的task，若存在这种task，
                    # 建立一个字典{l: 该task的task_idx}
                    found = find_dependent_tasks_for_layers(fwd_tasks, [vt.layers[0]])
                    print(f"...BWD... found fwd task id:{found}")
                    # 该BWD vt的首层媒体设置为MSG，Medium的rank属性设置为其源任务（src_idx）的rank
                    for l, src_idx in found.items():
                        vt.In['X'][l] = Medium("MSG",src_idx,TASKS)
                    print(f"找到BWD任务的源任务后, vt.In['X'][l]:{vt.In['X'][l]}")
                vt.In['W'] = ODict()
                vt.In['B'] = ODict()

                # 对layers属性中的每一层，其参数和buffer的 In 媒介为SHM
                for l in vt.layers:
                    vt.In['W'][l] = Medium("SHM")
                    vt.In['B'][l] = Medium("SHM")
                vt.In['T'] = ODict()
                # 若是最后一个计算层，则该任务最后一层的 target 的输入媒介为 DAT
                if vt.has_criterion:
                    vt.In['T'][R-1] = Medium('DAT')
                #     D_task = filter_tasks_by_attr_val(fwd_tasks, attr='has_data', value=True)[0]
                #     vt.In['T'][R-1] = Medium('DAT',D_task.idx,TASKS)

                # Out { 'dX': { L0:Medium() }, 'dW': { L0:Medium(), L1:Medium() }, 'W': {}, 'B': { L0:[Medium(),Medium()], L1:Medium() } } 

                if vt.has_data:
                    vt.Out['dX'] = ODict()

                # 当前BWD任务首层输出的梯度，会通过P2P传给（计算图中）前面一个BWD任务
                elif TASKS[vt.idx+2].rank != vt.rank:
                    vt.Out['dX'] = ODict({vt.layers[0]:Medium('P2P',vt.idx+2,TASKS)})
                else: # swap locally
                    vt.Out['dX'] = ODict({vt.layers[0]:Medium('SWP',vt.idx+2)})
                vt.Out['dW'] = ODict()
                vt.Out['W'] = ODict()
                vt.Out['B'] = ODict()
                # 参数的梯度的输出媒介为 LOC
                for l in vt.layers:
                    # 若执行优化器offload(其实这个工作只支持这个)
                    if opt_offld:
                        vt.Out['dW'][l] = Medium("LOC")
                    else:
                        vt.Out['dW'][l] = Medium("PIN",vt.idx+1)
                        vt.Out['W'][l] = Medium("PIN",vt.idx+1)
                    vt.Out['B'][l] = Medium("SHM")

            elif vt.type == 'UPD':
                # In { 'dW': { L0:Medium(), L1:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                vt.In['dW'] = ODict()
                vt.In['W'] = ODict()
                vt.In['K'] = ODict()
                # dW的输入媒介为LOC，W的输入媒介为SHM
                for l in vt.layers:
                    if opt_offld:
                        vt.In['dW'][l] = Medium("LOC")
                        vt.In['W'][l] = Medium("SHM")
                    else:
                        vt.In['dW'][l] = Medium("PIN",vt.idx-1)
                        vt.In['W'][l] = Medium("PIN",vt.idx-1)
                    vt.In['K'][l] = Medium("SHM")
                # Out { 'W': { L0:[Medium(),Medium()], L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                vt.Out['W'] = ODict()
                vt.Out['K'] = ODict()
                # W和K的输出媒介都是SHM
                for l in vt.layers:
                    vt.Out['W'][l] = Medium("SHM")
                    vt.Out['K'][l] = Medium("SHM")
            else:
                raise ValueError
        if verbose: print_tasks(TASKS, "setup data dependency") 

        # patch FWD-BWD JIT (PIN {W,B})
        # 5.若某些层的前向后向任务在一个GPU上，则这俩任务为可立即执行的pack。将这俩任务 W 和 B 连接的媒介设置为PIN(fwd的Out，
        #   bwd的In)，即前向后 W B 不移动，反向直接用

        # 遍历每一个GPU('GPU:%d')，取出其上的所有任务。若当前设备上既有FWD任务也有BWD任务：取出最后一个FWD任务
        # 和第一个BWD任务，对最后一个FWD任务中的每一层，若该层也存在于第一个BWD任务中，说明可被打包，即以just-in-time的方式执行
        # 📌分析：即该GPU上这些层在执行完FWD后不要卸载，留在GPU上。这样BWD时直接就能用了
        found = find_jit_fwdbwd_pairs_in_one_iter(TASKS, N) # { 'GPU:0': [fwd_idx,[L1,L2],bwd_idx], ... }  
        global MediumMustMatchIdxRank, MediumMustMatchIdx
        MediumMustMatch = MediumMustMatchIdxRank+MediumMustMatchIdx

        # 设置所有GPU上的JIT前后向任务的 W、B 的前向任务的输出、反向任务的输入媒介为PIN
        for _, value in found.items():
            fwd_vt, bwd_vt = TASKS[value[0]], TASKS[value[2]] # ordered by idx
            # 对于JIT pack中的每一层，将其前向任务/后向任务的 W、B 的 输出/输入 媒介都设置为PIN
            for l in value[1]:
                # if: both fwd_vt and bwd_vt don't care MediumMatch
                # (elif: either fwd_vt or bwd_vt cares MediumMatch, but fwd_vt.medium is paired with bwd_vt MediumMatch
                #   then pin {W,B} from fwd_vt to bwd_vt)
                # else:
                #   raise error
                for key in ['W','B']:
                    # 若 当前层l 不在jit fwd任务的Out['W']中，或者 fwd_vt.Out[key][l].medium不存在于MediumMustMatch中，
                    # 并且，当前层l不在jit bwd任务的 In['W']中，或bwd_vt.In[key][l].medium不存在于MediumMustMatch中
                    if (not (l in fwd_vt.Out[key]) or not (fwd_vt.Out[key][l].medium in MediumMustMatch)) and (not (l in bwd_vt.In[key]) or not (bwd_vt.In[key][l].medium in MediumMustMatch)):
                        # 前向任务在该层的Out依赖，其媒介为PIN
                        fwd_vt.Out[key][l] = Medium("PIN",bwd_vt.idx)
                        # 反向任务在该层的In依赖，其媒介为PIN
                        bwd_vt.In[key][l] = Medium("PIN",fwd_vt.idx)
                    else:
                        raise ValueError("Underdevelopment") 

        # patch replacing Last FWD's P2P(Y) to MSG
        # 脚本中未设置该参数，略
        # 24/10/9:
        # 在search.py中，若前后向microbatch大小不同，该参数会被自动置为true，无需在脚本中手动给定参数
        if last_fwd_msg:
            last_fwd_tasks = filter_tasks_by_attr_val(TASKS, attr='is_last_fwd', value=True)
            if last_fwd_tasks == []: # a single BWD task
                CONFIGS["last_fwd_msg"] = False
            else:
                last_fwd_task = last_fwd_tasks[0]
                last_fwd_task = TASKS[last_fwd_task.idx] # ordered
                first_bwd_task = filter_tasks_by_attr_val(TASKS, attr='has_criterion', value=True)
                first_bwd_task = filter_tasks_by_attr_val(first_bwd_task, attr='type', value='BWD')[0]
                first_bwd_task = TASKS[first_bwd_task.idx] # ordered
                # replace the last fwd
                l = last_fwd_task.layers[-1]
                if last_fwd_task.Out['Y'][l].medium == 'P2P':
                    last_fwd_task.Out['Y'][l] = Medium("MSG", first_bwd_task.idx, TASKS)
                # replace the first bwd
                l = first_bwd_task.layers[0]
                if first_bwd_task.In['X'][l].medium == 'P2P':
                    first_bwd_task.In['X'][l] = Medium("MSG", last_fwd_task.idx, TASKS)
    elif mode == 'vDP':
        ubatchszs_fwd = []
        ubatchszs_bwd = []
        for n in range(N):
            per_gpu_tasks = [] # == per_rank_tasks
            # ----- find per-GPU microbatch sizes -----
            DD = int(float(D)/N)
            if D%N != 0: # uneven batch size across GPUs
                if n < D%N:
                    DD += 1
            ubszs_fwd = split_minibatchsize(DD, u_fwd)
            ubszs_bwd = split_minibatchsize(DD, u_bwd)
            ubatchszs_fwd.append(ubszs_fwd)
            ubatchszs_bwd.append(ubszs_bwd)
            if verbose: print("[GPU#{}] ubszs_fwd={}, ubszs_bwd={}".format(n, ubszs_fwd, ubszs_bwd))
            # ----- create tasks from sized data and packed layers -----
            for a_pack in pack_fwd:
                vt = vTask( layers = a_pack, 
                            type = 'FWD', 
                            ubatchszs = ubszs_fwd )
                per_gpu_tasks.append(vt)
            for a_pack in pack_bwd:
                vt = vTask( layers=a_pack, 
                            type='BWD',
                            ubatchszs = ubszs_bwd )
                per_gpu_tasks.append(vt)
            for a_pack in pack_bwd:
                vt = vTask( layers=a_pack, 
                            type='UPD' )
                per_gpu_tasks.append(vt)
            if verbose: print_tasks(per_gpu_tasks, "created tasks (sized data & packed layers) on GPU:%d"%n)
            # ----- order tasks with jit update -----
            fwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='FWD')
            fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='layers')
            bwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='BWD')
            bwd_tasks = sort_tasks_by_attr(bwd_tasks, attr='layers', reverse=True)
            upd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='UPD')
            upd_tasks = sort_tasks_by_attr(upd_tasks, attr='layers', reverse=True)
            per_gpu_tasks = []
            for vt in fwd_tasks:
                per_gpu_tasks.append(vt)
            for vt1, vt2 in zip(bwd_tasks, upd_tasks):
                per_gpu_tasks.append(vt1) 
                per_gpu_tasks.append(vt2)
            for i, vt in enumerate(per_gpu_tasks):
                vt.idx = len(TASKS) + i # n*len(per_gpu_tasks) + i
            if verbose: print_tasks(per_gpu_tasks, "ordered tasks (jit update) on GPU:%d"%n)
            # ----- place/bind tasks -----
            for vt in per_gpu_tasks:
                vt.device = "GPU:%d" % n
                if vt.type == 'UPD' and opt_offld:
                    vt.device = "CPU:%d" % n
            if verbose: print_tasks(per_gpu_tasks, "placed/bind tasks on GPU:%d"%n)
            # ----- setup tasks' data dependency -----
            fwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='FWD')
            bwd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='BWD')
            bwd_tasks_1st_layer = leave_first_layer_in_each_task(bwd_tasks)
            upd_tasks = filter_tasks_by_attr_val(per_gpu_tasks, attr='type', value='UPD')
            set_has_data_criterion(per_gpu_tasks, R)
            set_the_last_fwd(per_gpu_tasks)
            for vt in per_gpu_tasks: # ordered by idx
                vt.layers = sorted(vt.layers)
                if vt.type == 'FWD':
                    # In { 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() } }
                    if vt.has_data: # vt.idx == 0:
                        vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                        # # for DAT's dst T tasks
                        # T_tasks = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)
                        # vt.In['X'][vt.layers[0]].set_for_T(T_tasks)
                    else: # swap locally
                        vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                    vt.In['W'] = ODict()
                    vt.In['B'] = ODict()
                    for l in vt.layers:
                        vt.In['W'][l] = Medium("SHM")
                        vt.In['B'][l] = Medium("SHM")
                    # Out { 'Y': { L1:Medium() }, 'X': { L1:Medium() }, 'W': {}, 'B': {} }
                    # swap locally
                    vt.Out['Y'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx+1)})
                    vt.Out['X'] = ODict()
                    found = find_dependent_tasks_for_layers(bwd_tasks_1st_layer, vt.layers)
                    for l, dst_idx in found.items():
                        vt.Out['X'][l] = Medium("MSG",dst_idx,per_gpu_tasks)
                    vt.Out['W'] = ODict()
                    vt.Out['B'] = ODict()
                elif vt.type == 'BWD':
                    if vt.has_criterion:
                        # In { 'dY':{}, 'InputX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {LLast:Medium()} }
                        vt.In['dY'] = ODict()
                        if vt.has_data: # a single BWD task
                            vt.In['X'] = ODict({vt.layers[0]:Medium('DAT')})
                            # # for DAT's dst T tasks
                            # vt.In['X'][vt.layers[0]].set_for_T([vt])
                        else: # regular case
                            # swap locally
                            vt.In['X'] = ODict({vt.layers[0]:Medium('SWP',vt.idx-1)})
                    else:
                        # In { 'dY': { L1:Medium() }, 'StashX': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {} }
                        # swap locally
                        vt.In['dY'] = ODict({vt.layers[-1]:Medium('SWP',vt.idx-2)})
                        vt.In['X'] = ODict()
                        found = find_dependent_tasks_for_layers(fwd_tasks, [vt.layers[0]])
                        for l, src_idx in found.items():
                            vt.In['X'][l] = Medium("MSG",src_idx,per_gpu_tasks)
                    vt.In['W'] = ODict()
                    vt.In['B'] = ODict()
                    for l in vt.layers:
                        vt.In['W'][l] = Medium("SHM")
                        vt.In['B'][l] = Medium("SHM")
                    vt.In['T'] = ODict()
                    if vt.has_criterion:
                        vt.In['T'][R-1] = Medium('DAT')
                        # if vt.has_data: # a single BWD task
                        #     vt.In['T'][R-1] = Medium('DAT',vt.idx,per_gpu_tasks)
                        # else: # regular case
                        #     D_task = filter_tasks_by_attr_val(fwd_tasks, attr='has_data', value=True)[0]
                        #     vt.In['T'][R-1] = Medium('DAT',D_task.idx,per_gpu_tasks)
                    # Out { 'dX': { L0:Medium() }, 'dW': { L0:Medium(), L1:Medium() }, 'W': {}, 'B': { L0:[Medium(),Medium()], L1:Medium() } }
                    if vt.has_data:
                        vt.Out['dX'] = ODict()
                    else: # swap locally
                        vt.Out['dX'] = ODict({vt.layers[0]:Medium('SWP',vt.idx+2)})
                    vt.Out['dW'] = ODict()
                    vt.Out['W'] = ODict()
                    vt.Out['B'] = ODict()
                    for l in vt.layers:
                        if opt_offld:
                            vt.Out['dW'][l] = Medium("LOC")
                        else:
                            vt.Out['dW'][l] = Medium("PIN",vt.idx+1)
                            vt.Out['W'][l] = Medium("PIN",vt.idx+1)
                        vt.Out['B'][l] = Medium("SHM")
                elif vt.type == 'UPD':
                    # In { 'dW': { L0:Medium(), L1:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                    vt.In['dW'] = ODict()
                    vt.In['W'] = ODict()
                    vt.In['K'] = ODict()
                    for l in vt.layers:
                        if opt_offld:
                            vt.In['dW'][l] = Medium("LOC")
                            vt.In['W'][l] = Medium("SHM")
                        else:
                            vt.In['dW'][l] = Medium("PIN",vt.idx-1)
                            vt.In['W'][l] = Medium("PIN",vt.idx-1)
                        vt.In['K'][l] = Medium("SHM")
                    # Out { 'W': { L0:[Medium(),Medium()], L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
                    vt.Out['W'] = ODict()
                    vt.Out['K'] = ODict()
                    for l in vt.layers:
                        vt.Out['W'][l] = Medium("SHM")
                        vt.Out['K'][l] = Medium("SHM")
                else:
                    raise ValueError
            if verbose: print_tasks(per_gpu_tasks, "set data dependency on GPU:%d"%n)
            # ----- save per GPU tasks -----
            TASKS += per_gpu_tasks     
    else:
        raise ValueError("Underdevelopment")
    
    # check results
    for i, vt in enumerate(TASKS): # must be ordered by idx globally
        assert i == vt.idx
    if verbose: print_tasks(TASKS, "final tasks")

    # add extra configs
    CONFIGS['ubatchszs_fwd'] = ubatchszs_fwd
    CONFIGS['ubatchszs_bwd'] = ubatchszs_bwd
    # 筛选出(type为反向传播)包含计算loss层的 task
    loss_tasks = filter_tasks_by_attr_val(TASKS, attr='has_criterion', value=True)
    loss_tasks = filter_tasks_by_attr_val(loss_tasks, attr='type', value='BWD')
    # 将 loss_rank 设置为 上面提取出的任务所在的rank（就是GPU序号）
    CONFIGS['loss_rank'] = loss_tasks[0].rank
    if verbose: print("added CONFIGS[loss_rank]={}".format(CONFIGS['loss_rank']))
    
    return TASKS

def verify_basics(tasks, configs, verbose=False):
    assert isinstance(tasks, list)
    # Layers are in ascending order
    for vt in tasks:
        assert vt.layers == sorted(vt.layers, reverse=False)
    if verbose: print("Verified: layer ascending order.")
    # All packed layers == model (no missing, no double count, no extra layers)
    layers_correct = list(range(configs['R']))
    tasks_verify = filter_tasks_by_attr_val(tasks, attr='type', value='FWD')
    layers_verify = []
    for vt in tasks_verify:
        layers_verify += vt.layers
    bwd_tasks = filter_tasks_by_attr_val(tasks, attr='type', value='BWD')
    last_bwd_task = filter_tasks_by_attr_val(bwd_tasks, attr='has_criterion', value=True)[0]
    layers_verify += last_bwd_task.layers
    layers_verify.sort(reverse=False)
    assert layers_verify == layers_correct
    for t in ['BWD','UPD']:
        tasks_verify = filter_tasks_by_attr_val(tasks, attr='type', value=t)
        layers_verify = []
        for vt in tasks_verify:
            layers_verify += vt.layers
        layers_verify.sort(reverse=False)
        assert layers_verify == layers_correct
    if verbose: print("Verified: layers matching model.")
    # Within a task, In layers == Compute layers == Out layers
    for vt in tasks:
        if vt.type == 'FWD':
            assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
            assert list(vt.In['W'].keys()) == vt.layers
            assert list(vt.In['B'].keys()) == vt.layers
            assert vt.layers[-1] in vt.Out['Y'] and len(vt.Out['Y'])==1
            for l in vt.Out['X'].keys(): # Empty layer skip
                assert l in vt.layers
            for l in vt.Out['W'].keys(): # Empty layer skip
                assert l in vt.layers
            for l in vt.Out['B'].keys(): # Empty layer skip
                assert l in vt.layers
        elif vt.type == 'BWD':
            # In
            if vt.has_criterion:
                assert not vt.In['dY']
                assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
                assert list(vt.In['W'].keys()) == vt.layers
                assert list(vt.In['B'].keys()) == vt.layers
                assert vt.layers[-1]==configs['R']-1 and vt.layers[-1] in vt.In['T'] and len(vt.In['T'])==1
            else:
                assert vt.layers[-1] in vt.In['dY'] and len(vt.In['dY'])==1 
                assert vt.layers[0] in vt.In['X'] and len(vt.In['X'])==1 
                assert list(vt.In['W'].keys()) == vt.layers
                assert list(vt.In['B'].keys()) == vt.layers
                assert not vt.In['T']
            # Out
            if 0 in vt.layers:
                assert not vt.Out['dX'] # Empty Dict/OrderedDict is False
            else:
                assert vt.layers[0] in vt.Out['dX'] and len(vt.Out['dX'])==1 
            assert list(vt.Out['dW'].keys()) == vt.layers
            if configs['opt_offld']:
                assert not vt.Out['W'] # Empty Dict/OrderedDict is False
            else:
                assert list(vt.Out['W'].keys()) == vt.layers
            assert list(vt.Out['B'].keys()) == vt.layers
        elif vt.type == 'UPD':
            assert list(vt.In['dW'].keys()) == vt.layers
            assert list(vt.In['W'].keys()) == vt.layers
            assert list(vt.In['K'].keys()) == vt.layers
            assert list(vt.Out['W'].keys()) == vt.layers
            assert list(vt.Out['K'].keys()) == vt.layers
    if verbose: print("Verified: In layers == Compute layers == Out layers.")
    # In: each layer has single medium()
    # Out: each layer has single medium(), except {B,W} can have a list of mediums
    list_medium_allowed_in_Out = ["B","W"]
    for vt in tasks:
        for key, lm in vt.In.items():
            for l, m in lm.items():
                assert isinstance(m, Medium)
        for key, lm in vt.Out.items():
            for l, m in lm.items():
                if key in list_medium_allowed_in_Out:
                    if isinstance(m, list):
                        for mm in m:
                            assert isinstance(mm, Medium)    
                    else:
                        assert isinstance(m, Medium)
                else:
                    assert isinstance(m, Medium)
    if verbose: print("Verified: single and list Medium().")
    # Among tasks, Out medium() must be paired with In medium(), expect [DAT,SHM,LOC]
    # 1) selected_medium = ["P2P","MSG","SWP","PIN"]
    # 2) create a copy of tasks
    # 3) traverse original dtask
    #       traverse vt.Out
    #           traverse each layer
    #               if medium in selected_medium:
    #                   match medium to In
    #                   del the medium in both In and Out in copy tasks
    # 4) confirm no selected_medium left in copy tasks
    global MediumMustMatchIdxRank, MediumMustMatchIdx
    dtasks_copy = make_dtasks(deepcopy(tasks))
    for Out_vt in tasks:
        for Out_key, lm in Out_vt.Out.items():
            for Out_l, m in lm.items():
                Out_ms = [m] if isinstance(m,Medium) else list(m)
                for i, Out_m in enumerate(Out_ms):
                    if Out_m() in MediumMustMatchIdxRank+MediumMustMatchIdx:
                        In_vt = filter_tasks_by_attr_val(tasks, attr='idx', value=Out_m.idx)[0] # In_vt = dtasks[Out_m.idx]                        
                        if Out_key == 'Y':
                            if Out_l != configs['R']-1:
                                In_key = 'X'
                                In_l = Out_l+1
                            else:
                                In_key = 'dY'
                                In_l = Out_l    
                        elif Out_key == 'X': # stashX
                            In_key = 'X'
                            In_l = Out_l
                        elif Out_key == 'dX':
                            assert Out_l != 0
                            In_key = 'dY'
                            In_l = Out_l-1
                        elif Out_key in ['dW','W','B','K']: # W/B of fwd,bwd,upd
                            In_key = Out_key
                            In_l = Out_l
                        else:
                            raise ValueError("Unknown Out[{}]".format(Out_key))
                        # match Out medium to In medium
                        In_m = In_vt.In[In_key][In_l]
                        Out_m() == In_m()
                        Out_m.idx == In_vt.idx
                        In_m.idx == Out_vt.idx
                        if Out_m() in MediumMustMatchIdxRank:
                            Out_m.rank == In_vt.rank
                            In_m.rank == Out_vt.rank
                        # del the medium in both In and Out in copy tasks
                        dtasks_copy[In_vt.idx].In[In_key][In_l] = Medium()
                        if isinstance(m,Medium):
                            dtasks_copy[Out_vt.idx].Out[Out_key][Out_l] = Medium()
                        elif isinstance(m,list):
                            dtasks_copy[Out_vt.idx].Out[Out_key][Out_l][i] = Medium()
    # if verbose: print_tasks(dtasks_copy, name="dtasks_copy") 
    for match in MediumMustMatchIdxRank+MediumMustMatchIdx:
        filtered_tasks = filter_tasks_by_attr_val(list(dtasks_copy.values()), attr="medium", value=match)
        assert filtered_tasks == [], "'{}':remaining_tasks={}".format(match, filtered_tasks) 
    if verbose: print("Verified: Out Medium() pairing with In Medium().")
    # # T sharing matches (first layer X.DAT -> last layers's BWD) 
    # DAT_tasks = filter_tasks_by_attr_val(tasks, attr="medium", value='DAT')
    # D_tasks = filter_tasks_by_attr_val(DAT_tasks, attr="has_data", value=True)
    # T_tasks = filter_tasks_by_attr_val(DAT_tasks, attr="has_criterion", value=True)
    # for src in D_tasks:
    #     for dst_idx, dst_rank in zip(src.In['X'][0].dst_idx_T, src.In['X'][0].dst_rank_T):
    #         dst_task = filter_tasks_by_attr_val(T_tasks, attr="idx", value=dst_idx)[0]
    #         assert dst_idx == dst_task.idx
    #         assert dst_rank == dst_task.rank
    #         assert dst_task.In['T'][configs['R']-1].idx == src.idx
    #         assert dst_task.In['T'][configs['R']-1].rank == src.rank
    #         # del the matched dst_task
    #         del_i = None
    #         for i, vt in enumerate(T_tasks):
    #             if vt.idx == dst_task.idx:
    #                 del_i = i
    #                 break
    #         T_tasks.pop(del_i)
    # assert len(T_tasks) == 0
    # if verbose: print("Verified: T sharing matches.")

def verify_scheduled(rtasks, configs, verbose=False):
    """ Verify correctness of tasks before serving Runtime. """
    if verbose: print("\n----- Verification -----")
    if configs['mode'] == 'vPP':
        tasks = unmake_rank_based_task_queue(rtasks)
        verify_basics(tasks, configs)
    elif configs['mode'] == 'vDP':
        for r in range(configs['N']):
            # verfiy per gpu tasks
            verify_basics(rtasks[r], configs)
        # verify D == summed ubatchsizes of all GPU 
        Dfwd, Dbwd = 0, 0
        for r in range(configs['N']):
            fwd_tasks = filter_tasks_by_attr_val(rtasks[r], attr='type', value='FWD')
            bwd_tasks = filter_tasks_by_attr_val(rtasks[r], attr='type', value='BWD')
            Dfwd += sum(fwd_tasks[0].ubatchszs) if fwd_tasks != [] \
                    else sum(bwd_tasks[0].ubatchszs) # a single BWD task
            Dbwd += sum(bwd_tasks[0].ubatchszs)
        assert Dfwd == configs['D']
        assert Dbwd == configs['D']
    else:
        pass
    if verbose: print("Verification Succeeded.\n")

def verify_layer_packs(pack_fwd, pack_bwd, num_layers):
    assert isinstance(pack_fwd, list) and isinstance(pack_bwd, list)
    # All packed layers = model (no missing, no double count, no extra layers, ascend)
    layers_correct = list(range(num_layers))
    # check BWD first
    layers_bwd = []
    for p in pack_bwd:
        layers_bwd += p
    # layers_bwd.sort(reverse=False)
    assert layers_bwd == layers_correct
    # check FWD then
    layers_fwd = []
    for p in pack_fwd:
        layers_fwd += p
    layers_fwd += pack_bwd[-1]
    # layers_fwd.sort(reverse=False)
    assert layers_fwd == layers_correct    

# 
def compose(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verify=True, verbose=False):
    """ top-level function """
    """ generate a task graph from given four-tuple configuration """

    if verify: verify_layer_packs(pack_fwd, pack_bwd, args.num_layers)
    # 1.创建并返回一个字典，里面全是配置信息
    CONFIGS = compose_task_configs(args, u_fwd, pack_fwd, u_bwd, pack_bwd, verbose)
    # 1.为fwd layer packs的每个layer pack生成一个类型为FWD的 vTask 的实例；为bwd layer packs的每个pack生成一个类型
    #   为BWD的 vTask实例，和一个类型为UDP的实例
    # 2.将反向任务和其对应的更新任务放在一起，即JIT(just-in-time)更新。而后更新所有任务的idx，即任务的执行顺序
    #   从 TASK 中筛选出所有 type='FWD' 的task
    # 3.使用round-robin方法，将task分配到GPU上。换句话说，就是给每个task的device属性赋值
    # 4.设置每一个task的数据依赖，即根据task和其前后task是否在一个GPU上，设置其输入输出该通过什么途径传递。
    #   换句话说，就是为task的In和Out属性赋值，这俩都是字典，形式为{'X' : {l:Medium()}, ...}。
    #   前向：每个task的输入(传进来的输入X、参数W、Buffer B)，输出(输出的Y、X、、W、B)
    #   后向：输入(dY, InputX（首个BWD任务）/StashX W, B, T )，输出(dX, dW, W, B, L1)
    #   更新：输入(dW，W，K)，输出(W，K)
    # 5.若某些层的前向后向任务在一个GPU上，则这俩任务为可立即执行的pack。将这俩任务 W 和 B 连接的媒介设置为PIN(fwd的Out，
    #   bwd的In)，即前向后 W B 不移动，反向直接用
    TASKS = compose_task_graph(CONFIGS, verbose)
    # 将传入的tasks列表中的任务，按照所在的GPU序号进行重组。即建立一个字典，在同一个GPU上的任务组成一个list对应一个GPU序号
    # 对每一个GPU序号执行：
    # 1.将GPU序号相同的task全拿出来
    # 2.将拿出来的task按照自身的idx属性排序
    # 3.建立一个字典 {GPU序号:[该GPU上的所有task]}
    # 返回这个字典
    # { rank0: [task0,task2,task5,...], rank1:... }
    rTASKS = convert_to_per_rank_task_queue(TASKS, args.num_gpus, verbose)
    if verify: verify_scheduled(deepcopy(rTASKS), deepcopy(CONFIGS), verbose)
    
    # 返回配置字典：CONFIGS、任务列表：TASKS、按GPU序号重组的task字典：rTASKS
    return CONFIGS, TASKS, rTASKS
