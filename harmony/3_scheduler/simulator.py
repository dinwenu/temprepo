# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
import time

from task_data_struct import Medium, vTask, unserialize_scheduled, unmake_rank_based_task_queue
from sim_data_struct import *
from sim_chrome_trace import save_to_chrome_trace

def _assert_assumption(CONFIGS):
    assert CONFIGS['opt_offld'] 
    """ - Ignore T """

# 创建vt对应的Event，创建的同时会建立Event和其他Event的依赖关系，即执行上的先后关系。由于是在Event之间建立依赖关系
# 而有些Event还没初始化，所以可能会用字符串来替代真实的Event，后面这些字符串会被替换为真正的Event
def create_a_task_events(mode, num_gpus, vt, events, stream_events, left_vt, left2_vt, ubscvt, TASKS, prefetch_offload):
    if mode == 'vPP':
        # 若vt是前向任务，且不含有loss计算层 
        if vt.type == "FWD" and not vt.has_criterion:
            # 1.实例化一个事件Event，表示一个swapin WB的事件
            # 2.将生成的event添加到event字典 { id（代表一个event的字符串）: Event() } 和对应的事件队列中 {self.stream: [Event()]}
            #   并将event的pos属性(在队列中的下标)，设置为事件队列的最后一个位置
            #   2.1.若输入的task的左边还有task，还会增加一步额外的操作，创建其左边任务的Event的字符串表示(用于创建event实例的5个参数)，
            #       加入到当前event的inputs列表中。表示Compute DEL事件执行完了才能开始执行Swap In WB事件（📌即上一个任务的最后一个事件）
            #
            # 若vt的前一个任务存在（且不prefetch offload）：compute DEL事件（前一个vt的最后一个事件）->SwapIn WB事件
            ev_w = In_WB(vt, events, stream_events, left_vt, False, prefetch_offload)
            # 对vt中的每一个micro batch size
            for i, u in enumerate(vt.ubatchszs):
                # 1.为task的输入X生成一个event
                # 2.针对具体的情况，为event的inputs属性赋值，即表示对应当前输入事件的输出事件
                #
                # 包含输入数据的FWD任务：创建SwapIn X事件，无需注册依赖任务
                # 输入媒介为P2P：创建P2PIn X事件，无需注册以来任务，但需注册对应的peer任务
                # 输入媒介为MSG：创建SwapIn X事件，源任务的最后一个subpack的CPU MSGY事件(在CPU上发送Y)->SwapIn X事件
                # 输入媒介为SWP：创建SwapIn X事件，left vt的Swapout Y事件->SwapIn X事件
                # 此外：
                # i=0：SwapIn WB事件->SwapIn X事件
                # i>0:Compute FWD事件->SwapIn X事件
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, None, None, prefetch_offload, ev_w if i == 0 else ev_fwd)
                # 📌sX是stashX
                # 1.实例化一个前向计算事件，并将生成的event添加到event字典 { id（代表一个event的字符串）: Event() } 和
                #   对应的事件队列中 {self.stream: [Event()]}。并将event的pos属性(在队列中的下标)，设置为事件队列的最后
                #   一个位置
                #   1.1.若micro batch idx = 0，还要将ev_w事件加入到当前事件的inputs属性中，即 ev_w （WB：参数和buffer） 为当前事件的输入依赖
                #   1.2.将ev_x加入当前事件的inputs属性中，即 ev_x （X:激活值） 为当前事件的输入依赖
                # 2.若一个task中有多层的输入都会被反向传播时用到，则从第2个需要的层开始相当于其他的subpack，为其创建新的event，新的event的spi
                #   从1开始累加，即subpack1、subpack2、...
                # 3.为每一个新实例化的subpack fwd event添加一个输入依赖事件，即Swapout掉stashX。
                #   ❓感觉像是写错了，没写错的话这个依赖应该是输出依赖，即fwd event执行完了才能开始swap out掉stashX
                #
                # 在i=0时：SwapIn WB事件->Compute FWD事件
                # 其他所有情况，包括i=0：输入X事件->Compute FWD事件
                # 若该vt除第0层外有其他层要发送StashX：创建Compute FWD事件，
                # （和第一个创建的computeFWD事件spi不同），SwapOut sX事件->ComputeFWD事件
                ev_fwd = Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload)
                # 建立swapout stashX和CPU MSGsX事件，一个vt会建立多个swapout sX事件，首个事件的依赖事件为ev_x，后面的依赖都为
                # 前面Compute FWD事件，即FWD执行完了才能开始卸载，而后建立相应的MSGsX事件，其依赖即为刚建立的swapoot sX事件
                # 对每一个subpack
                # 1.实例化一个swapout sX（ev_sx）事件，并将其加入到两个字典中。
                #   1.1.若当前的subpack是第一个subpack，即subpack是从layer0开始的：将 ev_x（vt的输入X事件） 添加到ev_sx的Inputs属性中，
                #       即ev_x为当前事件的输入依赖事件
                    # 📌分析：是正确的，这个swapout是FWD的首层，根本没开始执行FWD
                #   1.2.若是后面的subpack，添加 前一个subpack的前向计算 事件为ev_sx的需求输入任务。即前一个subpack的前向计算执行完了，才能
                #       开始当前subpack的swapout stashX任务
                # 2.实例化一个CPU上的MSGsX事件，表示CPU上发送StashX的事件
                #   2.1.将最后一个subpack的ev_sx设置为刚刚实例化的事件的需求输入事件，即暂存的X swapout以后CPU才能发送
                #
                # 除vt首个subpack外（首层之前肯定没有要FWD的层啊）：第spi-1个Compute FWD事件->第spi个Swapout sX事件
                # 所有情况：实例化一个CPU MSGsX事件，SwapOut sX事件->CPU MSGsX事件
                ev_msg = Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload)
                # 两种情况：
                # 1.若vt的Out['Y']最后一层的媒介为P2P
                #   1.1.实例化一个P2P事件，其spi为 ev_fwd（fwd计算事件）的spi，并将实例化的事件添加到两个字典中
                #   1.2.将ev_fwd添加到当前这个P2POut Y事件的需求输入列表中，表示前向计算的事件完成后才能开始P2POut
                #   1.3.此外，还要将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
                # 2.若Out['Y']最后一层的媒介为MSG，这种情况为最后一个fwd
                #   2.1.实例化一个Swap out Y事件，其spi为 ev_fwd（fwd计算事件）的spi，并将实例化的事件添加到两个字典中。
                #   2.2.将ev_fwd添加到当前这个Swapout Y事件的需求输入列表中，表示前向计算的事件完成后才能执行CPU上发送Y的事件
                # ❓2.3.实例化一个CPU MSGY事件，可能代表CPU上输出或输入Y的事件。将刚刚实例化的Swap out Y事件添加到MSGY事件的inputs列表中，
                #       表示Swapout Y之后才能执行CPU上MSGY事件
                #
                # 输出媒介为P2P：该vt的最后一个Compute FWD事件(spi值最大的那个)->P2POut Y事件，还需注册peer事件
                # 输出媒介为MSG：该vt的最后一个Compute FWD事件(spi值最大的那个)-SwapOut Y事件
                #               创建CPU MSGY事件，SwapOut Y事件->CPU MSGY事件
                # 输出媒介为SWP：该vt的最后一个Compute FWD事件(spi值最大的那个)-SwapOut Y事件
                ev_y = Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd)
            # events：{id：Event()，...}
            # 实例化一个Compute DEL事件，并将其加入到两个字典中
            #
            # SwapOut dWB事件->Compute DEL事件
            Compute_DEL(vt, events, stream_events, None, prefetch_offload, None)

        # 若vt是后向任务，且含有loss计算层
        elif vt.type == "BWD" and vt.has_criterion:
            # 若vt的前一个任务存在（且不prefetch offload）：compute DEL事件（前一个vt的最后一个事件）->SwapIn WB事件
            ev_w = In_WB(vt, events, stream_events, left_vt, False, prefetch_offload)
            for i, _ in enumerate(vt.ubatchszs):
                # 包含输入数据的FWD任务：创建SwapIn X事件，无需注册依赖任务
                # 输入媒介为P2P：创建P2PIn X事件，无需注册依赖任务，但需注册对应的peer任务
                # 输入媒介为MSG：创建SwapIn X事件，源任务的最后一个subpack的CPU MSGY事件(在CPU上发送Y)->SwapIn X事件
                # 输入媒介为SWP：创建SwapIn X事件，left vt的Swapout Y事件->SwapIn X事件
                # 此外：
                # i=0：SwapIn WB事件->SwapIn X事件
                # i>0:Compute BWD事件->SwapIn X事件
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, None, prefetch_offload, ev_w if i == 0 else ev_bwd)
                # 1.实例化一个Compute REC事件(重计算事件)，并将其加入到两个字典中
                # 2.若当前是首个micro batch，需将ev_w（即SwapIn WB）事件作为输入依赖
                # 3.实例化一个Compute BWD事件，并将其加入到两个字典中
                # 返回两个事件：Cpmpute REC事件、Compute BWD事件
                #
                # 在i=0时：SwapIn WB事件->Compute REC（重计算）事件
                # 其他所有情况，包括i=0：输入X事件->Compute REC（重计算）事件
                # 📌并没有为Compute BWD事件设置其依赖事件
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, None)
                # 两种情况
                # 1.若任务的Out[dX]的首层的媒介为P2P
                #   1.1.实例化一个P2POut dX事件 ev_dx，并加入到两个字典中
                #   1.2.将对应的那个p2p任务以字符串的形式赋给调用该函数的 ev_dx 的peer_id属性
                #   1.3.将 ev_bwd（即第i个 compute BWD事件）作为当前事件的输入依赖
                # 2.若任务的Out[dX]的首层的媒介为SWP，说明为vDP
                #   2.1.实例化一个SwapOut dX任务,并加入到两个字典中
                #   2.2.将 ev_bwd（即第i个 compute BWD事件）作为当前事件的输入依赖
                #
                # Compute BWD事件->P2POut dX/Swap out dX事件
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            # 实例化一个SwapOut dWB事件，将其加入两个字典中，并将最后一反向计算事件作为该事件的输入依赖
            #
            # 最后一个BWD事件->SwapOut dWB事件
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, None, prefetch_offload, None)
            # 实例化一个Compute DEL事件，将其加入到两个字典中，并将SwapOut dWB事件作为该事件的输入依赖
            #
            # SwapOut dWB事件->Compute DEL事件
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)

        # 若vt是后向任务，且不含有loss计算层
        elif vt.type == "BWD" and not vt.has_criterion:
            # 若vt的前一个任务存在（且不prefetch offload）：compute DEL事件（前一个vt的最后一个事件）->SwapIn WB事件
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            # 返回：vt的源task，以及vt的首层在源任务中对应的subpack idx
            src_vt, src_spi = find_stash_subpack_idx(vt, TASKS)
            for i, _ in enumerate(vt.ubatchszs):
                # 1.实例化一个SwapIn sX事件，并加入到两个字典中
                # 2.将CPU MSGsX事件添加到SwapIn sX事件的输入依赖中，表示CPU MSGsX事件完成后才能执行Swap In sX事件
                # 3.若i为0，即执行的是首个micro batch，添加的依赖事件为 Swapin WB
                #   否则，添加的依赖事件为执行前面的micro batch的 Compute BWD 事件，表示前面的BWD算完了才能开始 SwapIn sX事件
                #
                # src_vt的CPU MSGsX事件->SwapIn sX事件
                # 此外：
                # i=0：SwapIn WB事件->SwapIn sX事件
                # i>0:Compute BWD事件->SwapIn sX事件
                ev_sx = In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w if i == 0 else ev_bwd)
                # 1.若task的In['dY']的最后一层媒介为P2P，实例化一个P2PIn dY事件并加入到两个字典中，将对应的那个p2p任务以字符串的形式赋给
                #   调用该函数的event的peer_id属性
                #   vDP：若task的In['dY']的最后一层媒介为SwapIn，...，将其左边task的SwapOut dX事件加入到输入依赖中
                # 2.将 Compute REC 事件加入到P2PIn dY的依赖事件中，表示当前任务的第i个ubatch的反向Compute REC事件完成后，
                #   才能开始当前ubatch上的In dY事件
                #   ❓Compute REC是Compute_BWD函数生成的，为何还没被生成就加入到输入依赖中了？
                #
                # 两种情况：
                # 输入媒介为P2P：无需创建依赖事件，但要创建peer事件
                # 输入媒介为SWP：left vt的SwapOut dX事件->当前vt的SwapIn dY事件
                # 无论何种情况：
                # Compute REC事件->P2PIn dY/SwapIn dY事件
                ev_dy = In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, False, prefetch_offload)
                # 在i=0时：SwapIn WB事件->Compute REC（重计算）事件
                # 其他所有情况，包括i=0：输入stashX事件->Compute REC（重计算）事件
                # 📌P2PIn dY/SwapIn dY事件->Compute BWD事件
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_sx, ev_dy)
                #
                # Compute BWD事件->P2POut dX/SwapOut dX事件
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            #
            # 最后一个BWD事件->SwapOut dWB事件
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, None, prefetch_offload, None)
            #
            # SwapOut dWB事件->Compute DEL事件
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)

        # 若vt是参数更新任务
        elif vt.type == "UPD":
            # 1.实例化以一个CPU Update事件，并将其加入到两个字典中
            # 2.将SwapOut dWB事件作为当前事件的依赖事件
            #
            # left vt的SwapOut dWB事件（left vt就是对应当前UPDvt的BWDvt）->CPU Update事件
            CPU_Update(vt, events, stream_events, left_vt)
        else:
            raise ValueError
    elif mode == 'vDP':
        if vt.type == "FWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            for i, u in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, None, ev_w, prefetch_offload, ev_w if i == 0 else ev_y)
                ev_fwd = Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload)
                ev_msg = Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload)
                ev_y = Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd)
            Compute_DEL(vt, events, stream_events, None, prefetch_offload, ev_y)
        elif vt.type == "BWD" and vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            for i, _ in enumerate(vt.ubatchszs):
                ev_x = In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, ev_w, prefetch_offload, ev_w if i == 0 else ev_dx)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, None)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)
            ev_ard = Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "BWD" and not vt.has_criterion:
            ev_w = In_WB(vt, events, stream_events, left_vt, True, prefetch_offload)
            src_vt, src_spi = find_stash_subpack_idx(vt, TASKS)
            for i, _ in enumerate(vt.ubatchszs):
                ev_sx = In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w if i == 0 else ev_dx)
                ev_dy = In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, True, prefetch_offload)
                ev_rec, ev_bwd = Compute_BWD(vt, i, events, stream_events, ev_w, ev_sx, ev_dy)
                ev_dx = Out_dX(vt, i, events, stream_events, TASKS, ev_bwd)    
            ev_ard = Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx)
            ev_dw = Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx)
            Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, None)
        elif vt.type == "UPD":
            CPU_Update(vt, events, stream_events, left_vt)
        else:
            raise ValueError
    else:
        raise ValueError

def simulate(args, rTASKS, CONFIGS, TASKS=None, prefetch_offload=True, verbose=True, view=True):
    """ top-level function """
    """ estimate runtime of given task graph by an event-driven simulation """

    _assert_assumption(CONFIGS)
    TASKS = unmake_rank_based_task_queue(rTASKS) if TASKS is None else TASKS
    if CONFIGS['mode'] == 'vPP':
        ubatchszs_fwd = CONFIGS['ubatchszs_fwd']
        ubatchszs_bwd = CONFIGS['ubatchszs_bwd']
    elif CONFIGS['mode'] == 'vDP':
        ubatchszs_fwd = CONFIGS['ubatchszs_fwd'][0]
        ubatchszs_bwd = CONFIGS['ubatchszs_bwd'][0]
    else:
        raise ValueError

    # 该例子这俩值一样，执行else：ubscvt = None
    # 
    if CONFIGS["u_fwd"] != CONFIGS["u_bwd"]:
        ubscvt = UBSConverter(ubatchszs_fwd, ubatchszs_bwd, CONFIGS['u_bwd'], verbose)
    else:
        ubscvt = None
        assert ubatchszs_fwd == ubatchszs_bwd
    
    if CONFIGS['mode']=='vPP' and CONFIGS['N']>1:
        sim_mode = 'vPP'
    elif (CONFIGS['mode']=='vPP'and CONFIGS['N'] == 1) or CONFIGS['mode']=='vDP':
        sim_mode = 'vDP' # """ treat as a single GPU """
    else:
        raise NotImplementedError
    
    # 计算 rTASKS 中非空任务列表的数量
    # non_empty_gpus 变量的值表示当前被分配了任务的 GPU 的数量
    non_empty_gpus = sum([tasks != [] for tasks in rTASKS.values()])
    print(f"non_empty_gpus：{non_empty_gpus}")
    res = ODict()

    # ----------------------- Time --------------------------------
    if verbose: t_start = time.time() 
    ### make events with dependency
    events = ODict() # { id: Event() } # vPP for all ranks, vDP for rank0
    rank_stream_events = ODict() # { rank: { stream: [Event()] } or {} } 
    # GPU序号, [tasks]
    # 按照GPU序号的顺序，对每个GPU中的task实例化Event，每一个实例化的事件都会装进events和rank_stream_events这两个字典中
    for rank, tasks in rTASKS.items(): # { rank: [Task()] or [] }
        # 若模式为vDP，只处理rank0上的任务
        if sim_mode == 'vDP' and rank != 0:
            break
        rank_stream_events[rank] = ODict() # { stream: [Event()] } or {}
        left_vt, left2_vt = None, None    
        for vt in tasks:
            create_a_task_events(sim_mode, non_empty_gpus, vt, events, rank_stream_events[rank], left_vt, left2_vt, ubscvt, TASKS, prefetch_offload)
            if vt.type in ['FWD', 'BWD']:
                left2_vt = left_vt
                left_vt = vt
    # 对所有的事件：将其Inputs列表，即输入依赖，中的id（表示Event的字符串）替换为实际的Event实例
    for ev in events.values(): # convert remaining input ids to events
        ev.inputs = [inev if isinstance(inev, Event) else events[inev] 
                        for inev in ev.inputs]
    for ev in events.values(): # add p2p dependency
        # 1.将当前事件对应的P2P事件的所有依赖加入到当前事件的依赖中
        # 2.若对应的P2P事件不是其所在stream中的第1个事件，还需将P2P所在的rank的stream中的前一个事件加入到当前事件的依赖中
        # 3.清空当前事件Inputs中的重复依赖，并按照id进行排序
        ev.solve_peer(events, rank_stream_events)
    if verbose: 
        print_events(events) 
        print_rank_stream_events(rank_stream_events)

    # if sim_mode == 'vPP': debug(non_empty_gpus, events, rank_stream_events)
    ### dispatch events for execution
    dispatcher = Dispatcher(rank_stream_events)
    executor = Executor(args, non_empty_gpus, CONFIGS, TASKS, rank_stream_events)
    if verbose:
        print("=== Dispatching: %d Streams, %d Events ==="%(
                dispatcher.num_streams, len(events.keys()) ))

    while True:
        # 遍历 event_queues 中所有的list，直到某个list的首个event没有输入依赖，返回该event
        # 若不存在这样一个事件，则返回所有stream list的首个event
        # 即从首个不存在依赖的event开始执行
        ev = dispatcher.dispatch()
        if isinstance(ev, Event):
            # 1.拿到ev的起事时间：ev的起始时间为其所有依赖事件的结束时间和ev所在的rank上所在stream的结束时间中的最大值
            # 2.计算ev的持续时间：直接拿到或根据带宽计算事件的持续时间
            # 3.更新ev所在的rank上所在stream的结束时间，即ev的结束时间
            # 4.ev.is_done = True
            # 5.更新该GPU上的总计算时间
            # 6.更新执行过的event数
            executor.execute(ev)
            if verbose: print("Executed: %s"%(ev))
        elif isinstance(ev, list):
            res['global_endtime'] = float('inf')
            res['per_rank_endtime'] = [float('inf')] if sim_mode == 'vDP' else \
                                      [float('inf')] * len(rTASKS.keys())
            res['max_endidle'] = float('inf')
            res['avg_endidle'] = float('inf')
            res['avg_compute_to_globaltime'] = float('inf')
            res['avg_compute_to_ranktime'] = float('inf')
            print("=== Deadlock! after executed %d events. stop here: ==="%(executor.cnt))
            for e in ev: 
                print("\t%s" % (e))
            break

        # 若Dispatcher中的事件队列为空，直接返回done字符串，说明所有事件执行完毕
        elif ev == "done":
            executor.end()
            # 得到全局的endtime，即所有rank的所有stream中最大的结束时间
            res['global_endtime'] = executor.global_endtime
            # 每个rank中所有stream中最大的结束时间
            res['per_rank_endtime'] = executor.per_rank_endtime
            # 得到per_rank_endidle中的最大值
            res['max_endidle'] = executor.max_endidle
            # 计算平均的idle比率，即每个rank上每个流中最大的结束时间后还需等待的时间/global_endtime的均值
            res['avg_endidle'] = executor.avg_endidle
            # 计算每个GPU计算时间占总时间的比例，即每一个GPU上总的计算时间/global_endtime。将这些比率加起来/GPU总数，
            # 得到平均的计算占全局总时间的比例
            res['avg_compute_to_globaltime'] = executor.avg_compute_to_globaltime
            # 计算每个GPU上计算时间与该GPU上总时间的比值，加在一起再除以GPU数量。即平均的计算和单个GPU上的总时间的占比
            res['avg_compute_to_ranktime'] = executor.avg_compute_to_ranktime
            if verbose:
                t_end = time.time() 
                print("=== Simulation Done ===")
                print("Global End Time: %.2f sec (%s)"%(
                        res['global_endtime'], 
                        ", ".join("%.2f"%et for et in res['per_rank_endtime']) ))
                print("Max/Avg End Idle: %.0f%% / %.0f%%"%(
                        res['max_endidle']*100., 
                        res['avg_endidle']*100.))
                print("Compute2Global/Compute2Rank: %.0f%% / %.0f%%"%(
                        res['avg_compute_to_globaltime']*100., 
                        res['avg_compute_to_ranktime']*100.))
                print("Time Cost: %.3f sec (%d Streams, %d Events)"%   
                        (t_end-t_start, dispatcher.num_streams, executor.cnt))
            if view: 
                save_to_chrome_trace(args, CONFIGS, events)
            break
        else:
            raise ValueError
    
    # ----------------------- Memory --------------------------------
    if verbose:
        print("=== Estimating Memory ===")
        t_start = time.time() 
    C_of_task = CofTask(args.prof, sim_mode, CONFIGS['R'])
    # 该列表存放每个rank中所有任务的空间占用的最大值
    per_rank_memory = []
    for rank, tasks in rTASKS.items(): # { rank: [Task()] or [] }
        if sim_mode == 'vDP' and rank != 0:
            break
        if tasks == []:
            max_mem = 0.

        # 
        else:
            # 得到该rank中所有任务的空间占用的最大值
            max_mem = max([C_of_task(vt) for vt in tasks])
        per_rank_memory.append(max_mem) # bytes
    # 所有rank中的所有任务的空间占用的最大值
    res['global_memory'] = max(per_rank_memory)
    # 各个rank中的所有任务的空间占用的最大值
    res['per_rank_memory'] = per_rank_memory
    if verbose:
        t_end = time.time() 
        print("Memory: %.0f MB (%s)"% (
                res['global_memory']/1024./1024., 
                ", ".join("%.0f"%(m/1024./1024.) for m in res['per_rank_memory']) ) )
        print("Time Cost: %.3f sec"%(t_end-t_start))
    # ---------------------------------------------------------------
    
    return res

def sim_res_str(res, title="simulated"):
    return "=== %s : %.3f sec (idle max/avg: %.0f%%/%.0f%%) (compute2global/compute2rank: %.0f%%/%.0f%%), %.0f MB (%s) ==="% (
        title, 
        res['global_endtime'], 
        res['max_endidle']*100., 
        res['avg_endidle']*100., 
        res['avg_compute_to_globaltime']*100.,  
        res['avg_compute_to_ranktime']*100.,
        res['global_memory']/1024./1024.,
        ", ".join("%.0f"%(m/1024./1024.) for m in res['per_rank_memory']) )
