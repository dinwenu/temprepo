# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
import numpy as np
import random
from collections import deque

import sys; sys.path.append("../2_profiler")
from prof_data_struct import *
from analyze_profiles import time_of_pack, memory_of_pack, model_size_of_pack

from task_data_struct import Medium, vTask

# ❓到底是啥意思，不懂
# 已懂
# 返回task的最后一个subpack id，即一个task包含多层，若要保存其中多个层的输入X，则从每个对应的层开始的一些层被称为subpack
# 对前向任务，该值可能需要进行计算。对于BWD和UPD任务，返回的永远是0
def find_last_subpack_idx(vt):
    # 若是前向任务，且任务的Out中的X没有值，spi=0。
    if vt.type == "FWD":
        if not vt.Out['X']:
            spi = 0
        else:
            # spi=Out['X']键值对的数量
            spi = len(vt.Out['X'].keys()) + 1 - 1
            # 若Out['X']的key就是该vt的第一层，spi-=1
            # 📌为什么-1：因为下标从0开始，-1得到的spi就是最后一个subpack的下标，
            #    若vt的第0层不发送StashX，那么spi这个下标值也不用变，因为在 find_subpack_layers 中，
            #    vt的第0层会被添加到subpack的列表的首位，spi的下标依然是正确的
            #    📌换句话说，无论发不发vt首层的接收到的输入X，都默认该vt的首个subpack就是从首层开始的几层
            if vt.layers[0] in vt.Out['X']:
                spi -= 1

    # 对于BWD和UPD任务，spi永远=0
    else: # "BWD", "UPD"
        spi = 0
    assert spi >= 0
    return spi

# 返回：vt的源task，以及vt的首层在源任务中对应的subpack idx
def find_stash_subpack_idx(vt, TASKS):
    l, m = vt.layers[0], vt.In['X'][vt.layers[0]]; assert m.medium == "MSG"
    src_vt = TASKS[m.idx]; assert src_vt.idx == m.idx, "TASKS should be global"
    layers_stash = list(src_vt.Out['X'].keys())
    # 返回vt的首层在layers_stash中的下标，并+1
    src_spi = layers_stash.index(l) + 1
    # 若源task的首层存在于 layers_stash列表中，src_spi-1
    if src_vt.layers[0] in layers_stash:
        src_spi -= 1
    assert src_spi >= 0
    return src_vt, src_spi

# 返回BWD vt对应的src_task，即发送stashX的FWD vt，和src_task的最后一个subpack_idx（用于在由各个subpack首层组成的list中取值）
# 📌为什么返回最后一个subpack的id，因为这意味着
def find_output_subpack_idx(vt, TASKS):
    l, m = vt.layers[0], vt.In['X'][vt.layers[0]]; assert m.medium == "MSG"
    # 源任务(src_task)的idx
    src_vt = TASKS[m.idx]; assert src_vt.idx == m.idx, "TASKS should be global"
    # 
    src_spi = find_last_subpack_idx(src_vt)
    return src_vt, src_spi

def find_subpack_layers(vt, spi):
    if vt.type == "FWD":
        if not vt.Out['X']:
            assert spi == 0
            layers = vt.layers
        else:
            assert spi >= 0
            # 加上[vt.layers[-1]+1]显然是为了下面的range(start, stop)准备的，[vt.layers[-1]+1就是那个stop
            layers_stash = list(vt.Out['X'].keys()) + [vt.layers[-1]+1]
            # 若Out['X']中不包含该vt的首层，还需将首层加入到layers_stash列表中
            if vt.layers[0] not in layers_stash:
                layers_stash = [vt.layers[0]] + layers_stash
            # range(start, stop)
            layers = list(range(layers_stash[spi], layers_stash[spi+1]))
    else: # "BWD", "UPD"
        assert spi == 0
        layers = vt.layers
    return layers

class Event(object):
    def __init__(self, vt, ui, spi, stream, kind, ubs=None):
        self.vt = vt # affiliated vtask
        self.ui = ui # affiliated ubatch index
        self.spi = spi # affiliated subpack index
        self.stream = stream # "SwapIn"/"P2PIn"/"Compute" # for per-stream event queue
        self.kind = kind # "W"/"X"/"Y"/"FWD"/"BWD"
        # five tuple as id
        self.id = "%d.%d.%d.%s.%s"%(self.vt.idx, self.ui, self.spi, self.stream, self.kind)
        # microbatch size
        if ubs is None:
            self._find_ubs() # self.ubs = None or 2
        else:
            self.ubs = int(ubs) # override for MSG
        # subpack as event layers
        self._find_layers() # self.layers = [1] or [1,2,3]
        # dependency
        self.inputs = [] # required input events from other streams
        self.is_done = False # whether executed
        # p2p
        self.peer_id = None # peer event id
        self.pos = None # position idx in event queue
        # exec
        self.begin = 0.0 # begin time in sec
        self.dur = 0.0 # duration time in sec

    def _find_ubs(self):
        if self.kind in ["WB","DEL","ARD","dWB","Update"]:
            self.ubs = None
        elif self.kind in ["FWD","REC","BWD","sX","MSGsX","X","dX","MSGdX","Y","dY","MSGY"]:
            self.ubs = self.vt.ubatchszs[self.ui]
        else:
            raise NotImplementedError

    def _find_layers(self):
        if self.kind in ["WB","REC","BWD","DEL","ARD","dWB","Update"]:
            assert self.spi == 0
            self.layers = self.vt.layers
        elif self.kind in ["FWD"]:
            self.layers = find_subpack_layers(self.vt, self.spi)
        elif self.kind in ["sX","MSGsX"]:
            self.layers = [find_subpack_layers(self.vt, self.spi)[0]]
        elif self.kind in ["X","dX","MSGdX"]:
            assert self.spi == 0
            self.layers = [self.vt.layers[0]]
        elif self.kind in ["Y","dY","MSGY"]:
            assert self.spi >= 0
            self.layers = [self.vt.layers[-1]]
        else:
            raise NotImplementedError
            
    # 为inputs属性添加一个字符串，代表当前Event需要的（其他steams的）输入Event
    def inputs_add(self, vt, ui, spi, stream, kind):
        idx = vt.idx # if isinstance(vt, vTask) else vt
        # 返回spi（subpack index），对前向任务，该值可能需要进行计算。对BWD和UPD任务，返回的永远是0
        spi = find_last_subpack_idx(vt) if spi == -1 else spi
        # 将输入参数组成一个字符串加入到调用该函数的Event的inputs属性中，代表当前Event需要的（其他steams的）输入Event
        self.inputs.append("%d.%d.%d.%s.%s"%(idx, ui, spi, stream, kind))

    # 将参数ev添加到调用该函数的event的inputs列表中
    def inputs_add_ev(self, ev):
        assert isinstance(ev, Event)
        self.inputs.append(ev)
            
    # 1.将调用该方法的Event实例，添加到events这个字典中。{ id: Event() } 
    # 2.将Event添加到 self.stream 这种类型的事件队列中。per_stream_events：{stream: [Event()]}
    # 3.对Event的pos属性(在队列中的下标)赋值，即队列的最后一个位置
    def add_to(self, events, per_stream_events):
        # 将调用该方法的Event实例，添加到events这个字典中。{ id: Event() } 
        events[self.id] = self
        # 若Event.stream不在per_stream_events这个字典中，在字典中添加键值对 {self.stream：deque }
        if self.stream not in per_stream_events:
            per_stream_events[self.stream] = deque([])
        # 将Event添加到 self.stream 这种类型的事件队列中
        per_stream_events[self.stream].append(self)
        # 对Event的pos属性(在队列中的下标)赋值，即队列的最后一个位置
        self.pos = len(per_stream_events[self.stream])-1

    # 将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
    def register_peer(self, TASKS, ui, spi, stream, kind):
        """ register the peer Event ID """
        assert self.peer_id is None, "only one peer"
        # find peer vt
        if stream == "P2PIn":
            assert self.stream == "P2POut"
            if kind == "dY":
                m = self.vt.Out['dX'][self.vt.layers[0]]   
            elif kind == "X":
                m = self.vt.Out['Y'][self.vt.layers[-1]]
            else:
                raise ValueError
            assert m.medium == "P2P"
        elif stream == "P2POut":
            assert self.stream == "P2PIn"
            if kind == "Y":
                m = self.vt.In['X'][self.vt.layers[0]]
            elif kind == "dX":
                m = self.vt.In['dY'][self.vt.layers[-1]]
            else:
                raise ValueError
            assert m.medium == "P2P"
        else:
            raise ValueError
        # 即对应当前任务的那个任务
        peer_vt = TASKS[m.idx]; assert peer_vt.idx == m.idx, "TASKS is global"
        
        idx = peer_vt.idx
        # 返回task的最后一个subpack id，即一个task包含多层，若要保存其中多个层的输入X，则从每个对应的层开始的一些层被称为subpack
        # 对前向任务，该值可能需要进行计算。对于BWD和UPD任务，返回的永远是0
        spi = find_last_subpack_idx(peer_vt) if spi == -1 else spi
        # 将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
        self.peer_id = "%d.%d.%d.%s.%s"%(idx, ui, spi, stream, kind)

    # 1.将当前事件对应的P2P事件的所有依赖加入到当前事件的依赖中
    # 2.若对应的P2P事件不是其所在stream中的第1个事件，还需将P2P所在的rank的stream中的前一个事件加入到当前事件的依赖中
    # 3.清空当前事件Inputs中的重复依赖，并按照id进行排序
    def solve_peer(self, events, rank_stream_events):
        """ add peer's inputs and its previous stream Event to my inputs """
        if self.peer_id is None:
            return
        # 返回对应的P2P事件
        peer = events[self.peer_id]
        # 将对应的P2P事件的所有依赖加入到当前事件的依赖中
        self.inputs += peer.inputs 
        # 若对应的P2P事件不是其所在stream中的第1个事件，还需将P2P所在的rank的stream中的前一个事件加入到当前事件的依赖中
        if peer.pos-1 >= 0:
            self.inputs.append( 
                rank_stream_events[peer.vt.rank][peer.stream][peer.pos-1] )
        # 清空重复依赖并转化为列表
        self.inputs = list(set(self.inputs)) # remove double counting
        # 对inputs了列表按照id进行排序
        self.inputs = sorted(self.inputs, key=lambda e: e.id) # for result matching
             
    # 返回事件的结束事件（秒）
    @property
    def end(self): # end time in sec
        return self.begin + self.dur

    def show_layers(self):
        # assert list(range(self.layers[0], self.layers[-1]+1)) == list(self.layers)
        if len(self.layers) == 0:
            return "L--"
        elif len(self.layers) == 1:
            return "L%d" % (self.layers[0])
        else:
            return "L%d-%d" % (self.layers[0], self.layers[-1])
    def __str__(self):
        ### id, ubs, layers, [inputs ids], done
        ss = "%s, %s, %s, [%s], %s"%(
              self.id, 
              'U--' if self.ubs is None else 'U%d'%self.ubs,
              self.show_layers(),
              ",".join(inev.id for inev in self.inputs),
              "@%.3f_dur%.3f"%(self.begin, self.dur) if self.is_done else "-")
        return ss
    @property
    def name(self): # for display in chrome
        return "t{} {} {} {}".format(
                self.vt.idx, 
                '' if self.ubs is None else '#%d(%d)'%(self.ui, self.ubs),
                self.show_layers(), 
                self.kind)
        # return "t{} #{} {} {}".format(
        #         self.vt.idx, self.ui, self.show_layers(), self.kind)
        
# 
class UBSConverter(object):
    def __init__(self, ubatchszs_fwd, ubatchszs_bwd, u_bwd, verbose=True):
        """ examples: 
                [4,4,1], [4,4,1], 4
                [4,4,1], [2,2,2,2,1], 2
                [4,4,4,1], [3,3,3,3,1], 3
                [3,3,3,3,1], [4,4,4,1], 4
                [2,2,2,2,1], [4,4,1], 4
        """
        assert sum(ubatchszs_fwd) == sum(ubatchszs_bwd)
        self.ubatchszs_fwd = ubatchszs_fwd
        self.ubatchszs_bwd = ubatchszs_bwd
        self.u_bwd = u_bwd
        self.verbose = verbose
        # convertion
        # 将ubatchszs_fwd中每个ufwd按照u_bwd的大小拆分成一个tuple，每个值都是u_bwd，最后一个数可能是余数（实际上应该不会出现余数）
        # 📌分析：经过find_ubatchsizes函数(searcher.py)筛选的前后向ubsize应该不会出现这种情况。两者都是
        #         某个相同底数的次幂，都能被minibatchsize整除，说明互相之间也是可以整除的
        self._convert_ubatchsize()
        # 逐 ufwd 的建立，该 ufwd 拆分的每个 u_bwd 的全局下标，[[0,1](第0个ufwd的下标list),[2,3],...]
        self._map_idx_ufwd_to_ubwd()
        self._map_idx_ubwd_to_ufwd()
            
    def _concat(self, Cs):
        return int(sum(Cs))
    def _split(self, C, U):
        assert isinstance(C, int) and isinstance(U, int)
        if C >= U:
            if C % U == 0:
                ubatchszs = [U] * int(C/U)
            else:
                ubatchszs = [U] * int(C/U) + [ C%U ]
            assert sum(ubatchszs) == C
        else: # not sufficient to split
            ubatchszs = [C]
        return tuple(ubatchszs)
    
    # 将ubatchszs_fwd中每个ufwd按照u_bwd的大小拆分成一个tuple，每个值都是u_bwd，最后一个数可能是余数（实际上应该不会出现余数）
    # 📌分析：经过find_ubatchsizes函数(searcher.py)筛选的前后向ubsize应该不会出现这种情况。两者都是
    #         某个相同底数的次幂，都能被minibatchsize整除，说明互相之间也是可以整除的
    def _convert_ubatchsize(self):
        """ given ubatchszs, return per-ufwd converted list """
        per_ufwd_converted = [] # [[ubwd, ubwd], [ubwd, ubwd], [ubwd, 1]]
        
        residual = 0
        cnt_converted_ubwd = 0
        for ufwd in self.ubatchszs_fwd:
            # Use previous residual for split
            # 将ufwd(一个Int值，代表microbatch size)按照u_bwd的大小拆分成一个tuple，每个值都是u_bwd，最后一个数可能是余数
            split_u = self._split(self._concat((residual,ufwd)), self.u_bwd) # (c1,c2) or (c1,res) or (c1,) or (res,)
            residual = 0
            # Within split, check if each u is desired ubwd, if so make them as converted
            converted = []
            for j, u in enumerate(split_u):
                assert cnt_converted_ubwd < len(self.ubatchszs_bwd)
                desired_ubwd = self.ubatchszs_bwd[cnt_converted_ubwd]
                if u == desired_ubwd: # match
                    converted.append(u)
                    cnt_converted_ubwd += 1

                # 若ufwd按照ubwd拆分的最后一个值比ubatchszs_bwd中的最后一个值要小，则将该值赋给residual作为残余值
                # 📌分析：经过find_ubatchsizes函数(searcher.py)筛选的前后向ubsize应该不会出现这种情况。两者都是
                #         某个相同底数的次幂，都能被minibatchsize整除，说明互相之间也是可以整除的
                elif u < desired_ubwd: # residual
                    assert j == len(split_u)-1, "residual must be the last split"
                    residual = u
                else:
                    raise ValueError
            # Return converted
            # print("ufwd {} => ubwd {}".format(ufwd, converted))
            per_ufwd_converted.append(converted)
        self.per_ufwd_converted = per_ufwd_converted
        if self.verbose:
            print("[UBSConverter] per_ufwd_converted={}".format(self.per_ufwd_converted))

    # 给定ufwd的microbatch的序号（第几个），返回该ufwd转化后的ubwd list
    def find_converted(self, idx_ufwd):
        """ given microbatch idx of ufwd, return list of converted ubwd """
        return self.per_ufwd_converted[idx_ufwd]

    # 逐 ufwd 的建立，该 ufwd 拆分的每个 u_bwd 的全局下标，[[0,1](第0个ufwd的下标list),[2,3],...]
    def _map_idx_ufwd_to_ubwd(self):
        ufwd_to_ubwd = [] # [[ubwd#0, ubwd#1], [ubwd#2,ubwd#3], [ubwd#4,ubwd#5]]
        assert len(self.ubatchszs_fwd) == len(self.per_ufwd_converted)
        cnt = 0
        # 逐 ufwd 的建立，该 ufwd 拆分的每个 u_bwd 的全局下标，[[0,1](第0个ufwd的下标list),[2,3],...]
        for converted in self.per_ufwd_converted:
            ufwd_to_ubwd.append([cnt+i for i, _ in enumerate(converted)])
            cnt += len(converted)
        self.ufwd_to_ubwd = ufwd_to_ubwd
        assert cnt == len(self.ubatchszs_bwd)
        if self.verbose:
            print("[UBSConverter] idx of ufwd_to_ubwd={}".format(self.ufwd_to_ubwd))

    # 给定ufwd的microbatch的序号（第几个），返回该ufwd转化后的ubwd idx list，即转化后的ubwd列表中，每个ubwd的全局下标
    # [[0,1](第0个ufwd的下标list),[2,3],...]
    def find_idx_ubwd(self, idx_ufwd):
        """ given microbatch idx of ufwd, return list of converted ubwd idx"""
        return self.ufwd_to_ubwd[idx_ufwd]    

    # 遍历每个 ufwd 转换成的 ubwd 列表。建立列表中每个 ubwd 到其对应的 ufwd 的下标的映射
    # [0,0,1,1,2,2,...]
    # [ufwd#0, ufwd#0, ufwd#1, ufwd#1, ufwd#2, ufwd#2]
    def _map_idx_ubwd_to_ufwd(self):
        ubwd_to_ufwd = [] # [ufwd#0, ufwd#0, ufwd#1, ufwd#1, ufwd#2, ufwd#2]
        assert len(self.ubatchszs_fwd) == len(self.per_ufwd_converted)
        # 遍历每个 ufwd 转换成的 ubwd 列表。建立列表中每个 ubwd 到其对应的 ufwd 的下标的映射
        # [0,0,1,1,2,2,...]
        for idx_ufwd, converted in enumerate(self.per_ufwd_converted):
            for _ in range(len(converted)):
                ubwd_to_ufwd.append(idx_ufwd)
        self.ubwd_to_ufwd = ubwd_to_ufwd
        assert len(self.ubatchszs_bwd) == len(self.ubwd_to_ufwd)
        if self.verbose:
            print("[UBSConverter] idx of ubwd_to_ufwd={}".format(self.ubwd_to_ufwd))

    # 给定bwdvt的microbatch idx，返回产生该bwd microbatch的fwd microbatch idx
    def find_idx_ufwd(self, idx_ubwd):
        """ given microbatch idx of ubwd, return its producer idx of ufwd """
        return self.ubwd_to_ufwd[idx_ubwd]    

# 1.实例化一个事件Event，表示一个swapin WB的事件
# 2.将生成的event添加到event字典 { id（代表一个event的字符串）: Event() } 和对应的事件队列中 {self.stream: [Event()]}
#   并将event的pos属性(在队列中的下标)，设置为事件队列的最后一个位置
#   2.1.若输入的task的左边还有task，还会增加一步额外的操作，创建其左边任务的Event的字符串表示(用于创建event实例的5个参数)，
#       加入到当前event的inputs列表中。表示Compute DEL事件执行完了才能开始执行Swap In WB事件（📌即上一个任务的最后一个事件）
def In_WB(vt, events, stream_events, left_vt, delay_enqueue, prefetch_offload):
    # 实例化一个事件Event，表示一个swapin WB的事件
    ev_w = Event(vt, 0, 0, "SwapIn", "WB")
    # 若vt的前一个任务不存在
    if left_vt is None:
        # 1.将调用该方法的Event实例，添加到events这个字典中。events: { id（代表一个event的字符串）: Event() } 
        # 2.将Event添加到 self.stream 这种类型的事件队列中。stream_events：{self.stream: [Event()]}
        # 3.对Event的pos属性(在队列中的下标)赋值，即队列的最后一个位置
        ev_w.add_to(events, stream_events)
    # 该参数默认为false，不用看
    elif prefetch_offload: # prefetch at left vt
        ev_w.inputs_add(left_vt, len(left_vt.ubatchszs)-1, -1, "Compute", left_vt.type)
        if not delay_enqueue: # vPP-["F", "Bc"]
            ev_w.add_to(events, stream_events)
        else: # vDP or vPP-"Bn"
            pass
    # 若vt的前一个任务存在（且不prefetch offload）
    else: # no prefetch offload
        # 这五个参数相当于定义了一个Event，尽管函数里没有真正的生成一个Event
        # 为ev_w的inputs属性添加一个字符串，代表当前Event需要的（其他streams的）输入依赖事件
        ev_w.inputs_add(left_vt, 0, 0, "Compute", "DEL")
        ev_w.add_to(events, stream_events)
    
    return ev_w

# 1.为task的输入X生成一个event
# 2.针对具体的情况，为event的inputs属性赋值，即表示对应当前输入事件的输出事件
# 3.将ev_w_comp_out参数（一个vt）加入到新生成的event的inputs列表中
# 📌分析：个人觉得subpack idx不产生实际作用，仅表明subpack的数量（无论发不发送第0层的stashX,前几层都不算一个subpack）
def In_X(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ubscvt, ev_w, prefetch_offload, ev_w_comp_out):
    # assert ['F','Bc']
    # 包含输入数据的 fwd 任务
    if vt.has_data: # "Data"
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        assert left_vt is None # 1st task

    # 产生输入X的task和vt不在一个GPU上。这种情况包含：不包含输入数据的FWD任务、包含loss计算层的BWD任务
    elif vt.In["X"][vt.layers[0]].medium == "P2P": # "P2PX"
        ev_x = Event(vt, i, 0, "P2PIn", "X")
        # 为ev_x的peer_id属性赋值，即将代表对应当前任务的p2p任务以字符串的形式加入到ev_x的inputs列表中
        # 📌：分析，这其中的返回最后一个subpack的id我可以理解，毕竟要把最后一个stashX发送出去，后面才是
        #     P2PIn的发生
        ev_x.register_peer(TASKS, i, -1, "P2POut", "Y") # p2p dependency
        assert ubscvt is None

    # 这种情况是BWD任务，这种任务需要其前向任务的输入
    # 📌分析：对BWD的FWD（即重计算任务），必须等待对应的FWD任务发送完最后一个StashX
    # ❓：这我就不太理解了，为啥得等最后一个？
    elif vt.In["X"][vt.layers[0]].medium == "MSG": # last fwd "MSG" of vPP
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        # 返回BWD vt对应的src_task，即发送stashX的FWD vt，和src_task的最后一个subpack_idx（用于在由各个subpack首层组成的list中取值）
        src_vt, src_spi = find_output_subpack_idx(vt, TASKS)
        # 为ev_w的inputs属性添加一个字符串，代表当前Event依赖的（其他steams的）Event
        # 即CPU MSGY事件完成后，才能开始SwapIn X事件
        ev_x.inputs_add(src_vt, i, src_spi, "CPU", "MSGY") # msg dependency

    # 产生输入X的task和vt在一个GPU上，这种情况可能为：不包含输入数据的FWD任务、包含loss计算层的BWD任务
    elif vt.In["X"][vt.layers[0]].medium == "SWP": # "LocalX" of vDP
        ev_x = Event(vt, i, 0, "SwapIn", "X")
        # if 'Bc':
        #     ui = i if ubscvt is None else ubscvt.find_idx_ufwd(i)
        # elif 'F':
        #     ui = i
        # 给定bwdvt的microbatch idx，返回产生该bwd microbatch的fwd microbatch idx
        ui = i if ubscvt is None else ubscvt.find_idx_ufwd(i)
        ev_x.inputs_add(left_vt, ui, -1, "SwapOut", "Y") # swap dependency
    else:
        raise ValueError
    
    # 在gpt2的例子中该参数默认为false，执行else
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_x.add_to(events, stream_events) 
            else: # prefetch at left vt
                if len(left_vt.ubatchszs) >= 2: # double buffer
                    ev_x.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                else: # left_vt has only 1 ubatch
                    if left2_vt is not None:
                        ev_x.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_x.add_to(events, stream_events)
                if ev_w is not None: # 'vDP'-['F','Bc']
                    ev_w.add_to(events, stream_events) 
                else: # vPP
                    pass
        elif i == 1:
            if left_vt is not None:
                ev_x.inputs_add(left_vt, 0, 0, "Compute", "DEL")
            ev_x.add_to(events, stream_events)
        else: # i >= 2
            ev_x.inputs_add(vt, i-2, -1, "Compute", vt.type)
            ev_x.add_to(events, stream_events)
    else: # no prefetch offload
        # if i == 0:
        #     assert ev_w is not None
        #     ev_x.inputs_add_ev(ev_w)
        # else: # i >= 1
        #     if ev_comp_out is not None:
        #         ev_x.inputs_add_ev(ev_comp_out) # if vDP, ev_out; if vPP, ev_comp
        #     else:
        #         ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type) # if None dX
        if ev_w_comp_out is not None:
            # 将参数ev添加到调用该函数的event的inputs列表中
            ev_x.inputs_add_ev(ev_w_comp_out)
        else: # if None dX
            ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type)
        ev_x.add_to(events, stream_events)
    
    return ev_x

# 1.实例化一个SwapIn sX事件，并加入到两个字典中
# 2.将CPU MSGsX事件添加到SwapIn sX事件的输入依赖中，表示CPU MSGsX事件完成后才能执行Swap In sX事件
# 3.若i为0，即执行的是首个micro batch，添加的依赖事件为 Swapin WB
#   否则，添加的依赖事件为执行前面的micro batch的 Compute BWD 事件，表示前面的BWD算完了才能开始 SwapIn sX事件
def In_sX(vt, i, events, stream_events, left_vt, left2_vt, src_vt, src_spi, prefetch_offload, ev_w_comp_out):
    # assert 'Bn'
    # 实例化一个SwapIn sX事件
    ev_sx = Event(vt, i, 0, "SwapIn", "sX")
    # 将CPU MSGsX事件添加到SwapIn sX事件的输入依赖中，表示CPU MSGsX事件完成后才能执行Swap In sX事件
    ev_sx.inputs_add(src_vt, i, src_spi, "CPU", "MSGsX") # swap dependency
    
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_sx.add_to(events, stream_events)
            else: # prefetch at left vt
                if len(left_vt.ubatchszs) >= 2: # double buffer
                    ev_sx.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                else: # left_vt has only 1 ubatch
                    if left2_vt is not None:
                        ev_sx.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_sx.add_to(events, stream_events)
        elif i == 1:
            if left_vt is not None:
                ev_sx.inputs_add(left_vt, 0, 0, "Compute", "DEL")
            ev_sx.add_to(events, stream_events)
        else: # i >= 2
            ev_sx.inputs_add(vt, i-2, 0, "Compute", vt.type)
            ev_sx.add_to(events, stream_events)
    else: # no prefetch offload
        # if i == 0:
        #     assert ev_w is not None
        #     ev_sx.inputs_add_ev(ev_w)
        # else: # i >= 1
        #     if ev_comp_out is not None:
        #         ev_sx.inputs_add_ev(ev_comp_out) # if vDP, ev_out; if vPP, ev_comp
        #     else:
        #         ev_x.inputs_add(vt, i-1, -1, "Compute", vt.type) # if None dX

        # 若i为0，即执行的是首个micro batch，添加的依赖事件为 Swapin WB
        # 否则，添加的依赖事件为执行前面的micro batch的 Compute BWD 事件，表示前面的BWD算完了才能开始 SwapIn sX事件
        if ev_w_comp_out is not None:
            ev_sx.inputs_add_ev(ev_w_comp_out)
        else: # if None dX
            ev_sx.inputs_add(vt, i-1, -1, "Compute", vt.type)
        ev_sx.add_to(events, stream_events)
    
    return ev_sx

# 📌sX是stashX
# 1.实例化一个前向计算事件
# 2.若micro batch idx = 0，还要将ev_w事件加入到当前事件的inputs属性中，即 ev_w （WB：参数和buffer） 为当前事件的输入依赖
# 3.将ev_x加入当前事件的inputs属性中，即 ev_x （X:激活值） 为当前事件的输入依赖
# 4.将生成的event添加到event字典 { id（代表一个event的字符串）: Event() } 和对应的事件队列中 {self.stream: [Event()]}
#   并将event的pos属性(在队列中的下标)，设置为事件队列的最后一个位置
# 5.若一个task中有多层的输入都会被反向传播时用到，则从第2个需要的层开始相当于其他的subpack，为其创建新的event，新的event的spi
#   从1开始累加，即subpack1、subpack2、...
# 6.为每一个实例化的event添加一个输入依赖事件，即Swapout掉stashX。❓这个直觉上看起来像输出依赖
def Compute_FWD(vt, i, events, stream_events, ev_w, ev_x, prefetch_offload):
    # 'F'
    spi = 0
    # 实例化一个前向计算事件
    ev_fwd = Event(vt, i, spi, "Compute", "FWD") # subpack[0]
    # 若micro batch idx = 0，还要将ev_w事件加入到当前事件的inputs属性中，即 ev_w （WB：参数和buffer） 为当前事件的输入依赖
    if i == 0:
        ev_fwd.inputs_add_ev(ev_w)
    # 将ev_x加入当前事件的inputs属性中，即 ev_x （X:激活值） 为当前事件的输入依赖
    ev_fwd.inputs_add_ev(ev_x)
    # 1.将调用该方法的Event实例，添加到events这个字典中。events: { id（代表一个event的字符串）: Event() } 
    # 2.将Event添加到 self.stream 这种类型的事件队列中。stream_events：{self.stream: [Event()]}
    # 3.对Event的pos属性(在队列中的下标)赋值，即队列的最后一个位置
    ev_fwd.add_to(events, stream_events)
    # sub-packing by StashX
    # 层号，medium（媒介）
    for l, m in vt.Out['X'].items(): # layers in ascending order
        spi += 1
        # 若是vt的第0层
        if l == vt.layers[0]: # reuse subpack[0]
            spi -= 1; assert spi == 0

        # 执行到这，说明Out['X']中有多个键值对，即除layer0外，还有其他层的输入X会在反向传播时使用。此时，
        # 实例化新的fwd event，代表subpack，spi从1开始。并将新的event加入到events和stream_events字典中
        # 📌可见，每个FWD任务会被拆解为多个FWD事件，可通过spi识别各个被拆分的FWD事件
        else: # create a subpack[1+]
            ev_fwd = Event(vt, i, spi, "Compute", "FWD")
            ev_fwd.add_to(events, stream_events)
        # 该参数默认为false，因此这里执行
        # 为ev_fwd添加一个输入依赖event，即Swapout掉stashX才能继续进行 FWD 计算
        if not prefetch_offload:
            ev_fwd.inputs_add(vt, i, spi, "SwapOut", "sX")
    
    # 返回最后一个(spi)ev_fwd
    return ev_fwd

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
# 最后返回MSGsX事件
def Out_sX(vt, i, u, events, stream_events, left_vt, ubscvt, ev_x, prefetch_offload):
    # 'F'
    ev_msg = None
    spi = 0
    # sub-packing by StashX
    # 层号，medium（媒介）
    for l, m in vt.Out['X'].items(): # layers in ascending order
        spi += 1
        # 📌可见FWD vt的首层必定会发送StashX！！！！！！！！！！！！！！！！！！！！！！！！
        # ❓：是什么逻辑让部分BWD任务的首层必定是FWD任务的首层？
        # 分析：不一定是必须，这里是if，没if才是必须
        if l == vt.layers[0]: # reuse subpack[0]
            spi -= 1; assert spi == 0
            # 实例化一个swapout sX事件
            ev_sx = Event(vt, i, spi, "SwapOut", "sX")
            # 将 ev_x（vt的输入X事件） 添加到ev_sx的Inputs属性中，即ev_x为当前事件的依赖事件
            ev_sx.inputs_add_ev(ev_x)
            if prefetch_offload:
                if i == 0:
                    if left_vt is not None:
                        ev_sx.inputs_add(left_vt, 0, 0, "Compute", "DEL")
                else:
                    ev_sx.inputs_add(vt, i-1, -1, "Compute", "FWD")
            ev_sx.add_to(events, stream_events)

        # 执行到这，说明Out['X']中有多个键值对，即除layer0外，还有其他层的输入X会在反向传播时使用。此时，
        # 实例化新的fwd event，代表subpack，spi从1开始。并将新的event加入到events和stream_events字典中
        else: # create a subpack[1+]
            ev_sx = Event(vt, i, spi, "SwapOut", "sX")
            # 添加 前一个subpack的前向计算 事件为ev_sx的需求输入任务
            ev_sx.inputs_add(vt, i, spi-1, "Compute", "FWD")
            # 将刚刚实例化的下一个subpack的SwapOut sX事件添加到两个字典中
            ev_sx.add_to(events, stream_events)
        # "MSGsX"
        assert m.medium == "MSG"
        # 给定ufwd的microbatch的序号（第几个），返回该ufwd转化后的ubwd idx list，即转化后的ubwd列表中，每个ubwd的全局下标
        # [[0,1](第0个ufwd的下标list),[2,3],...]
        indice_bwd = [i] if ubscvt is None else ubscvt.find_idx_ubwd(i)
        # 给定ufwd的microbatch的序号（第几个），返回该ufwd转化后的ubwd list
        us_bwd = [u] if ubscvt is None else ubscvt.find_converted(i)
        for i_bwd, u_bwd in zip(indice_bwd, us_bwd):
            # 实例化一个CPU上的MSGsX事件
            # 📌注意，这里传入的micro batch index是i_bwd，即该事件的ubi直接对应BWDvt的ubi
            ev_msg = Event(vt, i_bwd, spi, "CPU", "MSGsX", ubs=u_bwd)     
            # 将最后一个subpack的ev_sx设置为刚刚实例化的事件的需求输入事件，即暂存的X swapout以后CPU才能发送
            ev_msg.inputs_add_ev(ev_sx)
            ev_msg.add_to(events, stream_events)
    
    return ev_msg

# 两种情况：
# 1.若vt的Out['Y']最后一层的媒介为P2P
#   1.1.实例化一个P2P事件，其spi为 ev_fwd（fwd计算事件）的spi，并将实例化的事件添加到两个字典中
#   1.2.将ev_fwd添加到当前这个P2POut Y事件的需求输入列表中，表示前向计算的事件完成后才能开始P2POut
#   1.3.此外，还要将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
# 2.若Out['Y']最后一层的媒介为MSG，这种情况为最后一个fwd
#   2.1.实例化一个Swap out Y事件，其spi为 ev_fwd（fwd计算事件）的spi，并将实例化的事件添加到两个字典中。
#   2.2.将ev_fwd添加到当前这个Swapout Y事件的需求输入列表中，表示前向计算的事件完成后才能执行CPU上发送Y的事件
#   2.3.实例化一个CPU MSGY事件，可能代表CPU上输出或输入Y的事件。将刚刚实例化的Swap out Y事件添加到MSGY事件的inputs列表中，
#       表示Swapout Y之后才能执行CPU上MSGY事件
def Out_Y(vt, i, u, events, stream_events, ubscvt, TASKS, ev_fwd):
    # 'F'
    # 若Out['Y']最后一层的媒介为P2P
    if vt.Out["Y"][vt.layers[-1]].medium == "P2P": # P2PY
        # 实例化一个P2P事件，其spi为 ev_fwd（fwd计算事件）的spi
        ev_y = Event(vt, i, ev_fwd.spi, "P2POut", "Y")
        # 将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
        ev_y.register_peer(TASKS, i, 0, "P2PIn", "X")
        # 将ev_fwd添加到当前这个P2POut Y事件的需求输入列表中，表示前向计算的事件完成后才能开始P2POut
        ev_y.inputs_add_ev(ev_fwd) 
        # 将实例化的事件添加到两个字典中
        ev_y.add_to(events, stream_events)

    # 若Out['Y']最后一层的媒介为MSG，这种情况为最后一个fwd
    # 📌事实上，FWD发送Y的媒介根本就不会是MSG
    elif vt.Out["Y"][vt.layers[-1]].medium == "MSG": # last fwd MSG
        # 实例化一个Swap out Y事件
        ev_y = Event(vt, i, ev_fwd.spi, "SwapOut", "Y")
        ev_y.inputs_add_ev(ev_fwd)
        ev_y.add_to(events, stream_events)
        # "MSGY"
        # 对最后一fwd任务，还需创建CPU MSGY事件，可能代表CPU上输出或输入Y的事件
        indice_bwd = [i] if ubscvt is None else ubscvt.find_idx_ubwd(i)
        us_bwd = [u] if ubscvt is None else ubscvt.find_converted(i)
        for i_bwd, u_bwd in zip(indice_bwd, us_bwd):
            ev_msg = Event(vt, i_bwd, ev_fwd.spi, "CPU", "MSGY", ubs=u_bwd)            
            # 将ev_y添加到MSGY事件的inputs列表中，表示Swapout Y之后才能执行CPU上发送Y的事件
            ev_msg.inputs_add_ev(ev_y)
            ev_msg.add_to(events, stream_events)
    elif vt.Out["Y"][vt.layers[-1]].medium == "SWP": # vDP only
        ev_y = Event(vt, i, ev_fwd.spi, "SwapOut", "Y")
        # 即FWD完成后，才能Swapout Y
        ev_y.inputs_add_ev(ev_fwd) 
        ev_y.add_to(events, stream_events)
    else:
        raise ValueError
    
    return ev_y

# 1.若task的In['dY']的最后一层媒介为P2P，实例化一个P2PIn dY事件并加入到两个字典中，将对应的那个p2p任务以字符串的形式赋给
#   调用该函数的event的peer_id属性
#   vDP：若task的In['dY']的最后一层媒介为SwapIn，...，将其左边task的SwapOut dX事件加入到输入依赖中
# 2.将 Compute REC 事件加入到P2PIn dY的依赖事件中，表示当前任务的第i个ubatch的反向Compute REC事件完成后，
#   才能开始当前ubatch上的In dY事件
def In_dY(vt, i, events, stream_events, left_vt, left2_vt, TASKS, ev_w, prefetch_after_rec, prefetch_offload):
    # assert 'Bn'
    if vt.In["dY"][vt.layers[-1]].medium == "P2P": # P2PdY
        # 实例化一个P2PIn dY事件
        ev_dy = Event(vt, i, 0, "P2PIn", "dY")
        # 将对应的那个p2p任务以字符串的形式赋给调用该函数的event的peer_id属性
        ev_dy.register_peer(TASKS, i, 0, "P2POut", "dX")
    elif vt.In["dY"][vt.layers[-1]].medium == "SWP": # LocaldY of vDP only
        # 实例化一个SwapIn dY事件
        ev_dy = Event(vt, i, 0, "SwapIn", "dY")
        # 将其左边task的SwapOut dX事件加入到输入依赖中
        ev_dy.inputs_add(left_vt, i, 0, "SwapOut", "dX") # swap dependency
        assert left_vt is not None
    else:
        assert False
    
    if prefetch_offload:
        if i == 0:
            if left_vt is None: # 1st task
                ev_dy.add_to(events, stream_events)
            else: # prefetch at left vt
                if prefetch_after_rec or (left_vt.type == "BWD" and not left_vt.has_criterion): # prefetch_after_rec: True for vDP, False for vPP
                    ev_dy.inputs_add(left_vt, len(left_vt.ubatchszs)-1, -1, "Compute", "REC")
                else: # FWD or BWD(criterion)
                    if len(left_vt.ubatchszs) >= 2: # double buffer
                        ev_dy.inputs_add(left_vt, len(left_vt.ubatchszs)-2, -1, "Compute", left_vt.type)
                    else: # left_vt has only 1 ubatch
                        if left2_vt is not None:
                            ev_dy.inputs_add(left2_vt, 0, 0, "Compute", "DEL")
                ev_dy.add_to(events, stream_events)
                ev_w.add_to(events, stream_events)
        else: 
            ev_dy.inputs_add(vt, i-1, 0, "Compute", "REC")
            ev_dy.add_to(events, stream_events)
    else:
        # 将 Compute REC 事件加入到P2PIn dY的依赖事件中，表示当前任务的第i个ubatch的反向Compute REC事件完成后，
        # 才能开始当前ubatch上的In dY事件
        ev_dy.inputs_add(vt, i, 0, "Compute", "REC")
        ev_dy.add_to(events, stream_events)
    
    return ev_dy

# 1.实例化一个Compute REC事件(重计算事件)，并将其加入到两个字典中
# 2.若当前是首个micro batch，需将ev_w（即SwapIn WB）事件作为输入依赖
# 3.实例化一个Compute BWD事件，并将其加入到两个字典中
# 返回连个事件：Cpmpute REC事件、Compute BWD事件
def Compute_BWD(vt, i, events, stream_events, ev_w, ev_x, ev_dy):
    # 实例化一个Compute REC事件
    # 很显然是重计算事件
    ev_rec = Event(vt, i, 0, "Compute", "REC")
    # 若当前是首个micro batch，需将ev_w（即SwapIn WB）事件作为输入依赖
    if i == 0:
        ev_rec.inputs_add_ev(ev_w)
    # 将输入X事件作为依赖
    ev_rec.inputs_add_ev(ev_x) # Bc: inputX, Bn: stashX
    ev_rec.add_to(events, stream_events)
    
    # 实例化一个Compute BWD事件
    ev_bwd = Event(vt, i, 0, "Compute", "BWD")
    if ev_dy is not None: # "Bn"
        ev_bwd.inputs_add_ev(ev_dy)
    ev_bwd.add_to(events, stream_events)
    
    # 返回连个事件：Cpmpute REC事件、Compute BWD事件
    return ev_rec, ev_bwd

# 两种情况
# 1.若任务的Out[dX]的首层的媒介为P2P
#   1.1.实例化一个P2POut dX事件 ev_dx，并加入到两个字典中
#   1.2.将对应的那个p2p任务以字符串的形式赋给调用该函数的 ev_dx 的peer_id属性
#   1.3.将 ev_bwd（即第i个 compute BWD事件）作为当前事件的输入以来
# 2.若任务的Out[dX]的首层的媒介为SWP，说明为vDP
#   2.1.实例化一个SwapOut dX任务,并加入到两个字典中
#   2.2.将 ev_bwd（即第i个 compute BWD事件）作为当前事件的输入以来
def Out_dX(vt, i, events, stream_events, TASKS, ev_bwd):
    if vt.layers[0] in vt.Out["dX"]:
        # assert not vt.has_data
        if vt.Out["dX"][vt.layers[0]].medium == "P2P": # P2PdX
            ev_dx = Event(vt, i, 0, "P2POut", "dX")
            ev_dx.register_peer(TASKS, i, -1, "P2PIn", "dY")
            ev_dx.inputs_add_ev(ev_bwd)
            ev_dx.add_to(events, stream_events)
        elif vt.Out["dX"][vt.layers[0]].medium == "SWP": # vDP only
            ev_dx = Event(vt, i, 0, "SwapOut", "dX")
            ev_dx.inputs_add_ev(ev_bwd)
            ev_dx.add_to(events, stream_events)
        else:
            assert False
    else:
        ev_dx = None
    
    return ev_dx

def Compute_ARD(vt, events, stream_events, num_gpus, prefetch_offload, ev_dx):
    # assert 'vDP'-['Bc','Bn']
    if num_gpus > 1:
        ev_ard = Event(vt, 0, 0, "Compute", "ARD")
        ev_ard.add_to(events, stream_events)
        if not prefetch_offload and ev_dx is not None:
            ev_ard.inputs_add_ev(ev_dx)
    else:
        ev_ard = None
    
    return ev_ard

# 实例化一个SwapOut dWB事件，将其加入两个字典中，并将最后一反向计算事件作为该事件的输入依赖
def Out_dWB(vt, events, stream_events, ev_bwd, ev_ard, prefetch_offload, ev_dx):
    # assert ["Bc",'Bn']
    ev_dw = Event(vt, 0, 0, "SwapOut", "dWB")
    ev_dw.inputs_add_ev(ev_bwd)
    if ev_ard is not None: # 'vDP' and num_gpus > 1
        ev_dw.inputs_add_ev(ev_ard)
    elif not prefetch_offload and ev_dx is not None: # 'vDP' and num_gpus == 1
        ev_dw.inputs_add_ev(ev_dx)
    ev_dw.add_to(events, stream_events)
    
    return ev_dw

# 
def Compute_DEL(vt, events, stream_events, ev_dw, prefetch_offload, ev_y):
    # 实例化一个Compute DEL事件
    ev_del = Event(vt, 0, 0, "Compute", "DEL")
    # 反向传播时发生：若ev_dw存在（SwapOut dWB事件），将其作为该事件的输入依赖
    if ev_dw is not None: # 'Bc','Bn'
        ev_del.inputs_add_ev(ev_dw)
    elif not prefetch_offload and ev_y is not None: # 'F' 
        ev_del.inputs_add_ev(ev_y)
    # 1.将调用该方法的Event实例，添加到events这个字典中。{ id: Event() } 
    # 2.将Event添加到 self.stream 这种类型的事件队列中。per_stream_events：{stream: [Event()]}
    # 3.对Event的pos属性(在队列中的下标)赋值，即队列的最后一个位置
    ev_del.add_to(events, stream_events)
    
    return ev_del

# 1.实例化以一个CPU Update事件，并将其加入到两个字典中
# 2.将SwapOut dWB事件作为当前事件的依赖事件
def CPU_Update(vt, events, stream_events, left_vt):
    assert left_vt is not None and left_vt.layers == vt.layers
    # 实例化以一个CPU Update事件
    ev = Event(vt, 0, 0, "CPU", "Update")
    # 将SwapOut dWB事件作为当前事件的依赖事件
    ev.inputs_add(left_vt, 0, 0, "SwapOut", "dWB")
    ev.add_to(events, stream_events)
    
    return ev

class Dispatcher(object):
    def __init__(self, rank_stream_events):# { rank: { stream: [Event()] }
        self.event_queues = deque([]) # non-empty event queues across all ranks
        #                    [ rank: {stream: [Event()], ... } , ... ]
        for stream_events in rank_stream_events.values(): # can be empty rank
            # eq:一个rank中的stream对应的装着event的list 
            # eq是一个deque:
            # [Event(),...]
            for eq in stream_events.values():
                assert len(eq) != 0
                self.event_queues.append(eq)
        # for rank, stream_events in rank_stream_events.items(): 
        #     for k in [k for k, v in stream_events.items() if v == []]:
        #         del stream_events[k]
        # for k in [k for k, v in rank_stream_events.items() if not v]:
        #     del rank_stream_events[k]
        self.num_streams = len(self.event_queues) # statisics
        print(f".....num_streams:{self.num_streams}")

    # 
    def _check_inputs(self, ev):
        if ev.inputs == []:
            return True
        else:
            return min([inev.is_done for inev in ev.inputs]) 
            # return min([self.events[id].is_done for id in ev.inputs]) 
            # Any False -> False; All True -> True
            
    # 遍历 event_queues 中所有的list，直到某个list的首个event没有输入依赖，返回该event
    # 若不存在这样一个事件，则返回所有stream list的首个event
    def dispatch(self):
        # 若event_queues中没有任何event deque了，直接返回done字符串
        if len(self.event_queues) == 0:
            return "done" # all events dispatched
        # 最大步数为，event_queues×2，即所有GPU上所有的stream的数量×2
        # 分析📌：for循环的作用是保证所有的event deque都被遍历一遍，确保能return一个ev
        #         尽管是一个for循环，但本意是只返回一个ev，并不是要返回所有遍历到的ev
        max_step = len(self.event_queues)*2
        for _ in range(max_step):
            # try a non-empty queue
            # 从event_queues中弹出一个deque, 即某个rank的装着某个stream所有event的list
            events = self.event_queues.popleft() # round-robin abitration
            # 若 events[0]，即一个rank上的某个stream上的首个事件，没有输入依赖，或其依赖全部执行完了，则执行：
            if self._check_inputs(events[0]): # event found
                # 弹出events列表的第一个event
                ev = events.popleft()
                # 若events中还有其他event，重新将events列表放回 event_queues 后面
                if len(events) != 0:
                    self.event_queues.append(events)
                # 返回该ev，用于模拟该ev的执行时间
                return ev # dispatch a single event
            # 若list的首个事件有输入依赖还没执行完，则重新把该list放回队列中
            self.event_queues.append(events)
        # deadlock 
        # 若没有找到一个没有输入依赖的事件，则返回所有stream list的首个event
        return [events[0] for events in self.event_queues] # dealock events

class Executor(object):
    def __init__(self, args, non_empty_gpus, CONFIGS, TASKS, rank_stream_events):
        self.prof = args.prof
        self.bw_swap = args.bw_swap
        self.bw_p2p = args.bw_p2p
        self.bw_msg = args.bw_msg
        self.time_del = args.time_del # 设置好的值，默认为0.04
        self.mode = CONFIGS["mode"]
        self.N = non_empty_gpus # CONFIGS["N"]
        self.R = CONFIGS["R"]
        self.TASKS = TASKS; assert TASKS[-1].idx == len(TASKS)-1, "TASKS is global"
        self.use_random = args.use_random
        if self.use_random:
            random.seed(args.seed)
            np.random.seed(args.seed)
        # { rank : { stream : last Event's end time } or {} }
        self.rank_stream_endtime = ODict()
        for r, stream_events in rank_stream_events.items(): # can be empty rank
            self.rank_stream_endtime[r] = ODict()
            for s in stream_events.keys():
                self.rank_stream_endtime[r][s] = 0.0 # sec
        # { rank : compute time accumulated }
        self.rank_compute = ODict()
        for r in rank_stream_events.keys(): # can be empty rank
            self.rank_compute[r] = 0.0 # sec
        # count executed events
        self.cnt = 0 
    
    # 直接拿到或根据带宽计算事件的持续时间
    def _duration(self, ev):
        ubs = ev.ubs
        l_start, l_end = ev.layers[0], ev.layers[-1] # subpack layers
        # 📌REC也做和FWD相同的事情，猜测REC是recompute
        # 答：REC就是RECompute
        if ev.kind in ["FWD","REC"]:
            # 根据type, ubatchsize从 第2阶段 保存的class中取出 [start_id, end_id] 这几层保存的时间（type为FWD/BWD/UDP）
            return time_of_pack(self.prof, "FWD", ubs, l_start, l_end, interp_ubatchsize=True) # sec
        elif ev.kind == "BWD":
            # 反向传播的时间包含了该层重计算的时间，因此要减去前向计算的时间
            return time_of_pack(self.prof, "BWD", ubs, l_start, l_end,     interp_ubatchsize=True) - \
            time_of_pack(self.prof, "FWD", ubs, l_start, l_end,     interp_ubatchsize=True)
        
        # 若在2阶段没有为
        elif ev.kind == "Update":
            return time_of_pack(self.prof, 'UPD', None, l_start, l_end, offload_optim=True)
        # 删除事件花费的时间，delete
        elif ev.kind == "DEL":
            return self.time_del # sec # empirical value
        # 
        elif ev.kind == "ARD":
            W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
            B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
            return 2.*(self.N-1)/self.N*(W+B)/self.bw_p2p # sec
        
        # WB和其导数，只能以swap的方式通信，除以的带宽为self.bw_swap/self.N
        elif ev.kind in ["WB","dWB"]:
            W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
            B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
            return float(W+B)/(self.bw_swap/self.N) # FWD-SwapIn/BWD-SwapIn/BWD-SwapOut in vDP/vPP
        
        # 若stream的名字以Swap开始，除以的带宽为self.bw_swap/self.N
        # 若stream的名字以P2P开始，除以的带宽为self.bw_p2p
        elif ev.kind in ["sX","X","dX"]:
            X = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
            if ev.stream.startswith("Swap"):
                return float(X)/(self.bw_swap/self.N) # FWD/BWD SwapIn/SwapOut in vDP/vPP
            elif ev.stream.startswith("P2P"):
                return float(X)/(self.bw_p2p) # FWD-P2PIn/BWD-P2POut in vPP
            else:
                raise NotImplementedError 
            
        # 若stream的名字以Swap开始，除以的带宽为self.bw_swap/self.N
        # 若stream的名字以P2P开始，除以的带宽为self.bw_p2p
        elif ev.kind in ["Y","dY"]:
            Y = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
                if l_end+1 < self.R else 0. 
            if ev.stream.startswith("Swap"):
                return float(Y)/(self.bw_swap/self.N) # FWD-SwapOut/BWD-SwapIn in vDP
            elif ev.stream.startswith("P2P"):
                return float(Y)/(self.bw_p2p) # FWD-P2PIn/BWD-P2POut in vPP 
            else:
                raise NotImplementedError 
            
        # MSG除以的带宽为 (self.bw_msg/self.N)
        elif ev.kind.startswith("MSG"): # ["MSGsX","MSGdX","MSGY"]
            kind = ev.kind.replace("MSG","")
            if kind == "sX":
                # 即找到对应任务的GPU序号，查看两者是否在一个GPU上，若在，则需要传输的数据量为0
                if self.TASKS[ev.vt.Out["X"][l_start].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    # l_start这一层在ubs下的 输入 的大小
                    M = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)

            # 📌实际上，dX的媒介就不可能是MSG
            elif kind == "dX":
                if self.TASKS[ev.vt.Out["dX"][l_start].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    M = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)

            # 📌实际上，Y的媒介就不可能是MSG
            elif kind in "Y":
                if self.TASKS[ev.vt.Out["Y"][l_end].idx].rank == ev.vt.rank:
                    M = 0. # self send
                else:
                    M = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
                        if l_end+1 < self.R else 0.
            else:
                raise NotImplementedError
            return float(M)/(self.bw_msg/self.N)*2. # CPU memory is half-duplex
        else:
            raise NotImplementedError 
            
    # 1.拿到ev的起事时间：ev的起始时间为其所有依赖事件的结束时间和ev所在的rank上所在stream的结束时间中的最大值
    # 2.计算ev的持续时间：直接拿到或根据带宽计算事件的持续时间
    # 3.更新ev所在的rank上所在stream的结束时间，即ev的结束时间
    # 4.ev.is_done = True
    # 5.更新该GPU上的总计算时间
    # 6.更新执行过的event数
    def execute(self, ev):
        # 拿到ev的起事时间：
        # ev的起始时间为其所有依赖事件的结束时间和ev所在的rank上所在stream的结束时间中的最大值
        ev.begin = max([inev.end for inev in ev.inputs] + 
                       [self.rank_stream_endtime[ev.vt.rank][ev.stream]])
        # 计算ev的持续时间
        # 该参数默认为false，略
        if self.use_random:
            ev.dur = random.uniform(0, 1.0) # sec
            if ev.stream.startswith("P2P"):
                ev.dur = 0.5
        else:
            # 直接拿到或根据带宽计算事件的持续时间
            ev.dur = self._duration(ev) # sec
        ev.is_done = True
        # 更新ev所在的rank上所在stream的结束时间，即ev的结束时间
        # ev.end是一个函数：return self.begin + self.dur
        self.rank_stream_endtime[ev.vt.rank][ev.stream] = ev.end
        # 更新GPU上的总计算时间
        if ev.kind in ["FWD","REC","BWD"]:
            self.rank_compute[ev.vt.rank] += ev.dur
        # 更新执行过的event数
        self.cnt += 1
    
    def end(self):
        ### end time
        # 每个rank中所有stream中最大的结束时间
        self.per_rank_endtime = []
        # ODict:{stream:endtime，...}
        # stream_endtime：对应一个rank的字典，里面装着多个形如 stream:endtime 的键值对 
        for stream_endtime in self.rank_stream_endtime.values(): # can be empty rank
            # 拿到当前rank上所有stream中最大的结束时间
            if stream_endtime:
                et = max([endtime for endtime in stream_endtime.values()])
            else: # empty rank
                et = 1.E-10 # zero
            # 将拿到的时间放到List中
            self.per_rank_endtime.append(et)
        # 得到全局的endtime，即所有rank的所有stream中最大的结束时间
        self.global_endtime = max(self.per_rank_endtime)
        ### end idle ratio
        # 对每一个rank的最终endtime，计算其结束后idle的比率，即等待时间/总的执行时间(全局endtime)
        self.per_rank_endidle = [ (self.global_endtime - et) / self.global_endtime 
                                    for et in self.per_rank_endtime ]
        # 得到最大的per_rank_endidle
        self.max_endidle = max(self.per_rank_endidle)
        # 计算平均的idle比率，即每个rank上每个流中最大的结束时间后还需等待的时间/global_endtime的均值
        self.avg_endidle = sum(self.per_rank_endidle)/len(self.per_rank_endidle)
        ### compute ratio
        num_ranks = len(self.rank_compute)
        # 计算每个GPU计算时间占总时间的比例，即每一个GPU上总的计算时间/global_endtime。将这些比率加起来/GPU总数，
        # 得到平均的计算占全局总时间的比例
        self.avg_compute_to_globaltime = sum([ ct/self.global_endtime 
                                            for ct in self.rank_compute.values() ]) \
                                            / num_ranks
        # 计算每个GPU上计算时间与该GPU上总时间的比值，加在一起再除以GPU数量。即平均的计算和单个GPU上的总时间的占比
        self.avg_compute_to_ranktime = sum([ct/et 
                                            for ct, et in zip(self.rank_compute.values(), self.per_rank_endtime)]) \
                                            / num_ranks
    
def print_events(events):
    print("------- Event: ID, Inputs, Done, Name -------")
    for id, ev in events.items(): 
        assert id == ev.id
        print("%s, '%s'" % (ev, ev.name))
    print()

def print_rank_stream_events(rank_stream_events):
    for rank, per_stream_events in rank_stream_events.items(): 
        print("------- Rank %d's Stream : [Events] -------" % rank)
        for stream, events in per_stream_events.items():
            print("%s: [%s]" % (stream, ", ".join(ev.id for ev in events) ))
    print()

def debug(non_empty_gpus, events, rank_stream_events):
    non_empty_rank = 0
    for stream_events in rank_stream_events.values():
        if stream_events:
            non_empty_rank += 1
    assert non_empty_rank == non_empty_gpus
    print("[DEBUG] non_empty_gpus check: passed")
    for stream_events in rank_stream_events.values(): 
        for eq in stream_events.values():
            for e in eq:
                assert id(events[e.id]) == id(e), "%s vs %s"%(events[e.id], e)
    print("[DEBUG] same reference test: passed")
    for ev in events.values(): 
        for inev in ev.inputs:
            assert isinstance(inev, Event), "!! {} is not Event".format(inev)
    for stream_events in rank_stream_events.values():
        for eq in stream_events.values():
            for e in eq:
                for inev in e.inputs:
                    assert isinstance(inev, Event), "!! {} is not Event".format(inev)
    print("[DEBUG] inputs are events: passed")

class CofTask(object):
    """ The memory cost of a task.
        Assumption:
        0) math equation modeled cost
        1) optimizer on CPU
        2) equal ubatch size in a group, FWD or BWD
        3) vPP always use P2P for input and output
        4) ignore T
        base version:
        5) fetch W
        6) prefetch X with double buffering
        7) no prefetch X across Tasks
        8) offload Y with double buffering
        9) offload X with double buffering
    """
    def __init__(self, prof, mode, num_layers):
        self.prof = prof
        self.mode = mode
        self.R = num_layers
                
    def __call__(self, vt):
        l_start, l_end = vt.layers[0], vt.layers[-1]
        if vt.type == 'UPD':
            return 0.
            # Ctask = memory_of_pack(self.prof, 'UPD', None, l_start, l_end, offload_optim=self.offload_optim) # bytes
            # return float(Ctask)
        
        assert vt.type in ['FWD','BWD']
        # microbatch size
        ubs = vt.ubatchszs[0]
        # size
        # 输入的大小
        InputX = self.prof["XMETA"].get_bytes(ubs, l_start, interp=True)
        # 若vt的首层不是第0层，则其输出的dX的大小和传入的InputX的大小相同（第0层不会往前传dX了，所以是0）
        dX = InputX if l_start != 0 else 0. 
        W = model_size_of_pack(self.prof, 'W', l_start, l_end) # bytes
        B = model_size_of_pack(self.prof, 'B', l_start, l_end) # bytes
        # 若不是最后一层，即计算损失的层，返回l_end+1层的输入，即l_end层的输出
        # 否则返回0
        Y = self.prof["XMETA"].get_bytes(ubs, l_end+1, interp=True) \
            if l_end != self.R-1 else 0. 
        # 传进来的dY的大小和传出去的Y的大小相同
        dY = Y
        if vt.type == 'FWD':
            StashX = sum([ self.prof["XMETA"].get_bytes(ubs, l, interp=True) 
                            for l in vt.Out['X'].keys() ])       
        else:
            StashX = InputX # critierion or non-criterion
        # memory
        # 对FWD，BWD：返回整个计算过程中[l_start, l_end]产生的显存占用
        # 对UDP：返回0 ，因为优化器被卸载到cpu了
        # ❓不包含l_start的输入吗？
        Ccompute = memory_of_pack(self.prof, vt.type, ubs, l_start, l_end, interp_ubatchsize=True) # bytes
        # if vt.type == 'FWD' and vt.has_data:
        #     Ccompute -= InputX
        
        Ctask = 0.
        if self.mode == 'vPP' and vt.type == 'FWD':
            Ctask += InputX
            Ctask += Ccompute
            Ctask += Y
            Ctask += StashX    
        elif self.mode == 'vPP' and vt.type == 'BWD':
            Ctask += StashX*2 # include grad
            Ctask += dY
            Ctask += Ccompute
            Ctask += dX # ❓：上面stashX不是×2了吗，这块是不是重复了
        elif self.mode == 'vDP' and vt.type == 'FWD':
            Ctask += InputX
            Ctask += Ccompute
            Ctask += Y
            Ctask += StashX
        elif self.mode == 'vDP' and vt.type == 'BWD': 
            Ctask += StashX*2 # include grad
            Ctask += dY
            Ctask += Ccompute
            Ctask += dX
        else:
            raise NotImplementedError
        return Ctask # bytes
