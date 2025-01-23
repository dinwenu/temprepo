# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
from collections import OrderedDict as ODict

import torch
import torch.distributed as dist
from torch.autograd import Variable

from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta
from task_data_struct import Medium, vTask
import threadsafe_data_struct

class MSGStashX(object):
    """ Handles gloo send/recv of stashing X between cpu processes. 
        Assumption:
            0) distributed environment already initialized
            1) uses task graph and profiled tensor metas
            2) only CPU tensor and no grad
            3) equal microbatchsize from FWD to BWD (after UBatchSizeConverter)
            4) only FWD(non-criterion) to BWD(non-criterion) has stashX
    """
    def __init__(self, rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack', pin_memory=True, verbose=False, nvprof=False): 
        self.rank = rank
        assert dist.get_rank() == rank
        self.group = dist.new_group(ranks=None, backend='gloo')
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.nvprof = nvprof
        # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
        self._initialize(rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering)
        # 1.创建一个发送数据的辅助线程 _send_helper_thread
        #   不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
        #   send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
        #   要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
        #   新的tesnor
        # 2.为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
        #   不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
        #   即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X。📌若没有收到数据会阻塞住
        self._start_helper_threads()

    # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
    # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号，输入X是输入到哪一层的)：rank(dst_rank)}
    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()
        # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
        # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号)：rank(dst_rank)}
        for vt in rtasks[self.rank]:
            # 
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.Out['X']: # FIX 4)
                for l,m in vt.Out['X'].items(): 
                    # 若媒介为MSG，
                    if m.medium == "MSG":
                        send_ranks[l] = m.rank # dst_rank
        return send_ranks

    # 对当前rank中除第一个BWD任务以外的BWD任务，若接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对
    # 添加到两个字典中并返回
    # 返回两个字典：{ layer_id: src_rank }，{ src_rank: [layer_id] }
    def _find_recv_ranks_layers(self, rtasks):
        recv_ranks = ODict() # { layer_id: src_rank } # can include self.rank
        recv_layers = ODict() # { src_rank: [layer_id] } # can include self.rank
        for vt in rtasks[self.rank]:
            if vt.type == 'BWD' and (not vt.has_criterion) and vt.In['X']: # FIX 4)
                for l,m in vt.In['X'].items(): 
                    if m.medium == "MSG":
                        recv_ranks[l] = m.rank # src_rank
                        if m.rank not in recv_layers:
                            recv_layers[m.rank] = []
                        recv_layers[m.rank].append(l)
        return recv_ranks, recv_layers

    # 在当前rank上寻找除最后一个fwd任务外的其他fwd任务
    # 将要发送的X对应的layer和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,bwd_ubsize), ...]
    # 
    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []
        # 对当前rank中的所有task，
        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            # 
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)
                # 遍历BWD的microbatch size列表
                for u in self.ubatchszs:
                    # 对该vt上每一个要发送的X，若媒介为MSG，将(layer_id,bwd_ubsize)作为元组加入到order list
                    for l,m in vt.Out['X'].items(): 
                        if m.medium == "MSG":
                            order.append((l,u)) # can include self queue
        return order
    
    # 找到发送输入X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
    # ubatchsize_bwd。{src_rank：[(l,u),...], ...}
    # 该字典是按rank号排序的
    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }
        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)
                    for u in self.ubatchszs:
                        for l,m in vt.Out['X'].items(): 
                            if m.medium == "MSG":
                                # 若src_rank的发送X的task的目标rank等于当前rank，将src_rank上对应该X的layer_id和ubatchsize_bwd
                                # 加入到该src_rank的列表中。src_rank：[(l,u),...]
                                if m.rank == self.rank: # can include self queue # and vt.rank != self.rank: 
                                    if src_rank not in orders:
                                        orders[src_rank] = []
                                    orders[src_rank].append((l,u))
        return orders
    
    # 1.找到当前rank上发送X的任务（媒介为MSG），构建一个字典：，{l(层号，输入X对应的那一层)：rank(dst_rank)}
    # 2.若上一步生成的字典不为空，即存在MSGX任务。实例化一个线程安全的字典，用于在线程间发送接收数据
    #   2.1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
    #   2.2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
    # 3.从当前rank的BWD任务中，把那些接收输入X的任务的信息提取出来并返回，即收集当前rank上的MSGinX信息，包含两个字典：
    #   self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
    #   self.recv_layers = { src_rank: [layer_id] } # can include self.rank  
    # 4.遍历src_rank，为这些源rank实例化一个线程安全的字典，并进行初始化。
    #   4.1.若src_rank就是当前rank，会进行一些额外的保险操作，以确保发送和接收是一一对应的
    # 5.将所有src_rank添加到 recv_tags 字典中：{src_rank：src_rank, ...}
    # 6.self.ubatchszs = ubatchszs_bwd
    # 7.
    #   7.1.将要发送的X对应的layer id和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,ubatchsize), ...]
    #   7.2.找到发送X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
    #       ubatchsize_bwd。{src_rank：[(l,u),...], ...}
    def _initialize(self, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack'):
        """
        Argument: ordering = the sending order (this ordering is self contained)
        """
        # setup send dict # { layer_id: dst_rank } # can include self.rank
        # 1.找到当前rank上发送X的任务（媒介为MSG），构建一个字典：，{l(层号，输入X对应的那一层)：rank(dst_rank)}
        # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
        # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号，输入X是输入到哪一层的)：rank(dst_rank)}
        self.send_ranks = self._find_send_ranks(rtasks)
        # 若存在要发送的任务
        if self.send_ranks:
            # 实例化一个线程安全的字典，该类中的字典只有一个线程能向其添加或删除对象
            self.send_dict = threadsafe_data_struct.OrderedDictionary() # between main and send threads
            # 1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
            # 2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
            self.send_dict.init_layer_ids(list(self.send_ranks.keys()))
            self.send_tag = self.rank
            if self.verbose: print_str = "[MSGStashX]\nrank{} set up send_dict=\n{}\n".format(self.rank, self.send_dict)
        else:
            self.send_dict = None
            if self.verbose: print_str = "[MSGStashX]\nrank{} has NO send job\n".format(self.rank)

        # setup recv dicts
        # 对当前rank中除第一个BWD任务以外的BWD任务，若接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对
        # 添加到两个字典中并返回
        # 返回两个字典：
        # self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
        # self.recv_layers = { src_rank: [layer_id] } # can include self.rank
        self.recv_ranks, self.recv_layers = self._find_recv_ranks_layers(rtasks)
        # 
        self.recv_dicts = ODict() # { src_rank: the thread safe dict } # can include self.rank
        # 对每一个发送X到当前rank的src_rank:
        # 对MSGX实例：对向第一个BWD任务所在的rank的MSGX实例发送Y的 src_rank，执行内部逻辑
        for r in sorted(set(self.recv_layers.keys())):
            # 若接受的X的来源就是自己这个rank，
            if r == self.rank: # loopback to self dict
                # 为 rank r 实例化一个线程安全的字典
                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary() # between send and main threads

                # =====相比else多出的逻辑，即多了一步检查操作，确保send_ranks中具有和从BWD任务中提取出来的接收X的layer id同样的layer id====
                # =====即确保同一个rank上的发送和接收是一一对应的====
                # 取出当前rank上对应源rank的layer id并排序，即接收X的layer的layer id
                self_layer_ids = sorted(self.recv_layers[self.rank])
                # l:输入X是输入到哪一层的，dst：目标rank
                # 保险操作：从send_ranks中把目标rank和当前rank相同的提取出来，确保其layer_id存在于接收X的层的list中
                for l,dst in self.send_ranks.items():
                    if dst == self.rank:
                        assert l in self_layer_ids

                # 1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
                # 2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
                self.recv_dicts[r].init_layer_ids(self_layer_ids)
            # 否则就是从其他rank传进来的，同样向接收字典recv_dicts中添加一个键值对，{来源rank：线程安全字典}
            # 并为该线程安全字典执行初始化流程
            else: # recv from other rank
                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary() # between recv and main threads
                self.recv_dicts[r].init_layer_ids(sorted(self.recv_layers[r]))
        #
        # 
        self.recv_tags = ODict() # { src_rank : tag }
        # 将所有src_rank添加到 recv_tags 字典中：{src_rank：src_rank, ...}
        for src_rank in sorted(self.recv_layers.keys()):
            self.recv_tags[src_rank] = src_rank
        if self.verbose:
            if self.recv_dicts:
                print_str += "rank{} set up recv_dicts (src_ranks={})\n".format(self.rank, list(self.recv_dicts.keys()))
                for src_rank, recv_dict in self.recv_dicts.items():
                    print_str += recv_dict.__repr__(title="thread-safe dict (%s)"%("self queue" if src_rank == self.rank else "src_rank=%d"%src_rank )) + "\n"
            else: # empty
                print_str += "rank{} has NO recv job\n".format(self.rank)
        # setup number of ubatches in both sending and recving
        assert isinstance(ubatchszs_bwd,list)
        self.ubatchszs = ubatchszs_bwd
        if self.verbose: print_str += "rank{} set up ubatchszs = {}\n".format(self.rank, self.ubatchszs)
        # setup send and recv order
        self.ordering = ordering
        if ordering == 'layer-by-layer':
            self.send_order = None
            self.recv_orders = None

        # 7.
        # 默认执行这个
        elif ordering == 'pack-by-pack':    
            # 将要发送的X对应的layer id和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,ubatchsize), ...]
            # 发送顺序：先是vt之间有序，vt内，按各个BWD ubsize排序，对每个ubsize，按vt要发送的层排序
            # [(vt1's layer1,u1),(vt1's layer1,u1),(vt2's layer1,u2),(vt2's layer1,u2),...]
            self.send_order = self._find_send_order(rtasks)
            # 找到发送X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
            # ubatchsize_bwd。{src_rank：[(l,u),...], ...}
            # 该字典是按rank号排序的
            self.recv_orders = self._find_recv_order(rtasks)
            print(f"rank:{self.rank}, recv orders:{self.recv_orders}")
            if self.verbose:
                print_str += "rank{} set up send_order = {}\n".format(self.rank, self.send_order)
                print_str += "rank{} set up recv_orders = {}\n".format(self.rank, self.recv_orders)
        else:
            raise ValueError
        # setup X_names
        self.layer_x_names = layer_x_names # { layer_id: X_names } # TODO: less stashing X after checking Identity chain --> stash X is always needed
        self.xmeta = xmeta # dictionary of TensorInfo
        
        if self.verbose: print(print_str)

    # 1.创建一个发送数据的辅助线程 _send_helper_thread
    #   不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
    #   send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
    #   要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
    #   新的tesnor
    # 2.为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
    #   不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
    #   即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X。📌若没有收到数据会阻塞住
    def _start_helper_threads(self):
        """ Start helper communication threads, one for each queue. """
        # Setup send thread
        cnt_send_thd = 0
        # { layer_id: dst_rank，... }
        if self.send_dict is not None:
            # 创建一个发送数据的辅助线程 _send_helper_thread
            # target 参数指定了线程要运行的目标函数，即在新线程中执行的函数
            # 
            # 不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
            # send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
            # 要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
            # 新的tesnor
            helper_thread = threading.Thread(target=self._send_helper_thread)
            helper_thread.daemon = True
            # 启动这个线程
            helper_thread.start()
            cnt_send_thd += 1
        # Setup recv thread for each queue (excluding self queue)
        cnt_recv_thd = 0
        # 为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
        for src_rank in self.recv_dicts.keys():
            if src_rank != self.rank:
                # 不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
                # 即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X
                helper_thread = threading.Thread(target=self._recv_helper_thread, args=(src_rank,))
                helper_thread.daemon = True
                helper_thread.start()
                cnt_recv_thd += 1
        # print("[MSGStashX] rank{} started send_helper_threadx{} & recv_helper_threadx{}".format(self.rank,cnt_send_thd,cnt_recv_thd))
    
    # 
    def _send_helper_thread(self):
        """ This method is to be executed from a helper daemon thread. """
        assert self.send_dict is not None # must be non-empty
        if self.ordering == "layer-by-layer":
            while True: # each tasks iteration
                for layer_id in self.send_dict.layer_ids: # in-order of FWD layers
                    for ubs in self.ubatchszs: 
                        named_tensors = self.send_dict.remove(layer_id)
                        dst_rank = self.send_ranks[layer_id]
                        if dst_rank == self.rank:
                            self._send2self(layer_id, named_tensors, self.pin_memory)
                        else:
                            self._send(layer_id, named_tensors, dst_rank)

        # 不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在send_dict中，
        # 不断尝试通过layer_id取send_dict中保存的tensor，发送出去
        # 要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
        # 新的tesnor
        elif self.ordering == "pack-by-pack":
            assert len(self.send_order) == len(self.send_dict.layer_ids) * len(self.ubatchszs)
            while True: # each tasks iteration
                for layer_id, _ in self.send_order: # [(layer_id, ubatchsize)，...]
                    # print("[MSGStashX] rank{} wait L{}, send_dict=\n{}\n".format(self.rank, layer_id, self.send_dict))

                    # 从 self.odict[layer_id] 列表中弹出并返回第一个元素
                    # { layer_id: [u1's {"name1": tensor1, "name2": [tensor2]}, u2's {}, ... ] }
                    named_tensors = self.send_dict.remove(layer_id)
                    # 目标rank
                    dst_rank = self.send_ranks[layer_id]
                    # 若要发送到的rank就是当前rank，就是要发送给自己
                    if dst_rank == self.rank:
                        self._send2self(layer_id, named_tensors, self.pin_memory)

                    # 调用 dist.send 方法将张量发送到dst_rank
                    else:
                        self._send(layer_id, named_tensors, dst_rank)
        else:
            raise ValueError
    
    # 通过将tesnor放到固定内存中，达到rank内互传的效果
    def _send2self(self, layer_id, named_tensors, pin_memory=True):
        """ Helper thread sends tensor to itself rank. """
        if pin_memory: # move tensors to pin memory if not already pinned.
            for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
                if isinstance(tensor, (torch.Tensor,Variable)):
                    assert not tensor.is_cuda and not tensor.requires_grad
                    named_tensors[name] = tensor.pin_memory() # If the tensor is not pinned, returns a new copy in pinned memory. Else, returns itself (already pinned).
                elif isinstance(tensor, (float,int)):
                    continue
                elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                    pinned_tensor = []
                    for t in tensor:
                        assert not t.is_cuda and not t.requires_grad
                        pinned_tensor.append(t.pin_memory())
                    named_tensors[name] = pinned_tensor
                else:
                    raise ValueError("unknown tensor={}".format(tensor))
        self.recv_dicts[self.rank].add(layer_id, named_tensors) 
        # print("[MSGStashX] rank{} _send2self enqueued (X{},{})".format(self.rank, layer_id, list(named_tensors.keys())))

    # 调用 dist.send 方法将张量发送到dst_rank
    def _send(self, layer_id, named_tensors, dst_rank):
        """ Helper thread sends tensor by calling dist.send(). """
        if self.nvprof: nvtx_range_push("__L{} MSG to dst{}".format(layer_id,dst_rank)) 
        # print("[MSGStashX] rank{} _sending L{} to dst{}".format(self.rank, layer_id, dst_rank))
        # named_metas = self.xmeta.get(1, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        for (name,tensor), name2 in zip(named_tensors.items(), self.layer_x_names[layer_id]): # { name: tensor/const, name: [tensors] }
            assert name == name2
            if isinstance(tensor, (torch.Tensor,Variable)):
                assert not tensor.is_cuda and not tensor.requires_grad
                dist.send(tensor, dst_rank, self.group, self.send_tag)
            elif isinstance(tensor, (float,int)):
                dist.send(torch.tensor(tensor), dst_rank, self.group, self.send_tag)
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                for t in tensor:
                    assert not t.is_cuda and not t.requires_grad
                    dist.send(t, dst_rank, self.group, self.send_tag)
            else:
                raise ValueError("unknown tensor={}".format(tensor))
        if self.nvprof: nvtx_range_pop() 
        # print("[MSGStashX] rank{} _sent L{} to dst{}".format(self.rank, layer_id, dst_rank))
            
        
    # 不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
    # 即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X，将读取到的named_tensors连同其layer_id
    # 构成一个元组加入到 self.recv_dicts[src_rank] 这个线程安全字典中
    def _recv_helper_thread(self, src_rank):
        """ This method is to be executed from a helper daemon thread. """
        assert src_rank != self.rank
        if self.ordering == "layer-by-layer":
            while True: # each tasks iteration
                for layer_id in self.recv_dicts[src_rank].layer_ids: # in-order of FWD layers
                    for ubs in self.ubatchszs: 
                        named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory) 
                        self.recv_dicts[src_rank].add(layer_id, named_tensors)

        # 显然执行这个
        elif self.ordering == "pack-by-pack":
            if self.verbose: print("rank{}: _recv_helper_thread(src_rank={}): self.recv_orders={}, self.recv_dicts={}".format(self.rank, src_rank, self.recv_orders, self.recv_dicts))
            print(f"rank:{self.rank}, {self.__class__.__name__}")
            print(f"rank:{self.rank}, {self.ubatchszs}")
            print(f"rank:{self.rank}, {self.recv_orders[src_rank]}")
            print(f"rank:{self.rank}, {self.recv_dicts[src_rank].layer_ids}")
            assert len(self.recv_orders[src_rank]) == len(self.recv_dicts[src_rank].layer_ids) * len(self.ubatchszs)
            while True: # each tasks iteration
                for layer_id, ubs in self.recv_orders[src_rank]: # [(layer_id, ubatchsize)]
                    # 调用dist.recv函数接收tesnor，接收的tensor放入一个ODict中返回
                    # 📌若没有收到数据会阻塞住
                    named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory) 
                    # 将收到的数据添加到 recv_dicts[src_rank] 这个线程安全字典中
                    self.recv_dicts[src_rank].add(layer_id, named_tensors)
        else:
            raise ValueError

    # 调用dist.recv函数接收tensor，接收的tensor放入一个ODict中返回
    def _recv(self, layer_id, ubatchsize, src_rank, pin_memory=True):
        """ Helper thread receives tensor by calling dist.recv(). """ 
        # print("[rank{}]\tmsg_handler._send: entered".format(self.rank))

        # 获取要接收的X的元数据，需要根据其大小和类型生成tensor
        named_metas = self.xmeta.get(ubatchsize, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        #
        named_tensors = ODict()
        # name即layer_id层的输入名称
        for name in self.layer_x_names[layer_id]:
            meta = named_metas[name]
            if isinstance(meta, TensorMeta):
                tensor = torch.empty(meta.shape, dtype=meta.dtype, device="cpu", pin_memory=pin_memory)
                # 将数据存储在创建的张量中
                # 若没有收到数据会阻塞住
                # print(f"[rank{self.rank}]正在接收layer{layer_id}")
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                named_tensors[name] = tensor 
            elif isinstance(meta, ConstMeta):
                tensor = torch.tensor(meta.const, device="cpu", pin_memory=pin_memory)
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                named_tensors[name] = tensor.item() # convert a 0-dim tensor to a python number
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                named_tensors[name] = []
                for m in meta:
                    tensor = torch.empty(m.shape, dtype=m.dtype, device="cpu", pin_memory=pin_memory)
                    dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                    named_tensors[name].append(tensor)
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[rank{}]\tmsg_handler._send: tensor(shape={}, dtype={}) sent".format(self.rank, tensor.shape, tensor.dtype))
        return named_tensors
    
    # Call by upstream thread. Nonblocking send. 
    # 向 send_dict 这个线程安全字典中的 odict[layer_id] 这个list添加：self.odict[layer_id].append(named_tensors)
    def isend(self, layer_id, named_tensors):
        ''' Call by upstream thread. Nonblocking send. 
            The same API for two ordering of 'layer-by-layer' and 'pack-by-pack' '''
        self.send_dict.add(layer_id, named_tensors) # tuple uses reference to tensor

    # 1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
    # 2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
    #   即一个 （name, tensor）。若内部没有tensor显然会被阻塞住(wait)
    def recv(self, layer_id):
        ''' Call by downstream thread. Blocking recv. Return named_tensors. '''
        src_rank = self.recv_ranks[layer_id] # { layer_id: src_rank } # can include self.rank
        # 从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
        # 若内部没有tensor显然会被阻塞住(wait)
        return self.recv_dicts[src_rank].remove(layer_id) # tuple uses reference to tensor    
    
    def has_no_send(self):
        return self.send_dict is None
    
    def has_no_recv(self):
        return False if self.recv_dicts else True
    
    
class MSGX(MSGStashX):
    """ Handles gloo send/recv of Y/dX between cpu processes. 
        NOTE: Tentative for last fwd task to bwd criterion
        TODO: 1) To support all Y/dX; 2) replace input data structure to queue
    """
    def __init__(self, rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack', pin_memory=True, verbose=False, nvprof=False): 
        super().__init__(rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering, pin_memory, verbose, nvprof)

    #############################################################################################################
    # 对下面两个确定rank的方法的总结：可见是这样一种场景，只有最后一个FWDvt向第一个BWDvt发送Y，且在cpu上发送，即媒介为MSG
    # 并没有管dX
    #############################################################################################################
        
    # 总结：找到要接收X的目标rank，即首个BWDvt所在rank，和该vt首层组成一个键值对
    #
    # 在当前rank上找最后一个FWD vt，若有，记录该vt输出Y的目标rank（输出媒介必须是MSG才会记录）
    #
    # 若当前rank中存在最后一个FWD任务且输出Y的媒介为MSG，向send_ranks字典添加一个键值对：
    # {l(层号，📌接收输出 Y 的那一层，是l+1不是l)：rank(dst_rank)}
    # 返回 send_ranks 字典
    # 📌分析：vPP任务的最后一个fwd任务输出Y的媒介为P2P，这里的send_ranks应该是空的
    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()
        for vt in rtasks[self.rank]:
            # 若是最后一个fwd任务，且会输出一个Y
            if vt.is_last_fwd and vt.Out['Y']:
                # l 为输出 Y 的那一层的layer_id
                l = vt.layers[-1]
                m = vt.Out['Y'][l]
                # m中保存的rank本来就是 dst_rank
                # 📌分析24/10/8：不管是vPP还是vDP，Out['Y'][l]这个媒介就不可能是MSG
                # ❌：以上回答错误，有这种情况，即前后想microbatch大小不一样
                print(f"MSGX实例, m.medium:{m.medium}")
                if m.medium == "MSG":
                    send_ranks[l+1] = m.rank # dst_rank
        if self.verbose: print("[MSGX] found send_ranks={}".format(send_ranks))
        return send_ranks

    # 总结：找到产生X并发送它的来源rank，即最后一个FWDvt所在的rank，和首个BWDvt首层的层号组成一个键值对
    #
    # 在当前rank上找第一个BWD vt，若有，记录给该vt发送输入X的来源rank（输入媒介必须是MSG才会记录）
    #
    # 若当前rank中存在第一个BWD任务，且接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对添加到两个字典中并返回
    # 返回两个字典：
    # self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
    # self.recv_layers = { src_rank: [layer_id] } # can include self.rank
    # 📌分析：vPP第一个BWD任务接收输入X的媒介应该是P2P,这里的两个字典应该是空的
    def _find_recv_ranks_layers(self, rtasks):
        recv_ranks = ODict() # { layer_id: src_rank } # can include self.rank
        recv_layers = ODict() # { src_rank: [layer_id] } # can include self.rank
        for vt in rtasks[self.rank]:
            if vt.type == 'BWD' and vt.has_criterion and vt.In['X']:
                l = vt.layers[0]
                m = vt.In['X'][l]
                if m.medium == "MSG":
                    recv_ranks[l] = m.rank # src_rank
                    if m.rank not in recv_layers:
                        recv_layers[m.rank] = []
                    recv_layers[m.rank].append(l)
        if self.verbose: print("[MSGX] found recv_ranks={}, recv_layers={}".format(recv_ranks, recv_layers))
        return recv_ranks, recv_layers

    #############################################################################################################
    # 对下面两个order方法的总结：对一下两个方法的命名和上面两个方法的命名是反过来的。_find_send_ranks虽然命名方式是
    # send，但其实找的是接收方，即目标rank。但下面这个_find_send_order就是确定发送的顺序。
    #############################################################################################################

    # 为最后一个fwd任务所在rank的MSGX实例配置发送输出Y的顺序（最后一个fwd任务并非真正的最后一个，真正的最后一个同时也是第一个bwd任务）
    #
    # 在当前rank上寻找最后一个fwd任务，若其输出Y的媒介为MSG，将“要接收输出Y”的层l和每一个ubatchszs_bwd组成一个元组
    # 加入到order列表中，[(接收Y的layer_id,ubatchsize), ...]，最后返回该列表
    # 📌分析：vPP的最后一个fwd任务输出Y的媒介为P2P，返回的 orders 应该是空的
    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []
        # 在当前rank上寻找最后一个fwd任务，若其输出Y的媒介为MSG，将要接收输出Y的层l和每一个ubatchszs_bwd组成一个元组
        # 加入到order列表中
        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            if vt.is_last_fwd:
                for u in self.ubatchszs:
                    l = vt.layers[-1]
                    m = vt.Out['Y'][l]
                    if m.medium == "MSG":
                        order.append((l+1,u))
        if self.verbose: print("[MSGX] found send order={}".format(order))
        return order
    
    # 为第一个bwd任务所在rank的MSGX实例配置接收（第一个fwd任务输出的Y的）顺序
    #
    # 在所有rank的任务上寻找最后一个fwd任务，若该任务输出的Y的媒介为MSG，且目标rank就是当前rank，则将
    # (l+1(要接收Y的层)，u)添加到src_rank对应的列表中。{src_rank：[(接收Y的l,u),...], ...}
    # 📌分析：vPP的最后一个fwd任务输出Y的媒介为P2P，返回的 orders 应该是空的
    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }
        # 在所有rank的任务上寻找最后一个fwd任务，若该任务输出的Y的媒介为MSG，且目标rank就是当前rank，则将
        # (l+1(要接收Y的层)，u)添加到src_rank对应的列表中
        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.is_last_fwd:
                    for u in self.ubatchszs:
                        l = vt.layers[-1]
                        m = vt.Out['Y'][l]
                        # 
                        if m.medium == "MSG":
                            # 若最后一个fwd任务发送Y的目标rank就是当前rank
                            if m.rank == self.rank: # can include self queue # and vt.rank != self.rank: 
                                if src_rank not in orders:
                                    orders[src_rank] = []
                                # 
                                orders[src_rank].append((l+1,u))
        if self.verbose: print("[MSGX] rank{} found recv orders={}".format(self.rank, orders))
        return orders
    

# 专门用于最后一个FWDvt→首个BWDvt在一个GPU情况下的底层存储结构和发送接收（实际上就是自己和自己发送）
# 其实主要还是实行在前后向microbatch大小不一致时切割X的功能
# 与harmony原版代码的区别在于，两个send方法中对tensor是否在cuda上的检查变为要确保tensor在cuda上
# 因为现在的tensro就是在cuda上的，不需要进行卸载
class MSGStashX_2(object):
    """ Handles gloo send/recv of stashing X between cpu processes. 
        Assumption:
            0) distributed environment already initialized
            1) uses task graph and profiled tensor metas
            2) only CPU tensor and no grad
            3) equal microbatchsize from FWD to BWD (after UBatchSizeConverter)
            4) only FWD(non-criterion) to BWD(non-criterion) has stashX
    """
    def __init__(self, ubscvt, rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack', pin_memory=True, verbose=False, nvprof=False): 
        self.rank = rank
        assert dist.get_rank() == rank
        self.group = dist.new_group(ranks=None, backend='gloo')
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.nvprof = nvprof

        self.converted_ubwd = ubscvt.find_converted(0)
        print(f"转化后的ubwd列表为:{self.converted_ubwd}")
        assert type(self.converted_ubwd) is list
        self.send_and_recv_num = len(self.converted_ubwd)
        print(f"发送/接收次数:{self.send_and_recv_num}")

        # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
        self._initialize(rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering)
        # 1.创建一个发送数据的辅助线程 _send_helper_thread
        #   不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
        #   send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
        #   要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
        #   新的tesnor
        # 2.为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
        #   不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
        #   即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X。📌若没有收到数据会阻塞住
        self._start_helper_threads()

    # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
    # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号，输入X是输入到哪一层的)：rank(dst_rank)}
    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()
        # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
        # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号)：rank(dst_rank)}
        for vt in rtasks[self.rank]:
            # 
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.Out['X']: # FIX 4)
                for l,m in vt.Out['X'].items(): 
                    # 若媒介为MSG，
                    if m.medium == "MSG":
                        send_ranks[l] = m.rank # dst_rank
        return send_ranks

    # 对当前rank中除第一个BWD任务以外的BWD任务，若接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对
    # 添加到两个字典中并返回
    # 返回两个字典：{ layer_id: src_rank }，{ src_rank: [layer_id] }
    def _find_recv_ranks_layers(self, rtasks):
        recv_ranks = ODict() # { layer_id: src_rank } # can include self.rank
        recv_layers = ODict() # { src_rank: [layer_id] } # can include self.rank
        for vt in rtasks[self.rank]:
            if vt.type == 'BWD' and (not vt.has_criterion) and vt.In['X']: # FIX 4)
                for l,m in vt.In['X'].items(): 
                    if m.medium == "MSG":
                        recv_ranks[l] = m.rank # src_rank
                        if m.rank not in recv_layers:
                            recv_layers[m.rank] = []
                        recv_layers[m.rank].append(l)
        return recv_ranks, recv_layers

    # 在当前rank上寻找除最后一个fwd任务外的其他fwd任务
    # 将要发送的X对应的layer和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,bwd_ubsize), ...]
    # 
    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []
        # 对当前rank中的所有task，
        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            # 
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)
                # 遍历BWD的microbatch size列表
                for u in self.ubatchszs:
                    # 对该vt上每一个要发送的X，若媒介为MSG，将(layer_id,bwd_ubsize)作为元组加入到order list
                    for l,m in vt.Out['X'].items(): 
                        if m.medium == "MSG":
                            order.append((l,u)) # can include self queue
        return order
    
    # 找到发送输入X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
    # ubatchsize_bwd。{src_rank：[(l,u),...], ...}
    # 该字典是按rank号排序的
    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }
        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)
                    for u in self.ubatchszs:
                        for l,m in vt.Out['X'].items(): 
                            if m.medium == "MSG":
                                # 若src_rank的发送X的task的目标rank等于当前rank，将src_rank上对应该X的layer_id和ubatchsize_bwd
                                # 加入到该src_rank的列表中。src_rank：[(l,u),...]
                                if m.rank == self.rank: # can include self queue # and vt.rank != self.rank: 
                                    if src_rank not in orders:
                                        orders[src_rank] = []
                                    orders[src_rank].append((l,u))
        return orders
    
    # 1.找到当前rank上发送X的任务（媒介为MSG），构建一个字典：，{l(层号，输入X对应的那一层)：rank(dst_rank)}
    # 2.若上一步生成的字典不为空，即存在MSGX任务。实例化一个线程安全的字典，用于在线程间发送接收数据
    #   2.1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
    #   2.2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
    # 3.从当前rank的BWD任务中，把那些接收输入X的任务的信息提取出来并返回，即收集当前rank上的MSGinX信息，包含两个字典：
    #   self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
    #   self.recv_layers = { src_rank: [layer_id] } # can include self.rank  
    # 4.遍历src_rank，为这些源rank实例化一个线程安全的字典，并进行初始化。
    #   4.1.若src_rank就是当前rank，会进行一些额外的保险操作，以确保发送和接收是一一对应的
    # 5.将所有src_rank添加到 recv_tags 字典中：{src_rank：src_rank, ...}
    # 6.self.ubatchszs = ubatchszs_bwd
    # 7.
    #   7.1.将要发送的X对应的layer id和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,ubatchsize), ...]
    #   7.2.找到发送X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
    #       ubatchsize_bwd。{src_rank：[(l,u),...], ...}
    def _initialize(self, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack'):
        """
        Argument: ordering = the sending order (this ordering is self contained)
        """
        # setup send dict # { layer_id: dst_rank } # can include self.rank
        # 1.找到当前rank上发送X的任务（媒介为MSG），构建一个字典：，{l(层号，输入X对应的那一层)：rank(dst_rank)}
        # 在当前rank的task中寻找：类型为FWD、非最后一个loss计算任务、且需要输出输入的任务
        # 若存在这样的任务，且输出媒介为MSG，将目标rank添加到字典中，{l(层号，输入X是输入到哪一层的)：rank(dst_rank)}
        self.send_ranks = self._find_send_ranks(rtasks)
        # 若存在要发送的任务
        if self.send_ranks:
            # 实例化一个线程安全的字典，该类中的字典只有一个线程能向其添加或删除对象
            self.send_dict = threadsafe_data_struct.OrderedDictionary() # between main and send threads
            # 1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
            # 2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
            self.send_dict.init_layer_ids(list(self.send_ranks.keys()))
            self.send_tag = self.rank
            if self.verbose: print_str = "[MSGStashX]\nrank{} set up send_dict=\n{}\n".format(self.rank, self.send_dict)
        else:
            self.send_dict = None
            if self.verbose: print_str = "[MSGStashX]\nrank{} has NO send job\n".format(self.rank)

        # setup recv dicts
        # 对当前rank中除第一个BWD任务以外的BWD任务，若接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对
        # 添加到两个字典中并返回
        # 返回两个字典：
        # self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
        # self.recv_layers = { src_rank: [layer_id] } # can include self.rank
        self.recv_ranks, self.recv_layers = self._find_recv_ranks_layers(rtasks)
        # 
        self.recv_dicts = ODict() # { src_rank: the thread safe dict } # can include self.rank
        # 对每一个发送X到当前rank的src_rank:
        # 对MSGX实例：对向第一个BWD任务所在的rank的MSGX实例发送Y的 src_rank，执行内部逻辑
        for r in sorted(set(self.recv_layers.keys())):
            # 若接受的X的来源就是自己这个rank，
            if r == self.rank: # loopback to self dict
                # 为 rank r 实例化一个线程安全的字典
                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary() # between send and main threads

                # =====相比else多出的逻辑，即多了一步检查操作，确保send_ranks中具有和从BWD任务中提取出来的接收X的layer id同样的layer id====
                # =====即确保同一个rank上的发送和接收是一一对应的====
                # 取出当前rank上对应源rank的layer id并排序，即接收X的layer的layer id
                self_layer_ids = sorted(self.recv_layers[self.rank])
                # l:输入X是输入到哪一层的，dst：目标rank
                # 保险操作：从send_ranks中把目标rank和当前rank相同的提取出来，确保其layer_id存在于接收X的层的list中
                for l,dst in self.send_ranks.items():
                    if dst == self.rank:
                        assert l in self_layer_ids

                # 1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
                # 2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
                self.recv_dicts[r].init_layer_ids(self_layer_ids)
            # 否则就是从其他rank传进来的，同样向接收字典recv_dicts中添加一个键值对，{来源rank：线程安全字典}
            # 并为该线程安全字典执行初始化流程
            else: # recv from other rank
                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary() # between recv and main threads
                self.recv_dicts[r].init_layer_ids(sorted(self.recv_layers[r]))
        #
        # 
        self.recv_tags = ODict() # { src_rank : tag }
        # 将所有src_rank添加到 recv_tags 字典中：{src_rank：src_rank, ...}
        for src_rank in sorted(self.recv_layers.keys()):
            self.recv_tags[src_rank] = src_rank
        if self.verbose:
            if self.recv_dicts:
                print_str += "rank{} set up recv_dicts (src_ranks={})\n".format(self.rank, list(self.recv_dicts.keys()))
                for src_rank, recv_dict in self.recv_dicts.items():
                    print_str += recv_dict.__repr__(title="thread-safe dict (%s)"%("self queue" if src_rank == self.rank else "src_rank=%d"%src_rank )) + "\n"
            else: # empty
                print_str += "rank{} has NO recv job\n".format(self.rank)
        # setup number of ubatches in both sending and recving
        assert isinstance(ubatchszs_bwd,list)
        self.ubatchszs = ubatchszs_bwd
        if self.verbose: print_str += "rank{} set up ubatchszs = {}\n".format(self.rank, self.ubatchszs)
        # setup send and recv order
        self.ordering = ordering
        if ordering == 'layer-by-layer':
            self.send_order = None
            self.recv_orders = None

        # 7.
        # 默认执行这个
        elif ordering == 'pack-by-pack':    
            # 将要发送的X对应的layer id和每一个ubatchszs_bwd组成一个元组，放到一个list中返回，[(layer_id,ubatchsize), ...]
            # 发送顺序：先是vt之间有序，vt内，按各个BWD ubsize排序，对每个ubsize，按vt要发送的层排序
            # [(vt1's layer1,u1),(vt1's layer1,u1),(vt2's layer1,u2),(vt2's layer1,u2),...]
            self.send_order = self._find_send_order(rtasks)
            # 找到发送X到当前rank的任务，放进字典中，字典的key为src_rank，val为一个list，装着该rank上对应X的layer_id和
            # ubatchsize_bwd。{src_rank：[(l,u),...], ...}
            # 该字典是按rank号排序的
            self.recv_orders = self._find_recv_order(rtasks)
            if self.verbose:
                print_str += "rank{} set up send_order = {}\n".format(self.rank, self.send_order)
                print_str += "rank{} set up recv_orders = {}\n".format(self.rank, self.recv_orders)
        else:
            raise ValueError
        # setup X_names
        self.layer_x_names = layer_x_names # { layer_id: X_names } # TODO: less stashing X after checking Identity chain --> stash X is always needed
        self.xmeta = xmeta # dictionary of TensorInfo
        
        if self.verbose: print(print_str)

    # 1.创建一个发送数据的辅助线程 _send_helper_thread
    #   不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
    #   send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
    #   要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
    #   新的tesnor
    # 2.为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
    #   不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
    #   即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X。📌若没有收到数据会阻塞住
    def _start_helper_threads(self):
        """ Start helper communication threads, one for each queue. """
        # Setup send thread
        cnt_send_thd = 0
        # { layer_id: dst_rank，... }
        if self.send_dict is not None:
            # 创建一个发送数据的辅助线程 _send_helper_thread
            # target 参数指定了线程要运行的目标函数，即在新线程中执行的函数
            # 
            # 不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在
            # send_dict中，不断尝试通过layer_id取send_dict中保存的tensor，发送出去
            # 要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
            # 新的tesnor
            helper_thread = threading.Thread(target=self._send_helper_thread)
            helper_thread.daemon = True
            # 启动这个线程
            helper_thread.start()
            cnt_send_thd += 1
        # Setup recv thread for each queue (excluding self queue)
        cnt_recv_thd = 0
        # 为每个 src_rank，即给当前rank发送X的rank，创建一个接收数据的辅助线程 _recv_helper_thread
        for src_rank in self.recv_dicts.keys():
            if src_rank != self.rank:
                # 不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
                # 即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X
                helper_thread = threading.Thread(target=self._recv_helper_thread, args=(src_rank,))
                helper_thread.daemon = True
                helper_thread.start()
                cnt_recv_thd += 1
        # print("[MSGStashX] rank{} started send_helper_threadx{} & recv_helper_threadx{}".format(self.rank,cnt_send_thd,cnt_recv_thd))
    
    # 
    def _send_helper_thread(self):
        """ This method is to be executed from a helper daemon thread. """
        assert self.send_dict is not None # must be non-empty
        if self.ordering == "layer-by-layer":
            while True: # each tasks iteration
                for layer_id in self.send_dict.layer_ids: # in-order of FWD layers
                    for ubs in self.ubatchszs: 
                        named_tensors = self.send_dict.remove(layer_id)
                        dst_rank = self.send_ranks[layer_id]
                        if dst_rank == self.rank:
                            self._send2self(layer_id, named_tensors, self.pin_memory)
                        else:
                            self._send(layer_id, named_tensors, dst_rank)

        # 不断的从头开始遍历 send_order 中保存的layer_id，即遍历所有要发送的X对应的layer id。要发送的tensor都保存在send_dict中，
        # 不断尝试通过layer_id取send_dict中保存的tensor，发送出去
        # 要发送就要把tensor从dict中删掉，即remove，若send_dict键layer_id对应的值为空，remove函数会阻塞住，直到其中加入了
        # 新的tesnor
        elif self.ordering == "pack-by-pack":
            # [8,8,8,8]
            # [4,4,4,4,4,4,4,4]
            # 2 == 1 × 8
            print(len(self.send_order))
            print(len(self.send_dict.layer_ids) * len(self.converted_ubwd))
            assert len(self.send_order) == len(self.send_dict.layer_ids) * len(self.converted_ubwd)
            while True: # each tasks iteration
                for layer_id, _ in self.send_order: # [(layer_id, ubatchsize)，...]
                    # print("[MSGStashX] rank{} wait L{}, send_dict=\n{}\n".format(self.rank, layer_id, self.send_dict))

                    # 从 self.odict[layer_id] 列表中弹出并返回第一个元素
                    # { layer_id: [u1's {"name1": tensor1, "name2": [tensor2]}, u2's {}, ... ] }
                    named_tensors = self.send_dict.remove(layer_id)
                    # 目标rank
                    dst_rank = self.send_ranks[layer_id]
                    # 若要发送到的rank就是当前rank，就是要发送给自己
                    if dst_rank == self.rank:
                        self._send2self(layer_id, named_tensors)

                    # 调用 dist.send 方法将张量发送到dst_rank
                    else:
                        self._send(layer_id, named_tensors, dst_rank)
        else:
            raise ValueError
    
    # 直接将tesnor放到底层存储结构中，达到rank内互传的效果
    def _send2self(self, layer_id, named_tensors):
        """ Helper thread sends tensor to itself rank. """
        self.recv_dicts[self.rank].add(layer_id, named_tensors) 
        # print("[MSGStashX] rank{} _send2self enqueued (X{},{})".format(self.rank, layer_id, list(named_tensors.keys())))

    # 调用 dist.send 方法将张量发送到dst_rank
    def _send(self, layer_id, named_tensors, dst_rank):
        """ Helper thread sends tensor by calling dist.send(). """
        if self.nvprof: nvtx_range_push("__L{} MSG to dst{}".format(layer_id,dst_rank)) 
        # print("[MSGStashX] rank{} _sending L{} to dst{}".format(self.rank, layer_id, dst_rank))
        # named_metas = self.xmeta.get(1, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        for (name,tensor), name2 in zip(named_tensors.items(), self.layer_x_names[layer_id]): # { name: tensor/const, name: [tensors] }
            assert name == name2
            if isinstance(tensor, (torch.Tensor,Variable)):
                assert tensor.is_cuda and not tensor.requires_grad
                dist.send(tensor, dst_rank, self.group, self.send_tag)
            elif isinstance(tensor, (float,int)):
                dist.send(torch.tensor(tensor), dst_rank, self.group, self.send_tag)
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                for t in tensor:
                    assert t.is_cuda and not t.requires_grad
                    dist.send(t, dst_rank, self.group, self.send_tag)
            else:
                raise ValueError("unknown tensor={}".format(tensor))
        if self.nvprof: nvtx_range_pop() 
        # print("[MSGStashX] rank{} _sent L{} to dst{}".format(self.rank, layer_id, dst_rank))
            
        
    # 不断地遍历 self.recv_orders[src_rank]：{src_rank：[(l,u),...], ...}
    # 即不断尝试接从src_rank接收对应layer l的microbatch大小为u的输入X，将读取到的named_tensors连同其layer_id
    # 构成一个元组加入到 self.recv_dicts[src_rank] 这个线程安全字典中
    def _recv_helper_thread(self, src_rank):
        """ This method is to be executed from a helper daemon thread. """
        assert src_rank != self.rank
        if self.ordering == "layer-by-layer":
            while True: # each tasks iteration
                for layer_id in self.recv_dicts[src_rank].layer_ids: # in-order of FWD layers
                    for ubs in self.ubatchszs: 
                        named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory) 
                        self.recv_dicts[src_rank].add(layer_id, named_tensors)

        # 显然执行这个
        elif self.ordering == "pack-by-pack":
            if self.verbose: print("rank{}: _recv_helper_thread(src_rank={}): self.recv_orders={}, self.recv_dicts={}".format(self.rank, src_rank, self.recv_orders, self.recv_dicts))
            assert len(self.recv_orders[src_rank]) == len(self.recv_dicts[src_rank].layer_ids) * len(self.ubatchszs)
            while True: # each tasks iteration
                for layer_id, ubs in self.recv_orders[src_rank]: # [(layer_id, ubatchsize)]
                    # 调用dist.recv函数接收tesnor，接收的tensor放入一个ODict中返回
                    # 📌若没有收到数据会阻塞住
                    named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory) 
                    # 将收到的数据添加到 recv_dicts[src_rank] 这个线程安全字典中
                    self.recv_dicts[src_rank].add(layer_id, named_tensors)
        else:
            raise ValueError

    # 调用dist.recv函数接收tensor，接收的tensor放入一个ODict中返回
    def _recv(self, layer_id, ubatchsize, src_rank, pin_memory=True):
        """ Helper thread receives tensor by calling dist.recv(). """ 
        # print("[rank{}]\tmsg_handler._send: entered".format(self.rank))

        # 获取要接收的X的元数据，需要根据其大小和类型生成tensor
        named_metas = self.xmeta.get(ubatchsize, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        #
        named_tensors = ODict()
        # name即layer_id层的输入名称
        for name in self.layer_x_names[layer_id]:
            meta = named_metas[name]
            if isinstance(meta, TensorMeta):
                tensor = torch.empty(meta.shape, dtype=meta.dtype, device="cpu", pin_memory=pin_memory)
                # 将数据存储在创建的张量中
                # 若没有收到数据会阻塞住
                # print(f"[rank{self.rank}]正在接收layer{layer_id}")
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                named_tensors[name] = tensor 
            elif isinstance(meta, ConstMeta):
                tensor = torch.tensor(meta.const, device="cpu", pin_memory=pin_memory)
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                named_tensors[name] = tensor.item() # convert a 0-dim tensor to a python number
            elif isinstance(meta, list): # output tuple of bert pretrainhead 
                named_tensors[name] = []
                for m in meta:
                    tensor = torch.empty(m.shape, dtype=m.dtype, device="cpu", pin_memory=pin_memory)
                    dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                    named_tensors[name].append(tensor)
            else:
                raise ValueError("unknown meta={}".format(meta))
            # print("[rank{}]\tmsg_handler._send: tensor(shape={}, dtype={}) sent".format(self.rank, tensor.shape, tensor.dtype))
        return named_tensors
    
    # Call by upstream thread. Nonblocking send. 
    # 向 send_dict 这个线程安全字典中的 odict[layer_id] 这个list添加：self.odict[layer_id].append(named_tensors)
    def isend(self, layer_id, named_tensors):
        ''' Call by upstream thread. Nonblocking send. 
            The same API for two ordering of 'layer-by-layer' and 'pack-by-pack' '''
        self.send_dict.add(layer_id, named_tensors) # tuple uses reference to tensor

    # 1.找到对应给定layer_id的src_rank，即从哪个rank上传X过来的
    # 2.从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
    #   即一个 （name, tensor）。若内部没有tensor显然会被阻塞住(wait)
    def recv(self, layer_id):
        ''' Call by downstream thread. Blocking recv. Return named_tensors. '''
        src_rank = self.recv_ranks[layer_id] # { layer_id: src_rank } # can include self.rank
        # 从该src rank对应的线程安全字典上，即从其内部的 self.odict[layer_id] 列表中弹出并返回第一个元素
        # 若内部没有tensor显然会被阻塞住(wait)
        return self.recv_dicts[src_rank].remove(layer_id) # tuple uses reference to tensor    
    
    def has_no_send(self):
        return self.send_dict is None
    
    def has_no_recv(self):
        return False if self.recv_dicts else True

# 专门用于最后一个FWDvt和首个BWDvt在一个GPU上，此时Y的传输媒介为SWP，原有的MSGX无法识别这种情况，导致
# 出现MSGX实例认为既没有发送也没有接收
class MSGX_2(MSGStashX_2):
    """ Handles gloo send/recv of Y/dX between cpu processes. 
        NOTE: Tentative for last fwd task to bwd criterion
        TODO: 1) To support all Y/dX; 2) replace input data structure to queue
    """
    def __init__(self, ubscvt, rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack', pin_memory=True, verbose=False, nvprof=False): 
        super().__init__(ubscvt, rank, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering, pin_memory, verbose, nvprof)
        

    #############################################################################################################
    # 对下面两个确定rank的方法的总结：可见是这样一种场景，只有最后一个FWDvt向第一个BWDvt发送Y，且在cpu上发送，即媒介为MSG
    # 并没有管dX
    #############################################################################################################
        
    # 总结：找到要接收X的目标rank，即首个BWDvt所在rank，和该vt首层组成一个键值对
    #
    # 在当前rank上找最后一个FWD vt，若有，记录该vt输出Y的目标rank（输出媒介必须是MSG才会记录）
    #
    # 若当前rank中存在最后一个FWD任务且输出Y的媒介为MSG，向send_ranks字典添加一个键值对：
    # {l(层号，📌接收输出 Y 的那一层，是l+1不是l)：rank(dst_rank)}
    # 返回 send_ranks 字典
    # 📌分析：vPP任务的最后一个fwd任务输出Y的媒介为P2P，这里的send_ranks应该是空的
    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()
        for vt in rtasks[self.rank]:
            # 若是最后一个fwd任务，且会输出一个Y
            if vt.is_last_fwd and vt.Out['Y']:
                # l 为输出 Y 的那一层的layer_id
                l = vt.layers[-1]
                m = vt.Out['Y'][l]
                # m中保存的rank本来就是 dst_rank
                # 📌分析24/10/8：不管是vPP还是vDP，Out['Y'][l]这个媒介就不可能是MSG
                # ❌：以上回答错误，有这种情况，即前后想microbatch大小不一样
                print(f"MSGX实例, m.medium:{m.medium}")
                if m.medium == "SWP":
                    print(f"目标rank为{m.rank}")
                    send_ranks[l+1] = self.rank # dst_rank
        if self.verbose: print("[MSGX] found send_ranks={}".format(send_ranks))
        return send_ranks

    # 总结：找到产生X并发送它的来源rank，即最后一个FWDvt所在的rank，和首个BWDvt首层的层号组成一个键值对
    #
    # 在当前rank上找第一个BWD vt，若有，记录给该vt发送输入X的来源rank（输入媒介必须是MSG才会记录）
    #
    # 若当前rank中存在第一个BWD任务，且接收的X的媒介为MSG，将接收该X的layer_id和src_rank互相作为键值对添加到两个字典中并返回
    # 返回两个字典：
    # self.recv_ranks = { layer_id（接受的层id）: src_rank（来源rank） } # can include self.rank
    # self.recv_layers = { src_rank: [layer_id] } # can include self.rank
    # 📌分析：vPP第一个BWD任务接收输入X的媒介应该是P2P,这里的两个字典应该是空的
    def _find_recv_ranks_layers(self, rtasks):
        recv_ranks = ODict() # { layer_id: src_rank } # can include self.rank
        recv_layers = ODict() # { src_rank: [layer_id] } # can include self.rank
        for vt in rtasks[self.rank]:
            if vt.type == 'BWD' and vt.has_criterion and vt.In['X']:
                l = vt.layers[0]
                m = vt.In['X'][l]
                print(f"来源rank为{m.rank}")
                if m.medium == "SWP":
                    # 不能使用m.rank赋值，因为媒介为SWP的情况下，m.rank为None（在该媒介实例化时不会赋值）
                    recv_ranks[l] = self.rank # src_rank
                    if self.rank not in recv_layers:
                        recv_layers[self.rank] = []
                    recv_layers[self.rank].append(l)
        if self.verbose: print("[MSGX] found recv_ranks={}, recv_layers={}".format(recv_ranks, recv_layers))
        return recv_ranks, recv_layers

    #############################################################################################################
    # 对下面两个order方法的总结：对一下两个方法的命名和上面两个方法的命名是反过来的。_find_send_ranks虽然命名方式是
    # send，但其实找的是接收方，即目标rank。但下面这个_find_send_order就是确定发送的顺序。
    #############################################################################################################
      

    # 为最后一个fwd任务所在rank的MSGX实例配置发送输出Y的顺序（最后一个fwd任务并非真正的最后一个，真正的最后一个同时也是第一个bwd任务）
    #
    # 在当前rank上寻找最后一个fwd任务，若其输出Y的媒介为MSG，将“要接收输出Y”的层l和每一个ubatchszs_bwd组成一个元组
    # 加入到order列表中，[(接收Y的layer_id,ubatchsize), ...]，最后返回该列表
    # 📌分析：vPP的最后一个fwd任务输出Y的媒介为P2P，返回的 orders 应该是空的
    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []
        # 在当前rank上寻找最后一个fwd任务，若其输出Y的媒介为MSG，将要接收输出Y的层l和每一个ubatchszs_bwd组成一个元组
        # 加入到order列表中
        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            if vt.is_last_fwd:
                for u in self.converted_ubwd:
                    l = vt.layers[-1]
                    m = vt.Out['Y'][l]
                    if m.medium == "SWP":
                        order.append((l+1,u))
        if self.verbose: print("[MSGX] found send order={}".format(order))
        return order
    
    # 为第一个bwd任务所在rank的MSGX实例配置接收（第一个fwd任务输出的Y的）顺序
    #
    # 在所有rank的任务上寻找最后一个fwd任务，若该任务输出的Y的媒介为MSG，且目标rank就是当前rank，则将
    # (l+1(要接收Y的层)，u)添加到src_rank对应的列表中。{src_rank：[(接收Y的l,u),...], ...}
    # 📌分析：vPP的最后一个fwd任务输出Y的媒介为P2P，返回的 orders 应该是空的
    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }
        # 在所有rank的任务上寻找最后一个fwd任务，若该任务输出的Y的媒介为MSG，且目标rank就是当前rank，则将
        # (l+1(要接收Y的层)，u)添加到src_rank对应的列表中
        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.is_last_fwd:
                    for u in self.converted_ubwd:
                        l = vt.layers[-1]
                        m = vt.Out['Y'][l]
                        # 
                        if m.medium == "SWP":
                            # 若最后一个fwd任务发送Y的目标rank就是当前rank
                            if m.rank == self.rank: # can include self queue # and vt.rank != self.rank: 
                                if src_rank not in orders:
                                    orders[src_rank] = []
                                # 
                                orders[src_rank].append((l+1,u))
        if self.verbose: print("[MSGX] found recv orders={}".format(orders))
        return orders

class LocalX(object):
    """ Handles local X/dX in vDP. 
        Assumption:
            1) only CPU tensor and no grad
            2) equal microbatchsize from FWD to BWD (after UBatchSizeConverter)
    """
    def __init__(self, rank, layer_ids):
        self.rank = rank
        self.self_dict = threadsafe_data_struct.OrderedDictionary()
        self.self_dict.init_layer_ids(layer_ids)
        # print("[LocalStashX] rank{} created self_dict={}".format(self.rank, self.self_dict.__repr__(title="self queue")))
    
    def isend(self, layer_id, named_tensors):
        ''' Call by upstream thread. '''
        self.self_dict.add(layer_id, named_tensors) # tuple uses reference to tensor

    def recv(self, layer_id):
        ''' Call by downstream thread. Blocking recv. Return named_tensors. '''
        return self.self_dict.remove(layer_id) # tuple uses reference to tensor    

# 专门用来暂存stage的输出
# 先完成一版不需要提前定义顺序的
class CacheStashX(object):
    def __init__(self, rank, layer_ids, nvprof=False):
        self.rank = rank
        self.nvprof = nvprof
        # self.self_dict = ODict()
        assert isinstance(layer_ids, list)
        # for id in sorted(layer_ids): 
        #     self.self_dict[id] = []

        self.put_queue = threadsafe_data_struct.Queue()

    def put(self, layer_id, named_tensors):
        # self.self_dict[layer_id].append(named_tensors) # tuple uses reference to tensor
        self.put_queue.add((layer_id, named_tensors))

    def fetch(self, layer_id):
        # named_tensors = self.self_dict[layer_id].pop(0)
        # return named_tensors
        return self.put_queue.remove()