# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
from collections import OrderedDict as ODict, defaultdict
from time import perf_counter as pc
import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop as nvtx_range_pop
from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta
from task_data_struct import Medium, vTask
import threadsafe_data_struct
import datetime
PINNED_LIFETIME_FIX = os.environ.get("HARMONY_PINNED_LIFETIME_FIX", "1") in ["1", "True", "true", "YES", "yes"]
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
        self.group = dist.new_group(ranks=None, backend='gloo')#,timeout=datetime.timedelta(seconds=30)
        self.pin_memory = pin_memory
        self.verbose = verbose
        self.nvprof = nvprof


        self._send_times = []
        self._recv_times = []

        self._recv_alloc_times = []
        self._recv_dist_times  = []
        # This function requires that 1) all processes in the main group (i.e. all processes that are part of the distributed job) enter this function, even if they are not going to be members of the group; 2) groups should be created in the same order in all processes.
        self._initialize(rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering)


        self._start_helper_threads()


    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()


        for vt in rtasks[self.rank]:
            #
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.Out['X']: # FIX 4)
                for l,m in vt.Out['X'].items():

                    if m.medium == "MSG":
                        send_ranks[l] = m.rank # dst_rank
        return send_ranks


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


    #
    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []

        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            #
            if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)

                for u in self.ubatchszs:

                    for l,m in vt.Out['X'].items():
                        if m.medium == "MSG":
                            order.append((l,u)) # can include self queue
        return order


    # ubatchsize_bwd。{src_rank：[(l,u),...], ...}

    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }
        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.type == 'FWD' and (not vt.has_criterion) and vt.is_gpu: # FIX 4)
                    for u in self.ubatchszs:
                        for l,m in vt.Out['X'].items():
                            if m.medium == "MSG":


                                if m.rank == self.rank: # can include self queue # and vt.rank != self.rank:
                                    if src_rank not in orders:
                                        orders[src_rank] = []
                                    orders[src_rank].append((l,u))
        return orders


    #   self.recv_layers = { src_rank: [layer_id] } # can include self.rank


    # 6.self.ubatchszs = ubatchszs_bwd
    # 7.


    #       ubatchsize_bwd。{src_rank：[(l,u),...], ...}
    def _initialize(self, rtasks, layer_x_names, xmeta, ubatchszs_bwd, ordering='pack-by-pack'):
        """
        Argument: ordering = the sending order (this ordering is self contained)
        """
        # setup send dict # { layer_id: dst_rank } # can include self.rank


        self.send_ranks = self._find_send_ranks(rtasks)
        print(f"rank:{self.rank}, send_ranks:{self.send_ranks}")
        send_layer_ids = list(self.send_ranks.keys())
        self.send_layer_count = {l:0 for l in send_layer_ids}
        print(f"rank:{self.rank}, send_layer_count:{self.send_layer_count}")


        if self.send_ranks:

            self.send_dict = threadsafe_data_struct.OrderedDictionary(self.rank) # between main and send threads


            self.send_dict.init_layer_ids(list(self.send_ranks.keys()))
            self.send_tag = self.rank
            if self.verbose: print_str = "[MSGStashX]\nrank{} set up send_dict=\n{}\n".format(self.rank, self.send_dict)
        else:
            self.send_dict = None
            if self.verbose: print_str = "[MSGStashX]\nrank{} has NO send job\n".format(self.rank)

        # setup recv dicts


        # self.recv_layers = { src_rank: [layer_id] } # can include self.rank
        self.recv_ranks, self.recv_layers = self._find_recv_ranks_layers(rtasks)
        print(f"rank:{self.rank}, recv_ranks:{self.recv_ranks}, recv_layers:{self.recv_layers}")

        self.recv_layer_count = {l:0 for l in self.recv_ranks.keys()}
        print(f"rank:{self.rank}, recv_layer_count:{self.recv_layer_count}")
        #
        self.recv_dicts = ODict() # { src_rank: the thread safe dict } # can include self.rank


        for r in sorted(set(self.recv_layers.keys())):

            if r == self.rank: # loopback to self dict

                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary(self.rank, f"recv_dict[{r}]") # between send and main threads


                self_layer_ids = sorted(self.recv_layers[self.rank])


                for l,dst in self.send_ranks.items():
                    if dst == self.rank:
                        assert l in self_layer_ids


                self.recv_dicts[r].init_layer_ids(self_layer_ids)


            else: # recv from other rank
                self.recv_dicts[r] = threadsafe_data_struct.OrderedDictionary(self.rank, f"recv_dict[{r}]") # between recv and main threads
                self.recv_dicts[r].init_layer_ids(sorted(self.recv_layers[r]))
        #
        #
        self.recv_tags = ODict() # { src_rank : tag }

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

        elif ordering == 'pack-by-pack':


            # [(vt1's layer1,u1),(vt1's layer1,u1),(vt2's layer1,u2),(vt2's layer1,u2),...]
            self.send_order = self._find_send_order(rtasks)
            # print(f"rank:{self.rank}, send_order:{self.send_order}")

            # ubatchsize_bwd。{src_rank：[(l,u),...], ...}

            self.recv_orders = self._find_recv_order(rtasks)
            # print(f"rank:{self.rank}, recv orders:{self.recv_orders}")
            if self.verbose:
                print_str += "rank{} set up send_order = {}\n".format(self.rank, self.send_order)
                print_str += "rank{} set up recv_orders = {}\n".format(self.rank, self.recv_orders)
        else:
            raise ValueError
        # setup X_names
        self.layer_x_names = layer_x_names # { layer_id: X_names } # TODO: less stashing X after checking Identity chain --> stash X is always needed
        self.xmeta = xmeta # dictionary of TensorInfo

        if self.verbose: print(print_str)


    def _start_helper_threads(self):
        """ Start helper communication threads, one for each queue. """
        # Setup send thread
        cnt_send_thd = 0
        # { layer_id: dst_rank，... }
        if self.send_dict is not None:


            #


            helper_thread = threading.Thread(target=self._send_helper_thread)
            helper_thread.daemon = True

            helper_thread.start()
            cnt_send_thd += 1
        # Setup recv thread for each queue (excluding self queue)
        cnt_recv_thd = 0

        for src_rank in self.recv_dicts.keys():
            if src_rank != self.rank:


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


        elif self.ordering == "pack-by-pack":

            # print(f"[rank{self.rank}] send_order: {self.send_order}", flush=True)
            send_order_idx = ""
            for idx, (layer_id, _) in enumerate(self.send_order):
                send_order_idx+=f"\t[rank{self.rank}] send_order[{idx}]: {layer_id}\n"
            # print(send_order_idx, flush=True)
            # print(f"\t[rank{self.rank}] send_ranks: {self.send_ranks}", flush=True)
            assert len(self.send_order) == len(self.send_dict.layer_ids) * len(self.ubatchszs)
            while True: # each tasks iteration
                layer_send_nth = defaultdict(int)
                for idx, (layer_id, _) in enumerate(self.send_order): # [(layer_id, ubatchsize)，...]
                    # print("[MSGStashX] rank{} wait L{}, send_dict=\n{}\n".format(self.rank, layer_id, self.send_dict))


                    # { layer_id: [u1's {"name1": tensor1, "name2": [tensor2]}, u2's {}, ... ] }
                    layer_send_nth[layer_id] += 1
                    nth = layer_send_nth[layer_id]

                    named_tensors = self.send_dict.remove(layer_id)


                    dst_rank = self.send_ranks[layer_id]

                    if dst_rank == self.rank:
                        self._send2self(layer_id, named_tensors, self.pin_memory)

                    else:

                        self._send(layer_id, named_tensors, dst_rank)


                    # del named_tensors

        else:
            raise ValueError


    def _send2self(self, layer_id, named_tensors, pin_memory=True):
        """ Helper thread sends tensor to itself rank. """
        if pin_memory: # move tensors to pin memory if not already pinned.
            for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
                if isinstance(tensor, (torch.Tensor,Variable)):
                    assert not tensor.is_cuda and not tensor.requires_grad
                    named_tensors[name] = tensor.pin_memory() # If the tensor is not pinned, returns a new copy in pinned memory. Else, returns itself (already pinned).
                elif isinstance(tensor, (float,int)):
                    continue
                elif isinstance(tensor, (list, tuple)): # output tuple of bert pretrainhead or position embeddings
                    pinned_tensor = []
                    for t in tensor:
                        assert not t.is_cuda and not t.requires_grad
                        pinned_tensor.append(t.pin_memory())
                    named_tensors[name] = pinned_tensor
                else:
                    raise ValueError("unknown tensor={}".format(tensor))
        self.recv_dicts[self.rank].add(layer_id, named_tensors)
        # print("[MSGStashX] rank{} _send2self enqueued (X{},{})".format(self.rank, layer_id, list(named_tensors.keys())))


    def _send(self, layer_id, named_tensors, dst_rank):
        """ Helper thread sends tensor by calling dist.send(). """
        if self.nvprof: nvtx_range_push("__L{} MSG to dst{}".format(layer_id,dst_rank))
        # print("[MSGStashX] rank{} _sending L{} to dst{}".format(self.rank, layer_id, dst_rank))
        # named_metas = self.xmeta.get(1, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        _t0 = pc()
        for (name,tensor), name2 in zip(named_tensors.items(), self.layer_x_names[layer_id]): # { name: tensor/const, name: [tensors] }
            assert name == name2

            if isinstance(tensor, (torch.Tensor,Variable)):
                assert not tensor.is_cuda and not tensor.requires_grad

                dist.send(tensor, dst_rank, self.group, self.send_tag)

            elif isinstance(tensor, (float,int)):

                dist.send(torch.tensor(tensor), dst_rank, self.group, self.send_tag)

            elif isinstance(tensor, (list, tuple)): # output tuple of bert pretrainhead
                for t in tensor:
                    assert not t.is_cuda and not t.requires_grad

                    dist.send(t, dst_rank, self.group, self.send_tag)

            else:
                raise ValueError("unknown tensor={}".format(tensor))
        self._send_times.append(pc() - _t0)
        if self.nvprof: nvtx_range_pop()
        # print("[MSGStashX] rank{} _sent L{} to dst{}".format(self.rank, layer_id, dst_rank))


    def _recv_helper_thread(self, src_rank):
        """ This method is to be executed from a helper daemon thread. """
        assert src_rank != self.rank
        if self.ordering == "layer-by-layer":
            while True: # each tasks iteration
                for layer_id in self.recv_dicts[src_rank].layer_ids: # in-order of FWD layers
                    for ubs in self.ubatchszs:
                        named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory)
                        self.recv_dicts[src_rank].add(layer_id, named_tensors)


        elif self.ordering == "pack-by-pack":
            if self.verbose: print("rank{}: _recv_helper_thread(src_rank={}): self.recv_orders={}, self.recv_dicts={}".format(self.rank, src_rank, self.recv_orders, self.recv_dicts))


            # print(f"[rank{self.rank}] ubatchszs: {self.ubatchszs}", flush=True)
            # print(f"[rank{self.rank}] recv_orders[{src_rank}]: {self.recv_orders[src_rank]}", flush=True)
            # print(f"[rank{self.rank}] recv_dicts[{src_rank}].layer_ids: {self.recv_dicts[src_rank].layer_ids}", flush=True)
            recv_orders_idx = ""
            for idx, (layer_id, ubs) in enumerate(self.recv_orders[src_rank]):
                recv_orders_idx+=f"\t[rank{self.rank}] recv_orders[{src_rank}][{idx}]: {layer_id}\n"
            # print(recv_orders_idx, flush=True)
            recv_dicts_idx = ""
            for layer_id in self.recv_dicts[src_rank].layer_ids:
                recv_dicts_idx+=f"\t[rank{self.rank}] recv_dicts[{src_rank}].layer_ids: {layer_id}\n"
            # print(recv_dicts_idx, flush=True)
            assert len(self.recv_orders[src_rank]) == len(self.recv_dicts[src_rank].layer_ids) * len(self.ubatchszs)
            while True: # each tasks iteration
                for idx, (layer_id, ubs) in enumerate(self.recv_orders[src_rank]): # [(layer_id, ubatchsize)]


                    named_tensors = self._recv(layer_id, ubs, src_rank, self.pin_memory)


                    self.recv_dicts[src_rank].add(layer_id, named_tensors)
        else:
            raise ValueError


    def _recv(self, layer_id, ubatchsize, src_rank, pin_memory=True):
        """ Helper thread receives tensor by calling dist.recv(). """
        # print("[rank{}]\tmsg_handler._send: entered".format(self.rank))


        named_metas = self.xmeta.get(ubatchsize, layer_id) # { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }
        #
        named_tensors = ODict()
        _alloc_sum = 0.0
        _recv_sum = 0.0

        for name in self.layer_x_names[layer_id]:
            meta = named_metas[name]
            if isinstance(meta, TensorMeta):
                _ta = pc()
                tensor = torch.empty(meta.shape, dtype=meta.dtype, device="cpu", pin_memory=pin_memory)
                _alloc_sum += pc() - _ta
                _tr = pc()
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                _recv_sum += pc() - _tr
                named_tensors[name] = tensor
            elif isinstance(meta, ConstMeta):
                _ta = pc()
                tensor = torch.tensor(meta.const, device="cpu", pin_memory=pin_memory)
                _alloc_sum += pc() - _ta
                _tr = pc()
                dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                _recv_sum += pc() - _tr
                named_tensors[name] = tensor.item()
            elif isinstance(meta, (list, tuple)): # output tuple of bert pretrainhead or position embeddings
                named_tensors[name] = []
                for m in meta:
                    _ta = pc()
                    tensor = torch.empty(m.shape, dtype=m.dtype, device="cpu", pin_memory=pin_memory)
                    _alloc_sum += pc() - _ta
                    _tr = pc()
                    dist.recv(tensor, src_rank, self.group, self.recv_tags[src_rank])
                    _recv_sum += pc() - _tr
                    named_tensors[name].append(tensor)
            else:
                raise ValueError("unknown meta={}".format(meta))
        self._recv_alloc_times.append(_alloc_sum)
        self._recv_dist_times.append(_recv_sum)
        self._recv_times.append(_alloc_sum + _recv_sum)
        return named_tensors

    # Call by upstream thread. Nonblocking send.

    def isend(self, layer_id, named_tensors):
        ''' Call by upstream thread. Nonblocking send.
            The same API for two ordering of 'layer-by-layer' and 'pack-by-pack' '''
        self.send_dict.add(layer_id, named_tensors) # tuple uses reference to tensor


    def recv(self, layer_id):
        ''' Call by downstream thread. Blocking recv. Return named_tensors. '''
        src_rank = self.recv_ranks[layer_id] # { layer_id: src_rank } # can include self.rank


        return self.recv_dicts[src_rank].remove(layer_id) # tuple uses reference to tensor

    def reset_times(self):
        """Reset recorded communication timings."""
        self._send_times = []
        self._recv_times = []
        self._recv_alloc_times = []
        self._recv_dist_times = []

    def get_times(self):
        """Return accumulated communication timings."""
        return list(self._send_times), list(self._recv_times)

    def get_recv_breakdown(self):
        """Return the receive timing breakdown."""
        return list(self._recv_alloc_times), list(self._recv_dist_times)

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


    #############################################################################################################


    #

    #


    def _find_send_ranks(self, rtasks):
        send_ranks = ODict()
        for vt in rtasks[self.rank]:

            if vt.is_last_fwd and vt.Out['Y']:

                l = vt.layers[-1]
                m = vt.Out['Y'][l]


                print(f"MSGX实例, m.medium:{m.medium}")
                if m.medium == "MSG":
                    send_ranks[l+1] = m.rank # dst_rank
        if self.verbose: print("[MSGX] found send_ranks={}".format(send_ranks))
        return send_ranks


    #

    #


    # self.recv_layers = { src_rank: [layer_id] } # can include self.rank

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


    #############################################################################################################


    #


    def _find_send_order(self, rtasks):
        """ find the send order of [(layer_id,ubatchsize)] from self rank. """
        order = []


        for vt in rtasks[self.rank]: # pre-run task graph at self rank
            if vt.is_last_fwd:
                for u in self.ubatchszs:
                    l = vt.layers[-1]
                    m = vt.Out['Y'][l]
                    if m.medium == "MSG":
                        order.append((l+1,u))
        if self.verbose: print("[MSGX] found send order={}".format(order))
        return order


    #


    def _find_recv_order(self, rtasks):
        """ find the recv orders of { src_rank : (layer_id,ubatchsize) } to self rank. """
        orders = ODict() # { src_rank: order }


        for src_rank, vts in rtasks.items(): # pre-run task graph at all ranks
            for vt in vts:
                if vt.is_last_fwd:
                    for u in self.ubatchszs:
                        l = vt.layers[-1]
                        m = vt.Out['Y'][l]
                        #
                        if m.medium == "MSG":

                            if m.rank == self.rank: # can include self queue # and vt.rank != self.rank:
                                if src_rank not in orders:
                                    orders[src_rank] = []
                                #
                                orders[src_rank].append((l+1,u))
        if self.verbose: print("[MSGX] rank{} found recv orders={}".format(self.rank, orders))
        return orders


class LocalX(object):
    """ Handles local X/dX in vDP.
        Assumption:
            1) only CPU tensor and no grad
            2) equal microbatchsize from FWD to BWD (after UBatchSizeConverter)
    """
    def __init__(self, rank, layer_ids):
        self.rank = rank
        self.self_dict = threadsafe_data_struct.OrderedDictionary(self.rank, "local_x")
        self.self_dict.init_layer_ids(layer_ids)
        # print("[LocalStashX] rank{} created self_dict={}".format(self.rank, self.self_dict.__repr__(title="self queue")))

    def isend(self, layer_id, named_tensors):
        ''' Call by upstream thread. '''
        self.self_dict.add(layer_id, named_tensors) # tuple uses reference to tensor

    def recv(self, layer_id):
        ''' Call by downstream thread. Blocking recv. Return named_tensors. '''
        return self.self_dict.remove(layer_id) # tuple uses reference to tensor
