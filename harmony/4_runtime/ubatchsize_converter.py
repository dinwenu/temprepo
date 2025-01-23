# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from collections import OrderedDict as ODict

import torch

from torch.autograd import Variable
from torch.cuda.nvtx import range_push as nvtx_range_push 
from torch.cuda.nvtx import range_pop as nvtx_range_pop 

import threadsafe_data_struct

# 实际调用的还是MSGstashX的线程发送tensor，即调用MSGstashX的isend方法，该类只相当于在正常的中间环节中添加一个额外的步骤，
# 用于将更大的FWD用microbatch拆分为BWD用microbatch。
# 分析：前向microbatch size默认是不能超过bwd microbatchsize的，道理很简单。若fwd小于bwd，该microbatch压根不能用于BWD的流程
# 只能被当作剩余数据等待下一个iteration的新数据进来，组合成一个足够大的tensor，在此之前BWD无法进行，根本就无法进行正常训练
class UBatchSizeConverter(object):
    """ Convert different microbatch sizes from forward to backward tasks.
        E.g. stashing X in vPP/vDP (including X to last layer pack in vDP)
        Assumption:
            0) know D, Ufwd, Ubwd in advance
            1) only CPU tensor and no grad
    """
    # u_fwd：fwd micro batch的大小
    # ubatchszs_fwd：fwd microbatchsize的列表
    # u_bwd：bwd micro batch的大小
    # ubatchszs_bwd：bwd microbatchsize的列表
    def __init__(self, rank, data_batchsize, u_fwd, ubatchszs_fwd, u_bwd, ubatchszs_bwd, output_method, pack_ordering=True, pin_memory=True, nvprof=False):
        self.rank = rank
        self.data_batchsize = data_batchsize
        self.u_fwd = u_fwd
        self.ubatchszs_fwd = ubatchszs_fwd
        self.u_bwd = u_bwd
        self.ubatchszs_bwd = ubatchszs_bwd
        # 保险操作，前后向microbatchsize不一样才会实例化该例，因此一样的话这里会输出一个warning信息
        if u_fwd == u_bwd: # assert u_fwd != u_bwd
            print("[UBatchSizeConverter] --- Warning: Ufwd = Ubwd ! ---") 
        assert data_batchsize >= u_fwd and data_batchsize >= u_bwd
        self.pin_memory = pin_memory
        self.nvprof = nvprof
        
        # 
        self._initialize(output_method, pack_ordering)
        self._start_helper_thread()

        # print("[UBatchSizeConverter] __init__: rank {} has D={}, Ufwd={} ({}), Ubwd={} ({})".format(self.rank, self.data_batchsize, self.u_fwd, self.ubatchszs_fwd, self.u_bwd, self.ubatchszs_bwd))

    # 初始化一些数据结构
    def _initialize(self, output_method, pack_ordering=True):
        """
        Initialize state needed for sub-thread. 
        Argument: output_method(layer_id, named_tensor)
                  pack_ordering = bool : whether convert in layer or pack ordering (this ordering is self contained)
        """
        # 实例化一个线程安全的栈
        self.input_queue = threadsafe_data_struct.Queue()
        self.residual = ODict() # { layer_id: named_tensors (with ubatchsize < Ubwd) }
        self.cnt_converted_ubatch = ODict() # { layer_id: cnt }
        self.output_method = output_method # can be dict
        assert self.output_method is not None
        self.pack_ordering = pack_ordering
        assert isinstance(pack_ordering, bool)

    def _start_helper_thread(self):
        helper_thread = threading.Thread(target=self._helper_thread)
        helper_thread.daemon = True
        helper_thread.start()
        # print("[UBatchSizeConverter] rank{} started converter helper thread".format(self.rank))
    
    def _helper_thread(self):
        """ This method is to be executed from a helper daemon thread. """
        if self.pack_ordering: # output in pack ordering: [X0:#1, X1:#1], [X0:#2, X1:#2]
            # print("[UBatchSizeConverter] uses pack ordering")
            while True:
                packed_named_tensors, is_convert = self.input_queue.remove() # [ [ layer_id, named_tensors ] ] # a pack at an ubatch
                if not is_convert:
                    for layer_id, named_tensors in packed_named_tensors:
                        self.output_method(layer_id, named_tensors)
                    continue
                # converting
                layer_converted = ODict() # { layer_id: [#1 cvt_named_tensor, #2 cvt_named_tensor] } 
                for layer_id, named_tensors in packed_named_tensors:
                    converted = self._convert_ubatchsize(layer_id, named_tensors) # this layer's [ { cvt_named_tensor } ]
                    if converted == []:
                        # print("[UBatchSizeConverter] rank{}: converted is empty".format(self.rank))
                        continue
                    else:
                        layer_converted[layer_id] = converted
                #
                if layer_converted:
                    num_ubwds = set()
                    for converted in layer_converted.values():
                        num_ubwds.add(len(converted))
                    assert len(num_ubwds) == 1, "layers in a pack must have equal num of ubwd to send"
                    for idx in range(list(num_ubwds)[0]):
                        for layer_id, converted in layer_converted.items():
                            self.output_method(layer_id, converted[idx])
                            # print("[UBatchSizeConverter] rank{}: outputed L{}".format(self.rank, layer_id))

        # 执行这个
        else: # output in layer ordering: X0:[#1,#2,#3] then X1:[#1,#2,#3]
            # print("[UBatchSizeConverter] uses layer ordering")
            while True:
                # isend放入input_queue中的is_convert默认为true
                layer_id, named_tensors, is_convert = self.input_queue.remove() # a layer at an ubatch
                # 所以这里也不会执行
                if not is_convert:
                    self.output_method(layer_id, named_tensors)
                    continue
                # converting
                if self.nvprof: nvtx_range_push("__L{} ConvertU(X)".format(layer_id)) 
                # 1.将传进来的stashX这个tensor按照BWD需要的microbatch大小进行分块，（当然，最后一个块可能没有u_bwd的大小）。
                #   若self.residual中存在 layer_id 这个键，说明上一次iteration中，该layer_id的输入X没有全部使用，存在剩余
                #   数据，即最后一个拆分出来的tensor块的大小小于反向时需要的microbatch的大小，不能直接被BWD任务使用。此时，
                #   将剩余数据拼接到当前输入tensor的后面 
                # 2.对拆分后的每一个tensor块进行检查，根据其大小是否等于反向传播时microbatch的大小，决定将其放入converted列表，
                #   还是self.residual中，即用于保存剩余数据的字典{layer_id: not_ready(一个字典{name:tensor})}
                # 3.最终返回converted列表，即所有符合标准，大小与反向Microbatchsize相同的tesnor块的列表 
                converted = self._convert_ubatchsize(layer_id, named_tensors) # this layer's [ { named_tensors of Ubwd } ]
                if converted == []:
                    # print("[UBatchSizeConverter] rank{}: converted is empty".format(self.rank))
                    if self.nvprof: nvtx_range_pop() 
                    continue
                else:
                    # 将convert好的tensor列表加入到MSGstashX的send_ditc字典中，这也意味着MSGstashX实例的线程将开始执行
                    # 向目标rank的发送任务。即最终还是用MSGstashX实例的isend方法将convert后的tensor发送到目标rank
                    for cvt_named_tensor in converted:
                        self.output_method(layer_id, cvt_named_tensor)
                        # print("[UBatchSizeConverter] rank{}: outputed L{}".format(self.rank, layer_id))
                    if self.nvprof: nvtx_range_pop() 

    # 1.将传进来的stashX这个tensor按照BWD需要的microbatch大小进行分块，（当然，最后一个块可能没有u_bwd的大小）。
    #   若self.residual中存在 layer_id 这个键，说明上一次iteration中，该layer_id的输入X没有全部使用，存在剩余
    #   数据，即最后一个拆分出来的tensor块的大小小于反向时需要的microbatch的大小，不能直接被BWD任务使用。此时，
    #   将剩余数据拼接到当前输入tensor的前面 ，而后再进行分块
    # 2.对拆分后的每一个tensor块进行检查，根据其大小是否等于反向传播时microbatch的大小，决定将其放入converted列表，
    #   还是self.residual中，即用于保存剩余数据的字典{layer_id: not_ready(一个字典{name:tensor})}
    # 3.最终返回converted列表，即所有符合标准，大小与反向Microbatchsize相同的tesnor块的列表     
    def _convert_ubatchsize(self, layer_id, named_tensors):
        """
        Helper thread converts one layer's tensors from Ufwd to Ubwd.
        Use previously stored residual tensors (not sufficient for Ubwd) for each convert call.
        Store residual tensors of this convert call for the next one.
        Return converted = [ { named_tensors of Ubwd } ] or []

        Note: the actually residual memory in pytorch == a residual Ubwd + an extra Ufwd == the concat'ed size 
              (i.e., _concat_tensors create an atomic big tensor. Even if a split of it gets deleted, the entire concat'ed memory is still there. Unless all splits gets deleted.)
        """
        # find new split
        named_split = ODict() # { name: (t1,t2), name: (c1,c2), name: [ (t1,t2), (t1,t2) ] }
        num_split = set()
        # self.residual = ODict() # { layer_id: named_tensors (with ubatchsize < Ubwd) }
        for name,tensor in named_tensors.items(): # { name: tensor/const, name: [tensors] }
            if isinstance(tensor, (torch.Tensor,Variable)):
                # assert not tensor.is_cuda and not tensor.requires_grad
                # 若self.residual中存在layer_id这个键，说明上一次iteration中，该layer_id的输入X没有全部使用，存在剩余
                # 数据，即最后一个拆分出来的tensor块的大小小于反向时需要的microbatch的大小，不能直接被BWD任务使用。此时，
                # 将剩余数据拼接到当前输入tensor的前面
                if layer_id in self.residual: # and name in self.residual[layer_id]:
                    # 在第0个维度上拼接tensor，并放到pinned memory中
                    concat_tensor = self._concat_tensors((self.residual[layer_id][name],tensor))
                else:
                    concat_tensor = tensor
                # 在第0个维度上，将tensor拆分为多个快， 每个分块的大小为 u_bwd (反向传播的microbatch size)，
                # 并将这些tensor以tuple形式返回。（当然，最后一个块可能没有u_bwd的大小）
                named_split[name] = self._split_tensor(concat_tensor, self.u_bwd) # (t1,t2) or (t1,res) or (t1,) or (res,)
                # 将当前tensor分块的数量添加到set中
                num_split.add(len(named_split[name]))
            elif isinstance(tensor, int):
                assert tensor in self.ubatchszs_fwd, "convert ubatchsize on unknown int value={}".format(tensor) # TODO: can use repeated int const
                if layer_id in self.residual: # and name in self.residual[layer_id]:
                    concat_tensor = self._concat_const_ubatchsizes((self.residual[layer_id][name],tensor))
                else:
                    concat_tensor = tensor
                named_split[name] = self._split_const_ubatchsize(concat_tensor, self.u_bwd) # (c1,c2) or (c1,res) or (c1,) or (res,)
                num_split.add(len(named_split[name]))
            elif isinstance(tensor, list): # output tuple of bert pretrainhead 
                tmp = []
                for i,t in enumerate(tensor):
                    # assert not t.is_cuda and not t.requires_grad
                    if layer_id in self.residual: # and name in self.residual[layer_id]:
                        concat_t = self._concat_tensors((self.residual[layer_id][name][i],t))
                    else:
                        concat_t = t
                    tmp.append(self._split_tensor(concat_t, self.u_bwd)) # (t1,t2) or (t1,res) or (t1,) or (res,)
                    num_split.add(len(tmp[-1]))
                named_split[name] = tmp
            else:
                raise ValueError("unknown tensor type to convert ={}".format(type(tensor)))
        # save residual and return converted
        assert len(num_split) == 1, "num_split must be unique"
        # 从self.residual中删除当前layer_id的键值对，因为剩余数据，即一个tensor块，已经拼接到当前tensor上了
        if layer_id in self.residual: # { layer_id: named_tensors (with ubatchsize < Ubwd) }
            del self.residual[layer_id] 
        # 若当前这个layer_id还没处理过，即还没进行过转化，初始化其转化为0
        if not layer_id in self.cnt_converted_ubatch: # { layer_id: cnt }
            self.cnt_converted_ubatch[layer_id] = 0
        # 以该layer的已转化次数为下标，从ubatchszs_bwd中取出bwd microbatch的大小
        u_bwd = self.ubatchszs_bwd[self.cnt_converted_ubatch[layer_id]]
        converted = []
        # 对拆分后的每一个tensor块进行检查，根据其大小是否等于反向传播时microbatch的大小，决定将其放入converted列表，
        # 还是self.residual中，即用于保存剩余数据的字典{layer_id: not_ready(一个字典{name:tensor})}
        # 遍历每一个tensor块，执行：
        # 1.对拆分后的每一个tensor块进行检查：
        #   1.1.若当前tensor块的大小和反向传播的大小相同，则表明该tensor块可以直接用，放进ready字典中{name：tensor}
        #   1.2.否则，不能直接给反向任务用，放进not_ready字典中
        # 2.
        # 2.1.若ready字典不为空，说明当前分割出来的tensor块的大小等于反向传播时microbatch的大小，将该字典添加到converted列表中，
        #     并将该layer_id的计数+1，表示执行了一次microbatch大小的转换。若所有的microbatch已经转换完了，将该层的计数置0，以便
        #     下一次iteration正确执行
        # 2.2.若not_ready字典不为空，说明最后一个tensor块的大小小于反向传播时microbatch的大小，不能直接用
        #     直接将其保存到self.residual这个剩余数据字典中{layer_id: not_ready(一个字典{name:tensor})}
        #
        # 一次循环只处理一个拆分后的tensor，也就是说一次往converted中装一个 name:tensor，最终converted中这几个键值对name都是一样的
        for j in range(list(num_split)[0]):
            ready = ODict() # { name: t1, name: c1, name: [t1,t1] }
            not_ready = ODict() 
            # 📌虽然用的是for循环，named_split这个字典只有一个键值对，因此for循环只执行了一次
            for name, split in named_split.items(): # { name: (t1,t2), name: (c1,c2), name: [ (t1,t2), (t1,t2) ] }
                # print("[UBatchSizeConverter] rank{}'s named_split has {}:{}".format(self.rank, name, split)) 
                if isinstance(split,tuple) and isinstance(split[j], (torch.Tensor,Variable)):
                    tensor = split[j]
                    # 若当前tensor块的大小和反向传播的大小相同，则表明该tensor块可以直接用，放进ready字典中{name：tensor}
                    if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                        ready[name] = tensor
                    # 否则，不能直接给反向任务用，放进not_ready字典中
                    elif tensor.size(0) < u_bwd: # residual
                        not_ready[name] = tensor
                    else:
                        raise ValueError
                elif isinstance(split,tuple) and isinstance(split[j], int):
                    tensor = split[j]
                    if tensor == u_bwd: # 0-dim matches desired ubatchsize
                        ready[name] = tensor
                    elif tensor < u_bwd: # residual
                        not_ready[name] = tensor
                    else:
                        raise ValueError
                elif isinstance(split,list):
                    # tmp_tensor, match_flag = [], []
                    # for s in split:
                    #     tensor = s[j]
                    #     tmp_tensor.append(tensor)
                    #     if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                    #         match_flag.append(True)
                    #     elif tensor.size(0) < u_bwd: # residual
                    #         match_flag.append(False)
                    #     else:
                    #         raise ValueError
                    # if match_flag == [True]*len(match_flag):
                    #     ready[name] = tmp_tensor
                    # elif match_flag == [False]*len(match_flag):
                    #     not_ready[name] = tmp_tensor
                    # else:
                    #     raise ValueError
                    tmp1, tmp2 = [], []
                    for s in split:
                        tensor = s[j]
                        if tensor.size(0) == u_bwd: # 0-dim matches desired ubatchsize
                            tmp1.append(tensor)
                        elif tensor.size(0) < u_bwd: # residual
                            tmp2.append(tensor)
                        else:
                            raise ValueError
                    if tmp1 != []:
                        ready[name] = tmp1
                    elif tmp2 != []:
                        not_ready[name] = tmp2
                    else:
                        raise ValueError
                else:
                    raise ValueError
            # 若ready字典不为空，说明当前分割出来的tensor块的大小等于反向传播时microbatch的大小，将该字典添加到converted列表中，
            # 并将该layer_id的计数+1，表示执行了一次microbatch大小的转换。若所有的microbatch已经转换完了，将该层的计数置0，以便
            # 下一次iteration正确执行
            if ready:
                assert list(ready.keys()) == list(named_tensors.keys()), "{} vs. {}".format(list(ready.keys()), list(named_tensors.keys()))
                # 将该字典添加到converted列表中
                converted.append(ready)
                # 将该layer_id的计数+1，表示执行了一次microbatch大小的转换
                self.cnt_converted_ubatch[layer_id] += 1
                cnt = self.cnt_converted_ubatch[layer_id]
                # 若还没执行到最后一次转换（总次数为BWD microbatchsize的列表，即反向要执行几个microbatch），
                # ubwd，即反向的microbatch size，取self.ubatchszs_bwd[cnt]，实际只要不是最后一个，值都是一样的
                if cnt < len(self.ubatchszs_bwd): # not last ubatch yet
                    u_bwd = self.ubatchszs_bwd[cnt]
                    # print("[UBatchSizeConverter] rank{}: converted L{}'s {} ubatches".format(self.rank,layer_id,cnt))
                
                # 所有的microbatch已经转换完了，将该层的计数置0，以便下一次iteration正确执行
                else: # last ubatch done (of this iteration)
                    u_bwd = -1 # prevent keep looping
                    self.cnt_converted_ubatch[layer_id] = 0 # reset for next iteration
                    assert not layer_id in self.residual, "no more residual left"
                    # print("[UBatchSizeConverter] rank{}: converted L{}'s All {} ubatches".format(self.rank,layer_id,cnt))
            # 若not_ready字典不为空，说明最后一个tensor块的大小小于反向传播时microbatch的大小，不能直接用
            # 直接将其保存到self.residual这个剩余数据字典中
            elif not_ready:
                assert j == list(num_split)[0]-1, "residual must be the last split"
                assert list(not_ready.keys()) == list(named_tensors.keys())
                self.residual[layer_id] = not_ready
            else:
                raise ValueError
        # clean up
        del named_split
        
        # 最终返回所有符合标准，即大小与反向Microbatchsize相同的tesnor块的列表
        return converted
                
    # 在第0个维度上拼接tensor，并放到pinned memory中
    def _concat_tensors(self, tensors):
        for t in tensors:
            # assert isinstance(t, (torch.Tensor,Variable))
            # assert not t.is_cuda and not t.requires_grad
            assert t.ndim > 0, "scalar tensor cannot be concat'ed"
        # dim=0 must be ubatchsize
        if self.pin_memory:
            # 在第0个维度上拼接tensor，并放到pinned memory中
            return torch.cat(tensors, dim=0).pin_memory() # create new memory # inherit tensor's device
        else:
            return torch.cat(tensors, dim=0)

    # 在第0个维度上，将tensor拆分为 split_size 个，并将这些tensor以列表形式返回
    def _split_tensor(self, t, split_size):
        # assert isinstance(t, (torch.Tensor,Variable))
        # assert not t.is_cuda and not t.requires_grad
        assert t.ndim > 0, "scalar tensor cannot be split'ed"
        # dim=0 must be ubatchsize
        return torch.split(t, split_size, dim=0) # share the same underlying memory # inherit tensor's device
        # tensor will be split into equally sized chunks (if possible). 
        # Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by split_size.
    
    def _concat_const_ubatchsizes(self, Cs):
        return int(sum(Cs))

    def _split_const_ubatchsize(self, C, U):
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
    
    # input：layer_id
    # input2：cpu_named_tensors
    # 将layer_id和input2：cpu_named_tensors加入到 input_queue 队列中，这意味着UBatchSizeConverter实例的线程
    # 将开始执行tensor大小的转换，而后将convert好的tensor列表加入到MSGstashX的send_ditc字典中，这也意味着
    # MSGstashX实例的线程将开始执行向目标rank的发送任务
    def isend(self, input, input2=None, is_convert=True):
        ''' 
        Call by upstream thread. Nonblocking send.
        Argument: 
            input, input2 = 
                [ [ layer_id, named_tensors ] ] -- a pack at an ubatch
                or [ layer_id, named_tensors ]  -- a layer at an ubatch
                or layer_id, named_tensors      -- a layer at an ubatch
            is_convert = whether to convert this ubatch
        '''
        # 该参数为false，执行else
        if self.pack_ordering:
            assert isinstance(input, list) and isinstance(input[0], list) and len(input[0])==2
            self.input_queue.add([input, is_convert])
        # 
        else:
            if input2 is None:
                assert isinstance(input, list) and len(input)==2
                self.input_queue.add( input+is_convert ) 
            else:
                self.input_queue.add([input, input2, is_convert]) 
