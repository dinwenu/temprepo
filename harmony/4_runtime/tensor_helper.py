# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import OrderedDict as ODict

import torch
import torch.distributed as dist
from torch.autograd import Variable

from prof_data_struct import ConstMeta, TensorMeta, XMeta, TMeta
from profiler import realize_X, realize_dX, realize_D, realize_T

# 将ODict中的值，即tensor，移动到pinned memory中
@torch.no_grad()
def pin_named_tensors(cpu_named_tensors):
    for name,tensor in cpu_named_tensors.items(): # { name: tensor/const, name: [tensors] }
        if isinstance(tensor, (torch.Tensor,Variable)):
            cpu_named_tensors[name] = tensor.pin_memory()
        elif isinstance(tensor, (float,int)):
            continue
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            tmp = []
            for t in tensor:
                tmp.append(t.pin_memory())
            cpu_named_tensors[name] = tmp
        else:
            raise ValueError("unknown tensor={}".format(tensor))

def synthesize_data(XMETA, TMETA, ubatchszs_fwd, ubatchszs_bwd, pin_memory=True):
    # from profiler import realize_D, realize_T # overhead: https://wiki.python.org/moin/PythonSpeed/PerformanceTips#Import_Statement_Overhead
    data_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]
    target_ubatches = [] # [ {named_tensors}, {named_tensors}, ... ]

    for u in ubatchszs_fwd:
        named_data = realize_D(XMETA, u, device="cpu", use_rand=False)
        if pin_memory:
            pin_named_tensors(named_data)
        data_ubatches.append(named_data)
    
    for u in ubatchszs_bwd:
        named_target = realize_T(TMETA, u, device="cpu", use_rand=False)
        if pin_memory:
            pin_named_tensors(named_target)
        target_ubatches.append(named_target)
    
    return data_ubatches, target_ubatches

# 启用tensor的梯度计算，并设置为保留梯度。若输入的tensor是复用之前的tensor，还需进行detach_()以及梯度清零操作
def _turn_on_grad(tensor):
    """ If tensor has no gradient (.grad=None/detached/not requires_grad), then just turn on its gradient.
    Else, keep its gradient buff (with zero out), detach from graph, then turns on its gradient. """
    assert isinstance(tensor, (torch.Tensor,Variable))
    if tensor.grad is None:
        assert tensor.is_leaf and not tensor.requires_grad
        tensor.requires_grad_(True)
        tensor.retain_grad()
    
    # 📌分析：P2P通信使用double buffer接收数据，会复用之前接收到的tensor，之前的tesnor已经用于梯度计算了。故需要将梯度重新置为
    #         叶子节点、梯度清零。
    else: # double buffer of P2PIn
        tensor.grad.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
        # 梯度清零
        tensor.grad.zero_()
        tensor.detach_()
        assert not tensor.requires_grad
        tensor.requires_grad_(True) # 将输入张量的 requires_grad 属性设置为 True，表示需要计算梯度
        tensor.retain_grad() # 保留输入张量的梯度，以便后续计算。
        assert tensor.grad is not None
            
# 
def turn_on_X_grad(X_named_tensors):
    for name, tensor in X_named_tensors.items():
        if name in ["input0","input1","input2","input_ids"]: # TODO: add identity chain to input (separate identity removal func to outside)
            continue
        if isinstance(tensor, (torch.Tensor,Variable)):
            _turn_on_grad(tensor)
        elif isinstance(tensor, (int,float)):
            continue
        elif isinstance(tensor, list): # output tuple of bert pretrainhead 
            [_turn_on_grad(t) for t in tensor]
        else:
            raise ValueError("unknown tensor={}".format(tensor))

# 把names，tensors装到ODict中返回，若names只有一个，ODict中显然只有一个键值对，不然就是多个键值对
def make_tensors_named(names, tensors):
    assert isinstance(tensors,list)
    named_tensors = ODict()
    if len(names) == 1 and len(tensors) > 1: # output tuple of bert pretrainhead
        named_tensors[names[0]] = tensors # tuple(tensors)
    else:
        for name, tensor in zip(names, tensors):
            named_tensors[name] = tensor
    return named_tensors

# 取出给定tensor的grad tensor，装在named_tensor字典中返回
@torch.no_grad()
def make_dX_from_X(X_named_tensors):
    """ Make reference to .grad of named_tensors for P2P/SWP sending 
        Assumption: 
        1) Not data tensors 
        2) TODO: input identiy chain removed before this call
        3) Except input identiy chain and input const, all tensors has .grad to be sent (to match metas of receiver) 
    """
    dX_named_tensors = ODict()
    for name, X in X_named_tensors.items():
        if isinstance(X,(torch.Tensor, Variable)):
            assert X.requires_grad and X.grad is not None
            dX_named_tensors[name] = X.grad.data
            # assert not X.grad.data.requires_grad
        elif isinstance(X, list): # output tuple of bert pretrainheader
            dX_named_tensors[name] = [ x.grad.data for x in X ]
            assert len(dX_named_tensors[name]) != 0
    return dX_named_tensors

# layerid: 后继BWD任务的最后一层的layer_id，l+1
# 返回前驱BWD任务首层的接收X的元数据，{ name:TensorMeta }
def make_dY_named_metas(XMETA, ubatchsize, layerid): 
    """ Make dY named metas based on profiled XMETA for P2P receiving
        Args: layerid = layer id of dY
        Assumption: The same as 'make_dX_from_X'.
    """
    # remove const, set float32
    named_metas = ODict() # { name:TensorMeta, name:[TensorMeta,TensorMeta] }
    # layerid+1：即前一个BWD任务的第一层
    for name, meta in XMETA.get(ubatchsize,layerid+1).items(): # named metas
        if isinstance(meta, TensorMeta):
            named_metas[name] = TensorMeta(name, meta.shape, dtype=torch.float32)
        elif isinstance(meta, list): # output tuple of bert pretrainheader
            named_metas[name] = [TensorMeta(name, m.shape, dtype=torch.float32) for m in meta]
    # 返回 { name:TensorMeta }
    return named_metas


class SucInfoForPrefetch(object):
    """ Wrapper of finding successor info for all prefetches 
        NOTE: 
        1) don't use cache, which slows down 10x
        2) don't inline _suc_info_* function, which slows down 2x
    """
    def __init__(self, rank, rTASKS, XMETA):
        self.rank_vtasks = rTASKS[rank]
        self.XMETA = XMETA
    
    def set(self, vt, rank_vt_idx):
        self.vt = vt
        self.rank_vt_idx = rank_vt_idx
    
    # 返回当前rank上，给定rank_vt_idx的下一个FWD或BWD任务（会略过UPD任务），即一个vt
    # rank_vt_idx即当前rank任务列表执行到第几个了
    def _find_suc_vt(self, rank_vt_idx, rank_vtasks):
        """ Assumption: max search distance is 2 tasks
            Return: None or found successor vt  """
        if rank_vt_idx + 1 >= len(rank_vtasks):
            return None
        if rank_vtasks[rank_vt_idx+1].type in ['FWD','BWD']:
            return rank_vtasks[rank_vt_idx+1]
        else:
            if rank_vt_idx+2 >= len(rank_vtasks):
                return None
            elif rank_vtasks[rank_vt_idx+2].type not in ['FWD','BWD']:
                return None
            else:
                return rank_vtasks[rank_vt_idx+2]
        
    def _find_case(self, vt, suc_vt):
        """" 
        case-1: current FWD (non-criterion), successor FWD (non-criterion)
        case-2: current FWD (non-criterion), successor BWD (criterion)
        case-3: current FWD (non-criterion), successor BWD (non-criterion) 
        case-4: current BWD (criterion),     successor BWD (non-criterion)
        case-5: current BWD (non-criterion), successor BWD (non-criterion) """
        if vt.type == 'FWD' and suc_vt.type == 'FWD':
            return 1
        #
        elif vt.type == 'FWD' and suc_vt.type == 'BWD' and suc_vt.has_criterion:
            return 2
        elif vt.type == 'FWD' and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 3
        elif vt.type == 'BWD' and vt.has_criterion and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 4
        elif vt.type == 'BWD' and (not vt.has_criterion) and suc_vt.type == 'BWD' and (not suc_vt.has_criterion):
            return 5
        else:
            raise ValueError

    # 返回当前rank上，给定rank_vt_idx的下一个FWD或BWD任务（会略过UPD任务）
    # rank_vt_idx即当前rank任务列表执行到第几个了
    def model(self):
        return self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)

    # 两种情况：
    # 1.下一个任务接收X的媒介为P2P，直接返回None
    # 2.返回 (suc_vt输入X的meta，suc_vt首层的media)
    def _suc_info_X_for_p2p_prerecv(self, suc_vt, XMETA):
        ### In {X} 
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        # 若下一个任务接收X的媒介不为P2P，直接返回None
        if m.medium != "P2P":
            return None
        
        # 返回 (suc_vt输入X的meta，suc_vt接收X的src_rank)
        else: # indeed P2PIn
            return XMETA.get(suc_vt.ubatchszs[0],l), m.rank

    # 两种情况：
    # 1.若suc_vt 这个BWD任务接收dY的媒介不为P2P，返回None
    # 2.返回suc_vt的前驱BWD任务，首层的接收X的元数据，{ name:TensorMeta }，来代表suc_vt最后一层接收dY的元数据
    # 返回 ( { name:TensorMeta }, 来源rank )
    def _suc_info_dY_for_p2p_prerecv(self, suc_vt, XMETA):
        ### In {dY}
        # l: 后继BWD任务的最后一层的layer_id
        l, m = suc_vt.layers[-1], suc_vt.In['dY'][suc_vt.layers[-1]]
        # 若当前BWD的后继BWD任务接收dY的媒介不为P2P，返回None
        if m.medium != "P2P":
            return None
        else: # indeed P2PIn
            # 返回suc_vt BWD任务的前驱BWD任务，首层的接收X的元数据，{ name:TensorMeta }
            return make_dY_named_metas(XMETA, suc_vt.ubatchszs[0], l), m.rank

    # 为后继任务准备输入信息，后继为FWD则准备输入X，后继为BWD则准备输入dY
    # 两种情况：
    # 1.后继任务是FWD任务，或第一个BWD任务（包含计算loss层），为其准备输入X的元数据以及（来源）媒介
    #   两种情况：
    #   1.1.后继任务接收X的媒介不为P2P，直接返回None
    #   1.2.否则，返回 (suc_vt输入X的meta，suc_vt接收X的src_rank)
    # 2.后继任务是BWD任务，则返回后继BWD任务dY的元数据以及（来源）媒介
    #   两种情况：
    #   2.1.若当前BWD的后继BWD任务接收dY的媒介不为P2P，返回None
    #   2.2.否则，返回suc_vt的前驱BWD任务首层的接收X的元数据，{ name:TensorMeta }，来代表suc_vt最后一层接收dY的元数据
    def p2pin(self):
        """ 
        case-1: -> P2PIn(X) @ FWD's ULast
        case-2: -> P2PIn(X) @ FWD's ULast
        case-3: -> P2PIn(dY) @ FWD's ULast
        case-4: -> P2PIn(dY) @ Recompute (criterion)'s ULast
        case-5: -> P2PIn(dY) @ BWD's ULast
        Return: None or 
                ( suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] }, suc_src = 0~3 ) """

        # 返回当前rank上，给定rank_vt_idx的下一个FWD或BWD任务（会略过UPD任务）
        # rank_vt_idx即当前rank任务列表执行到第几个了
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        # 1，2：FWD->FWD，最后一个FWD->第一个BWD(包含计算层)
        if case in [1,2]:
            # 两种情况：
            # 1.下一个任务接收X的媒介不为P2P，直接返回None
            # 2.返回 (suc_vt输入X的meta，suc_vt接收X的src_rank)
            return self._suc_info_X_for_p2p_prerecv(suc_vt, self.XMETA)
        
        # 3，4，5：FWD->不包含计算层的BWD（非第一个BWD任务），第一个BWD->BWD，BWD->BWD
        else:
            # 两种情况：
            # 1.后继BWD任务接收dY的媒介不为P2P，返回None
            # 2.返回suc_vt的前驱BWD任务，首层的接收X的元数据，{ name:TensorMeta }，来代表suc_vt最后一层接收dY的元数据
            # 返回 ( { name:TensorMeta }, 来源rank )
            return self._suc_info_dY_for_p2p_prerecv(suc_vt, self.XMETA)

    # 若suc_vt的输入媒介为MSG，返回suc_vt首层的层号、输入X的元数据
    def _suc_info_X_for_prefetch_msgx(self, suc_vt, XMETA):
        ### In {MSGX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        if m.medium != "MSG": 
            return None
        else: # last FWD convert to first BWD
            return l, XMETA.get(suc_vt.ubatchszs[0],l)

    # 若当前vt和后继vt的情况为：FWD -> 首个BWD，且suc_vt的输入X的媒介为MSG，返回suc_vt首层的层号、输入X的元数据
    def msgx(self):
        """ 
        case-2: -> MSGIn(X) @ FWD's ULast
        
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] } ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        if case in [2]:
            # 若suc_vt的输入媒介为MSG，返回suc_vt首层的层号、输入X的元数据
            return self._suc_info_X_for_prefetch_msgx(suc_vt, self.XMETA)
        else:
            return None

    def _suc_info_stashx_for_prefetch_stashx(self, suc_vt, XMETA):
        ### In {StashX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        # 若后继任务输入StashX的媒介不是MSG，直接返回None
        if m.medium != "MSG":
            return None
        else:
            return l, XMETA.get(suc_vt.ubatchszs[0],l) 

    # 若后继任务是BWD（非第一个BWD），且输入媒介是MSG，返回 (l(后继任务的首层id), 后继任务输入X的元数据) 。非MSG直接返回None
    # 其他情况直接返回None
    def stashx(self):
        """ 
        For StashX in vPP:
        case-3: -> SwapIn(StashX) @ FWD's ULast
        case-4: -> SwapIn(StashX) @ Recompute (criterion)'s ULast
        case-5: -> SwapIn(StashX) @ Recompute (non-criterion)'s ULast
        For StashX in vDP:
        case-3: -> doesn't exist
        case-4: -> same action
        case-5: -> same action
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] ) """
        # 返回当前rank上，给定rank_vt_idx的下一个FWD或BWD任务（会略过UPD任务），即一个vt
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        # 3，4，5：FWD->不包含计算层的BWD（非第一个BWD任务），第一个BWD->BWD，BWD->BWD
        # 若当前任务是BWD，后继任务也是BWD（还有一种情况，当前是最后一个FWD任务，但下一个任务不是第一个BWD任务，这个不太懂）
        if case in [3,4,5]:
            return self._suc_info_stashx_for_prefetch_stashx(suc_vt, self.XMETA)
        else:
            return None

    # 若suc_vt首层接收X的媒介不为SWP，返回None，
    # 否则返回 (l(suc_vt的首层)，l这层接收X的元数据)
    def _suc_info_X_for_prefetch_localx(self, suc_vt, XMETA):
        ### In {LocalX}
        l, m = suc_vt.layers[0], suc_vt.In['X'][suc_vt.layers[0]]
        # 若suc_vt首层接收X的媒介不为SWP，返回None
        if m.medium != "SWP": # swap locally for vDP
            return None
        # 否则返回 (l(suc_vt的首层)，l这层接收X的元数据)
        else:
            return l, XMETA.get(suc_vt.ubatchszs[0],l)

    # 若suc_vt最后一层接收dY的媒介不为SWP，返回None，
    # 否则返回 (l+1(suc_vt的最后一层+1即为当前vt的首层)，l+1这层接收X的元数据)
    def _suc_info_dY_for_prefetch_localx(self, suc_vt, XMETA):
        ### In {dY}
        l, m = suc_vt.layers[-1], suc_vt.In['dY'][suc_vt.layers[-1]]
        if m.medium != "SWP": # swap locally for vDP
            return None
        else:   
            return l+1, make_dY_named_metas(XMETA, suc_vt.ubatchszs[0], l)

    # 为后继任务准备输入信息(元数据)，后继为FWD/首个BWD则准备输入X，后继为BWD则准备输入dY
    # 两种情况
    # 1.FWD->FWD、最后一个FWD->第一个BWD(包含计算层)
    #   若suc_vt首层接收X的媒介不为SWP，返回None，否则返回 (l(suc_vt的首层)，l这层接收X的元数据)
    # 2.首个BWD->BWD、BWD(非首个)->BWD(非首个)
    #   若suc_vt最后一层接收dY的媒介不为SWP，返回None，否则返回 (l+1(suc_vt的最后一层+1即为当前vt的首层)，l+1这层接收X的元数据)
    def localx(self):
        """ 
        case-1: -> SwapIn(X) @ FWD's ULast
        case-2: -> SwapIn(X) @ FWD's ULast
        case-4: -> SwapIn(dY) @ BWD's ULast
        case-5: -> SwapIn(dY) @ BWD's ULast
        Return: None or 
                ( suc_layer_id, suc_named_metas = { name:TensorMeta, name:ConstMeta, name:[TensorMeta,TensorMeta] ) """
        suc_vt = self._find_suc_vt(self.rank_vt_idx, self.rank_vtasks)
        if suc_vt is None:
            return None
        case = self._find_case(self.vt, suc_vt)
        # 1，2：FWD->FWD，最后一个FWD->第一个BWD(包含计算层)
        if case in [1,2]:
            # 若suc_vt首层接收X的媒介不为SWP，返回None，
            # 否则返回 (l(suc_vt的首层)，l这层接收X的元数据)
            return self._suc_info_X_for_prefetch_localx(suc_vt, self.XMETA)
        # 4：首个BWD->BWD
        # 5：BWD(非首个)->BWD(非首个)
        elif case in [4,5]:
            # 若suc_vt最后一层接收dY的媒介不为SWP，返回None，
            # 否则返回 (l+1(suc_vt的最后一层+1即为当前vt的首层)，l+1这层接收X的元数据)
            return self._suc_info_dY_for_prefetch_localx(suc_vt, self.XMETA)
        else:
            return None


# def make_dX_from_X(X_named_tensors): # for send
#     dX_named_tensors = ODict()
#     for name, X in X_named_tensors.items(): # only tensor & required_grad can run autograd
#         if isinstance(X,(torch.Tensor, Variable)) and (X.requires_grad):
#             if X.grad is not None:
#                 dX_named_tensors[name] = X.grad.data
#                 assert not X.grad.data.requires_grad
#         elif isinstance(X, list): # output tuple of bert pretrainheader
#             multi_grad = []
#             for x in X:
#                 if isinstance(x,(torch.Tensor, Variable)) and (x.requires_grad):
#                     if x.grad is not None:
#                         multi_grad.append(x.grad.data)
#                         assert not x.grad.data.requires_grad
#             if len(multi_grad) != 0:
#                 dX_named_tensors[name] = multi_grad
#     # NOTE: alreay considered gradient of identiy chain to input (separate identity removal func to outside)
#     return dX_named_tensors # ready to use for swp & msg & p2p send

# def make_dX_meta_from_X(X_named_tensors): # for receive
#     dX_named_metas = ODict() # { name:TensorMeta, name:[TensorMeta,TensorMeta] }
#     for name, X in X_named_tensors.items(): 
#         if isinstance(X,(torch.Tensor, Variable)): # and (X.requires_grad): # only tensor & required_grad can run autograd
#             dX_named_metas[name] = TensorMeta(name, X.shape, dtype=torch.float32)
#         elif isinstance(X, list): # output tuple of bert pretrainheader
#             multi_grad = []
#             for x in X:
#                 if isinstance(x,(torch.Tensor, Variable)): # and (x.requires_grad):
#                     multi_grad.append(TensorMeta(name, x.shape, dtype=torch.float32))
#             if len(multi_grad) != 0:
#                 dX_named_metas[name] = multi_grad
#     # NOTE: when grad of identity chain to input exsits, need to ignore X.requires_grad, i.e., always make dX meta
#     # TODO: when removing grad. of identity chain to input, only X.requires_grad can make dX. (separate identity removal func to outside)
#     return dX_named_metas # ready to use for p2p-recv
