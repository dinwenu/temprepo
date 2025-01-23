# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict as ODict
from copy import deepcopy
import pickle
import os
import json

MediumMustMatchIdxRank = ["P2P","MSG"]
MediumMustMatchIdx = ["SWP","PIN"]

class Medium():
    """ a.k.a Channel """
    def __init__(self, medium=None, idx=None, tasks=None):
        assert medium in ["P2P","MSG","SWP","PIN","SHM","LOC","DAT",None]
        self.medium = medium
        self.idx = idx # single src/dst task idx # 表示来源或目标任务的idx
        if idx is None:
            self.rank = None
        else:
            if tasks is None:
                self.rank = None
            else:
                # 将tasks中满足 idx属性=value(该初始化函数的第二个参数) 的task以列表形式返回，在通过下标取出来
                # 返回idx属性为idx的vtask
                # 简单来说，就是把idx这个任务从整个TASK中筛选出来
                task = filter_tasks_by_attr_val(tasks, attr='idx', value=idx)[0]
                # 将rank属性设置为与返回的task的rank属性相同
                self.rank = task.rank # single src/dst rank
    
    def __call__(self):
        return self.medium

    def __str__(self): # "DAT[-/-]" or "P2P[2/1]" or "SWP[1/-]"
        if hasattr(self, 'dst_idx_T'):
            return "{}[{}/{}](dstT{}/{})".format(
                    '-' if self.medium is None else self.medium, 
                    '-' if self.idx is None else self.idx, 
                    '-' if self.rank is None else self.rank,
                    self.dst_idx_T, self.dst_rank_T)
        else:
            # Medium[idx(源或目标任务的idx)/rank(GPU序号)]
            return "{}[{}/{}]".format(
                    '-' if self.medium is None else self.medium, 
                    '-' if self.idx is None else self.idx, 
                    '-' if self.rank is None else self.rank)

    def __eq__(self, other):
        if dir(self) != dir(other): 
            return False
        # traverse all attributes, compare non-callables
        for attr_name in dir(self): 
            self_attr, other_attr = getattr(self, attr_name), getattr(other, attr_name)
            if not callable(self_attr) or not callable(other_attr):
                if self_attr != other_attr:
                    return False
        # compare __str__() results
        if self.__str__() != other.__str__():
            return False
        return True # both non-callable and __str__ equals
        # ref: calling dir on the object gives you back all the attributes of that object (sorted): ['__class__', '__delattr__', ..., 'bar', 'foo', 'func'] 
        # https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python

    # def set_for_T(self, T_tasks):
    #     ''' FWD’s X[0]:
    #             DAT.dst -> closest FWD
    #             DAT.dst -> closest BWD
    #         FWD/BWD’s T[R-1]: 
    #             DAT.src -> X[0]
    #     '''
    #     if self.medium == 'DAT':
    #         self.dst_idx_T = []
    #         self.dst_rank_T = []
    #         for vt in T_tasks:
    #             self.dst_idx_T.append(vt.idx)
    #             self.dst_rank_T.append(vt.rank)

class vTask():
    def __init__(self, idx=None, layers=None, type=None, ubatchszs=None, device=None):
        self.idx = idx # execution order of this task: 0,1,3 ...
        self.layers = layers # [0] for unpacked or [0,1] for packed layers; always ascending; used for RT compute
        self.type = type # 'FWD', 'BWD', 'UPD'
        # u：micro-batch size，即1个micro batch有几个数据，所有的u加起来就是1个minibatch的大小
        # data_batchsize/D：data_batchsize即一共有多少数据，D即minibatch size，1个minibatch有几个数据。相除即为minibatch
        # 的数量。
        self.ubatchszs = ubatchszs # [u1, u2, u3] means three grouped microbatches where each u is a microbatch size, so total sum is data_batchsize/D
        self.device = device # bind to which hardware: "GPU:0" (GPU-ID 0) or "CPU:0" (CPU-core 0), where "GPU:0" and "CPU:0" are run by the same process "rank 0".
        self.In = ODict() # required inputs (see swap model)
        self.Out = ODict() # requried outputs (see swap model)
        ''' In/Out example of vPP/vDP
        FWD:  
            In { 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() } }
            Out { 'Y': { L1:Medium() }, 'X': { L1:Medium() }, 'W': {}, 'B': {} }
        BWD[LLast]:  
            In { 'dY':{}, 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {LLast:Medium()} }
        BWD:  
            In { 'dY': { L1:Medium() }, 'X': { L0:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'B': { L0:Medium(), L1:Medium() }, 'T': {} }
            Out { 'dX': { L0:Medium() }, 'dW': { L0:Medium(), L1:Medium() }, 'W': {}, 'B': { L0:[Medium(),Medium()], L1:Medium() } }
        UPD:  
            In { 'dW': { L0:Medium(), L1:Medium() }, 'W': { L0:Medium(), L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
            Out { 'W': { L0:[Medium(),Medium()], L1:Medium() }, 'K': { L0:Medium(), L1:Medium() } }
        '''
        self.has_data = False # flag for T
        self.has_criterion = False # flag for T, and loss
        self.is_last_fwd = False # flag for the last FWD (non-criterion) task
    
    # 返回当前task所在的设备号，即从device中把数字提取出来，GPU：0 -> 0
    @property
    def rank(self):
        return int(self.device[4:])

    def set_new_rank(self, new_rank):
        assert isinstance(new_rank, int)
        self.device = self.device[:4] + str(new_rank)
        assert self.rank == new_rank

    @property
    def is_gpu(self):
        return self.device[:3] == 'GPU'

    @property
    def groupsz(self):
        return None if self.ubatchszs is None else len(self.ubatchszs)

    def show_layers(self):
        if self.layers is None:
            return "-"
        assert list(range(self.layers[0], self.layers[-1]+1)) == list(self.layers)
        if len(self.layers) == 1:
            return "L%d" % (self.layers[0])
        else:
            return "L%d-%d" % (self.layers[0], self.layers[-1])
    
    def show_ubatchszs(self):
        if self.ubatchszs is None:
            return "-"
        ubs = self.ubatchszs[0]
        tail = self.ubatchszs[-1]
        if ubs == tail:
            return "U%dx%d" % ( ubs, len(self.ubatchszs) )
        else:
            return "U%dx%d+%d" % ( ubs, len(self.ubatchszs)-1, tail )
    
    def show_inout(self, inout): 
        # "X:{L0:DAT[-/-]}, W:{L0:SHM[-/-],L1:SHM[-/-]}, ..., dW:{L0:SWP[0/-]-Coll[], ...}"
        assert isinstance(inout, ODict)
        key_strs = []
        for key, dic in inout.items():
            layer_strs = []
            for l, med in dic.items(): 
                if isinstance(med, Medium):
                    layer_strs.append( "L%d:%s"%(l,med) )
            key_str = ",".join(layer_strs)
            key_str = "%s:{%s}" % (key,key_str)
            key_strs.append(key_str)
        return ", ".join(key_strs)
    
    def __str__(self):
        # "< Index, Layer(s), Type, MicroBatchSize(s), Device, RequiredIn<X,dY,W,dW,B,K>, RequiredOut<X,dX,Y,W,dW,B,K> >\n"
        return  "<{}, {}, '{}', {}, '{}', In<{}>, Out<{}>>".format(
                "-" if self.idx is None else "%d"%(self.idx),
                self.show_layers(), # "-" if self.layers is None else "-".join([str(l) for l in self.layers]),
                "-" if self.type is None else "%s"%(self.type),
                self.show_ubatchszs(), # '-' if self.ubatchszs is None else "-".join([str(u) for u in self.ubatchszs]),
                "-" if self.device is None else "%s"%(self.device),
                self.show_inout(self.In),
                self.show_inout(self.Out))

    def __eq__(self, other):
        if dir(self) != dir(other): 
            return False
        # traverse all attributes, compare non-callables
        for attr_name in dir(self): 
            self_attr, other_attr = getattr(self, attr_name), getattr(other, attr_name)
            if not callable(self_attr) or not callable(other_attr):
                if self_attr != other_attr:
                    return False
        # compare __str__() results
        if self.__str__() != other.__str__():
            return False
        return True # both non-callable and __str__ equals

# 将tasks中所有满足attr=value的task组成一个新列表返回
def filter_tasks_by_attr_val(tasks, attr, value):
    
    def fn(vt):
        if attr in ['idx','type','device','has_data','has_criterion', 'is_last_fwd']:
            return getattr(vt,attr) == value
        elif attr in ['layers']:
            return value in getattr(vt,attr)
        elif attr in ['medium']:
            for key, lm in vt.In.items():
                for l, m in lm.items():
                    assert isinstance(m, Medium)
                    if m.medium == value:
                        return True
            for key, lm in vt.Out.items():
                for l, m in lm.items():
                    if isinstance(m, list):
                        for mm in m:
                            assert isinstance(mm, Medium)
                            if mm.medium == value:
                                return True
                    else:
                        assert isinstance(m, Medium)
                        if m.medium == value:
                            return True
            return False
        elif attr in ['rank']:
            return vt.rank == int(value)
        else:
            raise ValueError("Unknown task attribute {}".format(attr))
    
    # filter list out-of-place by filter function taking each element on True
    assert isinstance(tasks, list)
    filtered = filter(fn, tasks)
    return list(filtered)

def sort_tasks_by_attr(tasks, attr, reverse=False):
    
    def fn(vt):
        if attr in ['idx']:
            return int(getattr(vt, attr))
        elif attr == 'layers':
            return int(getattr(vt, attr)[0])
        elif attr == 'type':
            if getattr(vt, attr) == 'FWD':
                return 0
            elif getattr(vt, attr) == 'BWD':
                return 1
            elif getattr(vt, attr) == 'UPD':
                return 2
            else:
                assert False
        elif attr == 'device':
            return getattr(vt, attr)
        elif attr in ['In','Out']:
            raise ValueError("Underdevelopment")
        else:
            raise ValueError("Underdevelopment")
    
    # sort list out-of-place by key function taking each element
    assert isinstance(tasks, list)
    return sorted(tasks, key=fn, reverse=reverse)

# 对layers中的每一层执行：尝试从tasks中取出layers属性含有当前这一层的task，若存在这种task，
# 建立一个字典{l: 该task的task_idx}
def find_dependent_tasks_for_layers(tasks, layers):
    """ find tasks that contain these layers; return their indices by {l:task_idx} """
    found = ODict()
    # 对vt中的所有layer，
    for l in sorted(layers):
        # 取出tasks中，layers属性包含 l（当前遍历到的层）的task。这也说明 l 是其所在pack的首层
        # 将tasks中所有满足attr=value的task组成一个新列表返回
        found_vts = filter_tasks_by_attr_val(tasks, attr='layers', value=l)
        if len(found_vts) == 0:
            pass
        elif len(found_vts) == 1:
            found[l] = found_vts[0].idx
        else:
            raise ValueError("Found should be at most a single task!")
    
    return found

# 遍历每一个GPU('GPU:%d')，取出其上的所有任务。若当前设备上既有FWD任务也有BWD任务：取出最后一个FWD任务
# 和第一个BWD任务，对最后一个FWD任务中的每一层，若该层也存在于第一个BWD任务中，说明可被打包，即以just-in-time的方式执行
def find_jit_fwdbwd_pairs_in_one_iter(tasks, N):
    """ find task-pair of fwd and bwd that are jit (i.e., can be pinned on GPU) """
    assert isinstance(tasks, list)
    found = ODict() # { 'GPU:0': [fwd_idx,[L1,L2],bwd_idx], ... }
    # 遍历每一个GPU序号：[0,1,2,3]
    for i in range(N):
        device = 'GPU:%d'%i
        # 取出所有 device=当前GPU设备 的task
        device_tasks = filter_tasks_by_attr_val(tasks, attr='device', value=device)
        # 筛选出FWD任务
        fwd_tasks = filter_tasks_by_attr_val(device_tasks, attr='type', value='FWD')
        fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='idx')
        # 筛选出BWD任务
        bwd_tasks = filter_tasks_by_attr_val(device_tasks, attr='type', value='BWD')
        bwd_tasks = sort_tasks_by_attr(bwd_tasks, attr='idx')
        # 若当前设备上既有FWD任务也有BWD任务
        if len(fwd_tasks) != 0 and len(bwd_tasks) != 0:
            # 取出最后一个FWD任务和第一个BWD任务
            last_fwd, first_bwd = fwd_tasks[-1], bwd_tasks[0]
            # 对最后一个FWD任务中的每一层 l
            for l in last_fwd.layers:
                # 若该层也存在于第一个BWD任务中，说明可被打包，即以just-in-time的方式执行
                if l in first_bwd.layers: # JIT
                    if device not in found:
                        found[device] = [last_fwd.idx,[l],first_bwd.idx]
                    else:
                        found[device][1].append(l)
    # print("[find_jit_fwdbwd_pairs_in_one_iter] found={}".format(found))
    return found

# 深拷贝一份tasks，但所有task的vlayers属性只保留第一个值（该Pack的首层）
def leave_first_layer_in_each_task(tasks):
    """ return a copy of input tasks that are left with only first layer; for recompute-based stashing """
    tasks_w_only_1st_layer = deepcopy(tasks)
    for vt in tasks_w_only_1st_layer:
        vt.layers = [sorted(vt.layers)[0]] # force ascending order
    
    return tasks_w_only_1st_layer

# 为layers属性中包含第1层的vt的has_data属性赋值为true
# 为layers属性中包含最后一层的vt的has_criterion属性赋值为true
def set_has_data_criterion(tasks, R):
    for vt in tasks:
        # 检查vt的 layers 列表是否包含值为 0 的元素。如果包含，则将vt的 has_data 属性设置为 True
        vt.has_data = 0 in vt.layers
        # 检查vt的 layers 列表是否包含值为 R-1(表示最后一层) 的元素。如果包含，则vt的 has_criterion 属性设置为 True
        vt.has_criterion = (R-1) in vt.layers

# 设置最后一个fwd任务的is_last_fwd属性为true
def set_the_last_fwd(tasks):
    fwd_tasks = filter_tasks_by_attr_val(tasks, attr='type', value='FWD')
    if fwd_tasks == []: # a single BWD task
        return
    
    fwd_tasks = sort_tasks_by_attr(fwd_tasks, attr='layers')
    the_idx = fwd_tasks[-1].idx
    # 这个if用来确保tasks列表是顺序的
    if the_idx < len(tasks) and tasks[the_idx] == fwd_tasks[-1]: # ordered
        tasks[the_idx].is_last_fwd = True
        # print("\nset_the_last_fwd = True for {}".format(tasks[the_idx]))
    else: # unordered
        for vt in tasks:
            vt.is_last_fwd = vt.idx == the_idx
        
def print_tasks(tasks, name="TASKS"):
    print("\n----- {} -----".format(name))
    if isinstance(tasks, list):
        for vt in tasks:
            print(vt)
    elif isinstance(tasks, ODict) or isinstance(tasks, dict):
        for idx, vt in tasks.items():
            print(vt)

def make_dtasks(tasks):
    assert isinstance(tasks, list)
    dtasks = ODict() # make tasks to ODict ordered by index { idx: vt of idx }
    for vt in sort_tasks_by_attr(tasks, attr='idx', reverse=False):
        dtasks[vt.idx] = vt
    
    return dtasks

# 对每一个GPU序号执行：
# 1.将GPU序号相同的task全拿出来
# 2.将拿出来的task按照自身的idx属性排序
# 3.建立一个字典 {GPU序号:[该GPU上的所有task]}
# 返回这个字典
def make_rank_based_task_queue(tasks, N):
    ''' use tasks (ordered by idx globally) to make rtasks (w/ rank correctness, and per-rank ascending idx)
    '''
    assert isinstance(tasks, list)
    for i, vt in enumerate(tasks): # must be ordered by idx globally
        assert i == vt.idx
    
    rtasks = ODict() # { rank<0>: [task0,task2,task5,...] }
    for r in range(N):
        # 将GPU序号相同的task全拿出来
        rank_tasks = filter_tasks_by_attr_val(tasks, attr='rank', value=r)
        # 将拿出来的task按照自身的idx属性排序
        rank_tasks = sort_tasks_by_attr(rank_tasks, attr='idx', reverse=False)
        # 建立一个字典 {GPU序号:[该GPU上的所有task]}
        rtasks[r] = rank_tasks
    
    return rtasks

def unmake_rank_based_task_queue(rtasks):
    ''' use rtasks (w/ rank correctness, and per-rank ascending idx) to make tasks (ordered by idx globally)
    '''
    tasks = []
    for r, vts in rtasks.items(): # { rank0: [task0,task2,task5,...] }
        for vt in vts:
            assert r == vt.rank
            tasks.append(vt)
        assert vts == sort_tasks_by_attr(vts, attr='idx', reverse=False)
    tasks = sort_tasks_by_attr(tasks, attr='idx', reverse=False)
    for i, vt in enumerate(tasks): # must be ordered by idx globally
        assert i == vt.idx
    
    return tasks

# 将传入的tasks列表中的任务，按照所在的GPU序号进行重组。即建立一个字典，在同一个GPU上的任务组成一个list对应一个GPU序号
# 对每一个GPU序号执行：
# 1.将GPU序号相同的task全拿出来
# 2.将拿出来的task按照自身的idx属性排序
# 3.建立一个字典 {GPU序号:[该GPU上的所有task]}
# 返回这个字典
# { rank0: [task0,task2,task5,...], rank1:... }
def convert_to_per_rank_task_queue(tasks, N, verbose=True):
    # 对每一个GPU序号执行：
    # 1.将GPU序号相同的task全拿出来
    # 2.将拿出来的task按照自身的idx属性排序
    # 3.建立一个字典 {GPU序号:[该GPU上的所有task]}
    # 返回这个字典
    rTASKS = make_rank_based_task_queue(tasks, N)
    if verbose: print_rtasks(rTASKS, name="rTASKS")

    # 将上面字典中的所有任务全部放入一个list中，并重新按照idx排序，检测是否能恢复成原始的tasks
    assert unmake_rank_based_task_queue(rTASKS) == tasks
    
    return rTASKS
    
def print_rtasks(rtasks, name="rTASKS"):
    print("\n----- {} -----".format(name))
    assert isinstance(rtasks, ODict)
    for r, vts in rtasks.items():
        print("rank{} has:".format(r))
        for vt in vts:
            print(vt)

def serialize_scheduled(rtasks, configs, path_dir, fname, base_dir="sched"):
    if ".pickle" not in fname:
        fname += ".pickle"
    assert os.path.exists(path_dir)
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_dir = os.path.join(path_dir, base_dir)
        os.makedirs(full_dir, exist_ok=True)
        # 文件夹路径+文件名(fname)，即最终要写入的full_path
        full_path = os.path.join(full_dir, fname)
    
    # 把rtasks和configs写入到文件中
    with open(full_path,'wb') as f:
        pickle.dump([rtasks, configs], f)
    print("schedule serialized to: {}".format(full_path))

# 从指定路径的文件中反序列化（unserialize）调度（schedule）数据
def unserialize_scheduled(path_dir, fname, base_dir="sched", verbose=True):
    if ".pickle" not in fname:
        fname += ".pickle"
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_path = os.path.join(path_dir, base_dir, fname)
    print(f"full_path: {full_path}")
    assert os.path.exists(full_path)
    
    with open(full_path,'rb') as f:
        rtasks, configs = pickle.load(f)
    print("schedule unserialized from: {}".format(full_path))
    if verbose: 
        print_rtasks(rtasks, name="rTASKS")
        print("CONFIGS={}".format(configs))
    
    return rtasks, configs

