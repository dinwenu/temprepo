# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from math import ceil, floor

# 将传进来的list转化为两个list。
# 一个layer_pack，即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号。
# 一个per_pack_memories，每个元素表示对应位置的layer pack的空间占用（bytes）
def convert_to_layer_packs(packed_memories): 
    # packed_memories: [array([100, 50, 110]), array([100, 50, 110])] # bytes
    if packed_memories == []:
        return []
    assert isinstance(packed_memories, list)
    assert isinstance(packed_memories[0], np.ndarray) 
    
    # convert to layer packs
    cnt = 0
    layer_packs = [] # [ [0,1,2], [3,4,5] ]
    per_pack_memories = [] # [ 260, 260 ]
    for pack in packed_memories:
        num_layers = len(pack)
        assert num_layers > 0, "no empty pack allowed"
        layer_packs.append( list(range(cnt, cnt+num_layers)) )
        cnt += num_layers
        per_pack_memories.append(pack.sum())
    # assert cnt == len(per_layer_memories)
    
    return layer_packs, per_pack_memories

def print_memory_packing(layer_packs, per_pack_memories, title="", tab="\t"):
    assert isinstance(layer_packs, list) and isinstance(per_pack_memories, list)
    memories = np.array(per_pack_memories)
    print("%s-------%s-------"%(tab,title))
    for i, (layers, mem) in enumerate(zip(layer_packs, memories)):
        print("%s#%04d: L%04d-%04d: %6.0f MB"%
              (tab, i, layers[0], layers[-1], mem/1024./1024.))
    print("%sper_pack_memories: mean %.0f, std %.0f, max %.0f, min %.0f MB"%
        (tab, memories.mean()/1024./1024., memories.std()/1024./1024., memories.max()/1024./1024., memories.min()/1024./1024.))

# per_layer_memories：一个list，里面装着每一层的空间占用
# capacity：GPU的容量（in bytes）
# 使用贪婪方法打包层，即只要当前pack中所有的层+首层的输入不超过GPU的容量，就一直往pack中添加新的layer。否则，开始下一个pack
# 返回：layer_pack（list），即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
def greedy_memory_packing(per_layer_memories, capacity, 
                        reverse=False, per_layer_x=None,
                        verbose=False, title="greedy_memory_packing", tab="\t"):
    """ 
    Arguments:
        per_layer_memories: read-only

        reverse: if False, greedy packing from 1st to last layer, leave fraction in the last criterion pack.
                 if True, greedy packing from last to 1st layer, leave fraction in the 1st pack. 
                 packed layer id is always ascending. 
        
        per_layer_x: if not None, add back x size in each pack; otherwise, more pack will have less memory (x stripped off) when summing up per_layer_memories; ready-only
    """
    
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    # if verbose: print("\tper_layer_memories (MB): {}".format(memories/1024./1024.)) 
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    if per_layer_x is None:
        xmems = np.zeros(len(memories))
    else:
        xmems = np.array(per_layer_x)
        assert len(xmems) == len(memories)
    
    # pack memories by accumulating to capacity
    if reverse is False:
        packed_memories = [] # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        # 首层的输入+首层参数大小首层的输入+首层参数大小
        new_pack_sum = memories[0] + xmems[0]
        # assert new_pack_sum < capacity, "capacity cannot hold even one layer"  
        # 若首层的空间占用超过GPU容量，直接打印信息并返回
        if new_pack_sum >= capacity:
            if verbose: print("capacity cannot hold even one layer; return None")
            return None
        new_pack = [ new_pack_sum ]
        for mem, x in zip(memories[1:], xmems[1:]): # until the last layer
            new_pack_sum += mem
            if new_pack_sum < capacity:
                new_pack.append(mem)
            else:
                packed_memories.append(np.array(new_pack))
                new_pack_sum = mem + x
                # assert new_pack_sum < capacity, "capacity cannot hold even one layer"  
                if new_pack_sum >= capacity:
                    if verbose: print("capacity cannot hold even one layer; return None")
                    return None
                new_pack = [ new_pack_sum ]
        packed_memories.append(np.array(new_pack))
    else:
        # reverse layer order
        memories = np.flip(memories) # return a view of m with the entries reversed (underlying memory shared)
        xmems = np.flip(xmems)
        # pack
        packed_memories = [] # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        # 📌在layer反转的情况下，空间占用在一开始不会加上最后一个layer的输入占用，只保存最后一层的参数+输出激活
        # 📌分析：肯定不能加上输入，因为现在是逆序，算下一层的空间占用时候本身就包括下一层的输出激活，这个激活就是当前层的输入，肯定不能加上啊
        new_pack_sum = memories[0] # the last layer
        # assert new_pack_sum + xmems[0] < capacity, "capacity cannot hold even one layer"
        # 先检列表中首个层(最后一层)总的空间占用是否大于GPU的容量，若大于直接返回，后面也不用看了
        # 首层的输入+首层参数大小+首层输出激活
        if new_pack_sum + xmems[0] >= capacity:
            if verbose: print("capacity cannot hold even one layer; return None")
            return None
        new_pack = [ new_pack_sum ]
        # 保留最后一层的输入大小
        prev_x = xmems[0]
        # 第0个值上面已经检查了，从第一个值，即倒数第二个层开始，遍历到首层
        for mem, x in zip(memories[1:], xmems[1:]): # until the first layer
            # 加上当前layer的空间占用与输出大小
            # 📌此赋值不改变new_pack这个list中存的值
            new_pack_sum += mem
            # 若加上当前layer的参数、输出激活、输入后，依然不超过GPU的空间占用
            # 就将当前层的空间占用（参数+输出）加入到 new_pack 列表的末尾
            if new_pack_sum + x < capacity:
                # 只记录当前层的参数+激活大小
                new_pack.append(mem)
            else:
                # 把prev_x，即当前layer pack（即new_pack）这个list中最后一个layer层的输入大小（现在是逆序的，
                # 正序看是当前pack的首个layer），累加到new_pack的最后一个值中
                # ❓new_pack的最后一个值，不应该加上pack中最后一层的输入吗？为什么加上首层的输入？
                # ✅答：看for循环的最后一句，从正序的视角来看，就是把首层的输入给算上
                new_pack[-1] += prev_x
                # 将pack加入到packed_memories列表中，代表完成了一个pack的打包
                packed_memories.append(np.array(new_pack))
                # 新pack的大小即从当前layer开始，其大小被初始化为当前layer的空间占用（参数+激活）
                new_pack_sum = mem
                # assert new_pack_sum + x < capacity, "capacity cannot hold even one layer"
                # 若新pack的首个layer的大小+其输入的大小超过的GPU容量，跟上面的逻辑一样，直接结束
                if new_pack_sum + x >= capacity:
                    if verbose: print("capacity cannot hold even one layer; return None")
                    return None
                # 
                new_pack = [ new_pack_sum ]
            
            # prev_x 持续更新为当前layer的输入大小
            # ✅
            prev_x = x

        # 将最后一个pack的最后一个layer，即顺序上的第一个layer的输入大小累加到对应位置上
        new_pack[-1] += prev_x
        # 将最后一个pack加入到packed_memories列表中
        packed_memories.append(np.array(new_pack))
        # # confirm add x into each pack
        # cnt = 0
        # for pack in packed_memories: 
        #     cnt += len(pack)
        #     idx = cnt - 1
        #     assert pack[-1] == memories[idx] + xmems[idx]
        # restore layer order

        # 将 packed_memories 这个列表逆序
        packed_memories.reverse() # inplace
        # 将packed_memories中的每个打包好的pack进行翻转
        for i in range(len(packed_memories)):
            packed_memories[i] = np.flip(packed_memories[i])
    
    # confirm correctness
    assert len(memories) == sum(len(pack) for pack in packed_memories)
    if per_layer_x is None:
        assert memories.sum() == sum(pack.sum() for pack in packed_memories)
    assert max(pack.sum() for pack in packed_memories) < capacity
    # if verbose: print("\tpacked_memories (MB): {}".format(
    #                    [pack/1024./1024. for pack in packed_memories]))
    # convert to layer packs

    # 将传进来的list转化为两个list。
    # 一个layer_pack，即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号。
    # 一个per_pack_memories，每个元素表示对应位置的layer pack的空间占用（bytes）
    layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
    
    # 
    if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
    # 最终只返layer_pack
    return layer_packs

# 将数组 A 分割成 S 个子数组（或称为“packs”），以使这些子数组的元素之和大致相等
# 返回一个list，里面装着拆分后的sub_list
def _balanced_sum_split(A, S):
    """ split 'A' into 'S' 'packs' such that 'packs' have approximately equal sum """

    assert isinstance(A, np.ndarray) and A.dtype in (np.int64, np.float64), "A = {} ({})".format(A, A.dtype)
    # get prefix sum
    # 每一个位置都是包括该位置值的前缀和
    prefix_sum = A.cumsum()
    # approximate per-pack sum
    # 大概的per pack的平均执行时间：总的执行时间 ÷ 份数S
    pack_sum = prefix_sum[-1] // S if A.dtype == np.int64 else prefix_sum[-1] / S
    # get prefix sum of per-pack sum
    # 得到S-1个前缀和，若S=2，结果为[pack_sum，pack_sum*2]
    prefix_pack_sum = np.array(range(1, S)) * pack_sum 
    # binary search the indices such that the prefix per-pack sums are inserted
    # 找到prefix_pack_sum应该插入的位置，这个位置原本的数即第一个大于等于prefix_sum的数，即最接近prefix_sum的值
    indices = np.searchsorted(prefix_sum, prefix_pack_sum, side='left')
    # split into approximately equal-sum packs
    # 在该位置分裂数组A
    packs = np.split(A, indices) # a list of sub-arrays as views into A (underlying memory is shared between sub-arrays as A)
    
    return packs # [array([0, 1, 2]), array([3, 4, 5])] (memory shared with input A)
 
# per_layer_memories：一个list，里面装着每一层的空间占用
# 
def balanced_memory_packing(per_layer_memories, capacity, 
                            per_layer_x=None, 
                            verbose=False, title="balanced_memory_packing", tab="\t"):
    """ 
    Argument:
        per_layer_memories: read-only
        capacity: a forced constraint during packing
        per_layer_x: if not None, add back x size in each pack; otherwise, more pack will have less memory (x stripped off) when summing up per_layer_memories; ready-only
    """
    
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    # if verbose: print("\tper_layer_memories (MB): {}".format(memories/1024./1024.)) 
    memories = np.array(per_layer_memories) # a numpy object with a new memory (int)
    if per_layer_x is not None:
        assert len(per_layer_x) == len(memories)
    
    # find num of packs (initial guess)
    # 直接使用模型总的空间占用除以GPU的容量，得到应该打包的数量
    num_packs = ceil(memories.sum()/float(capacity)) # ceil:向上取整
    
    # parition into num of packs (under capacity constraint)
    packed_memories = None
    # print(f"num_packs:{num_packs}, len(memories):{len(memories)}")
    # exit(0)
    for S in range(num_packs, len(memories)+1):
        # balance the memory per pack
        packed = _balanced_sum_split(memories, S) # [array([100, 50, 110]), array([100, 50, 110])] # bytes
        # check empty pack
        if sum([ int(len(pack)==0) for pack in packed ]) != 0:
            if verbose: print("\tbalanced %d packs has empty pack; try more packs"%S) 
            continue
        # check memory of each pack
        if per_layer_x is None:
            if max(pack.sum() for pack in packed) < capacity:
                packed_memories = packed # found
                break
        else:
            is_exceed = False
            idx = 0
            for pack in packed: 
                if pack.sum() + per_layer_x[idx] > capacity:
                    is_exceed = True
                    break
                idx += len(pack)
            if not is_exceed:
                packed_memories = packed # found 
                idx = 0
                for pack in packed_memories: # add x in results (don't add x during compare, because memories and packed_memories share the memory)
                    pack[0] += per_layer_x[idx]
                    idx += len(pack)
                break
        # continue packing       
        if verbose: print("\tbalanced %d packs exceed capacity; try more packs"%S)  
    
    # check results
    if packed_memories is None:
        return None
    else: # confirm conrrectness
        assert len(memories) == sum(len(pack) for pack in packed_memories)
        if per_layer_x is None:
            assert memories.sum() == sum(pack.sum() for pack in packed_memories)
        # if verbose: print("\tpacked_memories (MB): {}".format(
        #                    [pack/1024./1024. for pack in packed_memories]))  
        # convert to layer packs
        layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
        
        if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
        return layer_packs

# 对packs中的所有pack计算，该pack所有层的大小+该pack首层的输入大小。选出其中的最大值
# 若最大值 < GPU容量，返回该packs
# 若最大的pack > GPU容量，返回None，表示不能reuse packing
def reuse_memory_packing(packs, per_layer_memories, per_layer_x, capacity, 
                         verbose=False, tab="\t"):
    """ Reuse input packing (e.g. FWD reuses BWD's).
        If packs' memory is within capacity, then return it.
        Else, return None. """
    
    assert packs is not None
    if packs == []:
        return []
    assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    if per_layer_x is None:
        xmems = np.zeros(len(memories))
    else:
        xmems = np.array(per_layer_x)
        assert len(xmems) == len(memories)

    # print(packs)
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    # print(memories)
    # print(memories[[1,2,3]])
    # exit(0)
    
    # check if memory < capacity
    # 对packs中的所有pack计算，该pack所有层的大小+该pack首层的输入大小
    # 选出其中的最大值，若最大值 < GPU容量，返回该packs
    if max([ memories[p].sum() + xmems[p[0]] for p in packs ]) < capacity:
        if verbose: print("{}reuse packs: {}".format(tab,packs)) 
        return packs
    else:
        # if verbose: print("{}not reusable packs: {}".format(tab,packs)) 
        return None

# 按照每个pack执行时间尽可能相等的原则进行打包。此外，需要注意隐含的另一个原则，即打包数会尽可能的少。
# 在首个符合GPU容量的打包完成后，不会再进行后续的打包
# 即从将整个model切成两份开始，检测按时间均匀方法切割的layer pack有没有超过GPU的容量。没超过直接返回，超过了继续测试切成
# 3份，以此类推
def balanced_time_packing(per_layer_times, 
                          per_layer_memories, per_layer_x, capacity, 
                          verbose=False, title="balanced_time_packing", tab="\t"):
    """ 
    Argument: per_layer_times: Compute time of each layer (no Swap nor P2P)
    """
    
    assert isinstance(per_layer_times, list)
    assert isinstance(per_layer_memories, list)
    assert isinstance(per_layer_x, list)
    assert len(per_layer_times) == len(per_layer_memories) and \
           len(per_layer_memories) == len(per_layer_x)
    times = np.array(per_layer_times) # a numpy object with a new memory
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    
    # find num of packs (initial guess)
    # 直接使用模型总的空间占用除以GPU的容量，得到最少应该打包的数量
    num_packs = ceil(memories.sum()/float(capacity)) # 向上取整 
    
    # parition into num of packs (under capacity constraint)
    packed_memories = None
    print(f"\tnum_packs:{num_packs}, len(memories):{len(memories)}")
    #                2                           28
    # print(list(range(num_packs, len(memories)+1)))
    # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    # exit(0)
    # 至少要切分成2份，因此range从2开始。最多切layer的数量份
    # 即从将整个model切成两份开始，检测按时间均匀方法切割的每个layer pack有没有超过GPU的容量。没超过直接返回，超过了继续测试切成
    # 3份，以此类推
    for S in range(num_packs, len(memories)+1):
        # balance the time per pack
        # 将数组 times 分割成 S 个子数组（或称为“packs”），以使这些子数组的元素之和大致相等，即时间和大致相等
        # 返回一个list，里面装着拆分后的sub_list
        packed_times = _balanced_sum_split(times, S)
        if verbose: print("\tS: {}, packed_times: {}".format(S, packed_times))
        # check empty pack
        if sum([ int(len(pack)==0) for pack in packed_times ]) != 0:
            if verbose: print("\tbalanced %d packs has empty pack; try more packs"%S) 
            continue
        # check memory of each pack
        is_exceed = False
        idx = 0

        # 检查每一个pack是否超过了GPU容量
        for pack in packed_times: 
            # 若该pack中所有layer空间占用（参数+激活）的和 加上 该pack首层输入的大小 > GPU的容量，直接退出当前循环，执行S+1，
            # 即更多的打包
            if memories[idx: idx+len(pack)].sum() + per_layer_x[idx] > capacity:
                is_exceed = True
                break
            idx += len(pack)
        # 若所有pack都通过了检查，则将memories按照每个pack的长度进行切片，且每个切片的首个值都加上了该层的输入大小
        # 直接break掉，不进行后续的打包
        if not is_exceed:
            packed_memories = [] # found
            idx = 0
            for pack in packed_times: 
                # 从memories中截取出对应当前pack的部分
                pack_mem = memories[idx: idx+len(pack)]
                pack_mem[0] += per_layer_x[idx]
                packed_memories.append(pack_mem)
                idx += len(pack)
            break
        # continue packing       
        if verbose: print("\tbalanced %d packs exceed capacity; try more packs"%S)  
    
    # check results
    if packed_memories is None:
        return None
    else: # confirm conrrectness
        assert len(memories) == sum(len(pack) for pack in packed_memories)
        if verbose: 
            print("\tpacked_times (sec) sum: {}".format(
                    [pack.sum() for pack in packed_times] ))
            # print("\tpacked_memories (MB): {}".format(
            #             [pack/1024./1024. for pack in packed_memories]))                
        # convert to layer packs
        # 将传进来的list转化为两个list。
        # 一个layer_pack，即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号。
        # 一个per_pack_memories，每个元素表示对应位置的layer pack的空间占用（bytes）
        layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)

        print(f"layer_packs:{layer_packs}")
        
        if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
        
        # 返回layer_packs，该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
        return layer_packs
    
def balanced_time_packing_2(per_layer_times, 
                          per_layer_memories, per_layer_x, capacity, num_gpus,
                          verbose=False, title="balanced_time_packing", tab="\t"):
    """ 
    Argument: per_layer_times: Compute time of each layer (no Swap nor P2P)
    """
    
    assert isinstance(per_layer_times, list)
    assert isinstance(per_layer_memories, list)
    assert isinstance(per_layer_x, list)
    assert len(per_layer_times) == len(per_layer_memories) and \
           len(per_layer_memories) == len(per_layer_x)
    times = np.array(per_layer_times) # a numpy object with a new memory
    memories = np.array(per_layer_memories) # a numpy object with a new memory
    
    # find num of packs (initial guess)
    # 最少打包的数量直接设置为GPU数量
    num_packs = num_gpus
    
    # parition into num of packs (under capacity constraint)
    packed_memories = None
    print(f"\tnum_packs:{num_packs}, len(memories):{len(memories)}")
    #                2                           28
    # print(list(range(num_packs, len(memories)+1)))
    # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    # exit(0)
    # 至少要切分成2份，因此range从2开始。最多切layer的数量份
    # 即从将整个model切成两份开始，检测按时间均匀方法切割的每个layer pack有没有超过GPU的容量。没超过直接返回，超过了继续测试切成
    # 3份，以此类推
    for S in range(num_packs, len(memories)+1):
        # balance the time per pack
        # 将数组 times 分割成 S 个子数组（或称为“packs”），以使这些子数组的元素之和大致相等，即时间和大致相等
        # 返回一个list，里面装着拆分后的sub_list
        packed_times = _balanced_sum_split(times, S)
        if verbose: print("\tS: {}, packed_times: {}".format(S, packed_times))
        # check empty pack
        if sum([ int(len(pack)==0) for pack in packed_times ]) != 0:
            if verbose: print("\tbalanced %d packs has empty pack; try more packs"%S) 
            continue
        # check memory of each pack
        is_exceed = False
        idx = 0

        # 检查每一个pack是否超过了GPU容量
        for pack in packed_times: 
            # 若该pack中所有layer空间占用（参数+激活）的和 加上 该pack首层输入的大小 > GPU的容量，直接退出当前循环，执行S+1，
            # 即更多的打包
            if memories[idx: idx+len(pack)].sum() + per_layer_x[idx] > capacity:
                is_exceed = True
                break
            idx += len(pack)
        # 若所有pack都通过了检查，则将memories按照每个pack的长度进行切片，且每个切片的首个值都加上了该层的输入大小
        # 直接break掉，不进行后续的打包
        if not is_exceed:
            packed_memories = [] # found
            idx = 0
            for pack in packed_times: 
                # 从memories中截取出对应当前pack的部分
                pack_mem = memories[idx: idx+len(pack)]
                pack_mem[0] += per_layer_x[idx]
                packed_memories.append(pack_mem)
                idx += len(pack)
            break
        # continue packing       
        if verbose: print("\tbalanced %d packs exceed capacity; try more packs"%S)  
    
    # check results
    if packed_memories is None:
        return None
    else: # confirm conrrectness
        assert len(memories) == sum(len(pack) for pack in packed_memories)
        if verbose: 
            print("\tpacked_times (sec) sum: {}".format(
                    [pack.sum() for pack in packed_times] ))
            # print("\tpacked_memories (MB): {}".format(
            #             [pack/1024./1024. for pack in packed_memories]))                
        # convert to layer packs
        # 将传进来的list转化为两个list。
        # 一个layer_pack，即该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号。
        # 一个per_pack_memories，每个元素表示对应位置的layer pack的空间占用（bytes）
        layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)

        print(f"layer_packs:{layer_packs}")
        
        if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title, tab=tab)
        
        # 返回layer_packs，该list中的每一个元素都是一个list，表示一个layer pack。layer_pack中的元素为layer的序号
        return layer_packs

# 将分割点转化为打包方案：
# split [x,y,z] will be converted to [ [0,...,x], [x+1,...,y], [y+1,...,z] ] 
def convert_splits_to_packs(splits, inclusive=True, verbose=False):
    """ 
    Assumption: splits elements are always increasing

    If input "splits" is inclusive:
        split [x,y,z] will be converted to [ [0,...,x], [x+1,...,y], [y+1,...,z] ] 
    else input "splits" is exclusive:
        split [x,y,z] will be converted to [ [0,...,x-1], [x,...,y-1], [y,...,z-1] ] 
        
    Return: converted packs for vt.layers
    """
    packs = []
    
    if inclusive:
        prev_s = -1
        for s in splits:
            assert prev_s < s
            packs.append( list(range(prev_s+1,s+1)) )
            prev_s = s
    else:
        prev_s = 0
        for s in splits:
            assert prev_s < s
            packs.append( list(range(prev_s,s)) )
            prev_s = s
    if verbose: print("convert_splits_to_packs: {}".format(packs))
    
    return packs

# pack_size：一个pack有多少层
# 返回正向的打包方案和反向的打包方案
# 正向:[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20]]
# 反向：[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
def manual_pack(R, pack_size, reverse=True, verbose=False):
    """ a.k.a constant size packing.
        pack 'R' layers by constant 'pack_size'
        
        when R % pack_size !=0, 
            if reverse: first pack is less than 'pack_size'
            otherwise:  last pack is less than 'pack_size'  """
    assert R != -1 and pack_size != -1
    # reverse即第一个pack的层数可能会小于pack_size
    if reverse:
        # R：28, pack_size:7 -> [6, 13, 20, 27]
        splits = list(range(R-1, -1, -pack_size))[::-1]
    else:
        splits = list(range(pack_size-1, R, pack_size))
        if splits[-1] != R-1:
            splits.append(R-1)
    
    # 将分割点转化为打包方案：
    # split [x,y,z] will be converted to [ [0,...,x], [x+1,...,y], [y+1,...,z] ] 
    pack_bwd = convert_splits_to_packs(splits, verbose=verbose)
    pack_fwd = convert_splits_to_packs(splits[:-1], verbose=verbose)# splits[:-1]，选择除最后一个元素外的所有元素

    return pack_fwd, pack_bwd

# # # Old
# def constant_memory_packing(per_layer_memories, capacity, verbose=False, title="constant_memory_packing"):
#     """ Assume uniform layers, i.e., each layer has the same 'constant' memory """
#     """ Deprecated: 1) no capacity constraint 2) optimal might be not constant """
#     assert isinstance(per_layer_memories, list) # [ 100, 50, 110, 100, 50, 110 ] # bytes
#     memories = np.array(per_layer_memories)
#     R = len(memories)
#     avg_memory_per_layer = memories.sum() / R
#     constant_packsize = int(floor(capacity/avg_memory_per_layer))
#     if verbose: print("\tconstant_packsize = %d"%constant_packsize)
#     pack_sizes = [ R%constant_packsize ] + \
#                  [ constant_packsize ] * int(R/constant_packsize)
#     if pack_sizes[0] == 0:
#         pack_sizes.pop(0)
#     assert sum(pack_sizes) == R
#     # packed_memories: [array([100]), array([100, 50, 110]), array([100, 50, 110])]
#     packed_memories = []
#     cnt = 0
#     for num_layers in pack_sizes:
#         packed_memories.append( np.array(memories[cnt:cnt+num_layers]) )       
#         cnt += num_layers
#     # confirm correctness
#     assert len(memories) == sum(len(pack) for pack in packed_memories)
#     assert memories.sum() == sum(pack.sum() for pack in packed_memories)
#     if verbose: print("\tpacked_memories (MB): {}".format(
#                        [pack/1024./1024. for pack in packed_memories]))
#     # convert to layer packs
#     layer_packs, per_pack_memories = convert_to_layer_packs(packed_memories)
#     
#     if verbose: print_memory_packing(layer_packs, per_pack_memories, title=title)
#     return layer_packs
