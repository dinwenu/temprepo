# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
from collections import OrderedDict as ODict

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
# 一个线程安全的队列
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, tensor):
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.cv.release()
        return tensor
    
"""
Implementation of a thread-safe dictionary with one producer and one consumer.
"""
# 实现了一个线程安全的字典，其中包括一个生产者和一个消费者
class OrderedDictionary:
    def __init__(self):
        self.odict = ODict() # { layer_id: [u1's {"name1": tensor1, "name2": [tensor2]}, u2's {}, ... ] }
        # 声明了一个线程条件对象，其中包含了一个锁和一个等待队列
        self.cv = threading.Condition()

    def __repr__(self, title="thread safe dict"): #.format( "-" if src_rank is None else "(src_rank%d)"%(src_rank) )
        
        def show_named_tensors(named_tensors):
            sss = []
            for name, tensor in named_tensors.items():
                sss.append("{}:{}".format(name, type(tensor)))
            return "{ %s }" % (", ".join(sss))
        
        s = "----- %s -----\n"%(title)
        for layer_id, named_tensors_list in self.odict.items():
            ss = ", ".join([show_named_tensors(named_tensors) for named_tensors in named_tensors_list])
            s += "L{}:[{}]\n".format(layer_id, ss)
        s += "-------------------------------"
        return s

    # 1.初始化有序字典，即为传进来的 layer_ids 这个list中的 layer_id 执行：self.odict[id] = []
    # 2.初始化一个成员变量，layer_ids，是一个列表，包含了所有传进来的layer_id，且是有序的
    def init_layer_ids(self, layer_ids): # always ascending
        assert isinstance(layer_ids,list)
        for id in sorted(layer_ids): 
            self.odict[id] = []
        self.layer_ids = list(self.odict.keys())
    
    def add(self, layer_id, named_tensors):
        # 在多线程环境中，当一个线程调用 cv.acquire() 时，它会尝试获取条件变量的锁。
        # 如果条件变量的锁已经被其他线程获取了，那么该线程会被阻塞，直到条件变量的锁被释放
        self.cv.acquire()
        # if layer_id not in self.odict:
        #     self.odict[layer_id] = []
        self.odict[layer_id].append(named_tensors)
        # self.cv.notify() 被用来通知等待在条件变量 self.cv 上的某个线程，以便该线程可以继续执行
        # 需要注意的是，notify() 方法只会通知一个等待的线程，如果有多个线程等待在条件变量上，只有其中一个线程会被唤醒
        self.cv.notify()
        # self.cv.release() 被用来释放条件变量 self.cv 的锁
        self.cv.release()

    # 从 self.odict[layer_id] 列表中弹出并返回第一个元素
    def remove(self, layer_id):
        self.cv.acquire()
        # if layer_id not in self.odict:
        #     self.cv.release()
        #     return None
        while len(self.odict[layer_id]) == 0:
            # 使当前线程等待并释放锁。当其他线程调用 self.cv.notify() 或 self.cv.notify_all() 时，
            # 当前线程将被唤醒并重新尝试获取锁。
            self.cv.wait()
        named_tensors = self.odict[layer_id].pop(0)
        self.cv.release()
        return named_tensors

    
