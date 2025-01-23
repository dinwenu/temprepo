# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from collections import OrderedDict
import os
import graph

# 1.从graph中删除所有的Input层，并收集这些层的扇出节点（即指向的那些层）
# 2.从网络的第一层开始不断建立反链(一个装着node_id的list)，新的反链是基于旧的反链建立的，并为每个反链实例化一个AntichainNode，
#   同时建立旧AntichainNode到新AntichainNode之间的边，最终构成由反链节点构成的有向无环图
#   就gpt2这个例子来说，每个AntichainNode就是一个单独的节点，与原来的图没有任何区别。
#   📌可以说，antinode的存在就是为了步骤3。antinode中的antichain成员变量装着该反链中所有node的node_id，即互相之间没有
#   直接的路径。那完全可以这么理解，这个antichain就是同一层的node，这样步骤3中这些Node会被标记为同一层
# 3.为每个AntiChainNodes的vlayer_id属性赋值，即标记节点的深度
# 4.重新建立输入层和其指向的node之间的边，并将输入层的vlauer_id属性赋值为0
def partition(input_dir, output_dir=None, verbose=False):
    # 根据保存的txt文件，恢复graph类的实例
    gr = graph.load_graph(input_dir, graph.GRAPH_FILENAME)
    if verbose: 
        print("--- start partition ---")

    # Remove inputs in graph, since inputs should always be in the first vlayer. 
    # (We will add the inputs back after partition)
    # NOTE: assume sources are the last fan-in for their consumer nodes
    # 找到graph的输入层(node)放在列表里返回
    sources = gr.sources()
    # 用来装要移除的输出层的顺序字典
    nodes_to_remove = OrderedDict()
    # 从graph中删除所有的Input层，并收集这些层的扇出节点（即指向的那些层）
    for source in sources:
        # 若该层是Input层
        if source.node_desc.startswith("Input"):
            nodes_to_remove[source] = []
            # 收集当前Input层指向的所有node，放入字典中
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)
            # 将输入层、该层的所有边及其所有映射关系从整个graph中移除
            gr.remove_node(source)
    if verbose: 
        print("sources to remove: {}".format([str(node) for node in nodes_to_remove.keys()]))

    # Remove all unneeded sinks that are not used, makes code generation easier.
    # 返回所有没有出度的node（汇点：sink node）
    sinks = gr.sinks()
    # 若汇点的描述以 __getitem__ 为开头，则从图中删除该汇点
    for sink in sinks:
        # ❓什么节点会以 __getitem__ 开头？
        if sink.node_desc.startswith("__getitem__"):
            gr.remove_node(sink)
            if verbose: 
                print("sink to remove: {}".format(sink))
    
    # Make DAG and sort it
    # Make DAG
    # 从网络的第一层开始不断建立反链(一个装着node_id的list)，新的反链是基于旧的反链建立的，并为每个反链实例化一个AntichainNode，
    # 同时建立旧AntichainNode到新AntichainNode之间的边，最终构成由反链节点构成的有向无环图
    # 就gpt2这个例子来说，每个AntichainNode就是一个单独的节点，与原来的图没有任何区别
    antichain_gr = gr.antichain_dag()
    if verbose: 
        print("Antichain Graph:\n{}".format(antichain_gr))
    # sort it
    # 拓扑排序算法，用于对图数据结构中的节点进行拓扑排序，并返回排序后的节点列表
    # 📌该函数内部首先使用node的node_desc进行排序，而antinode的node_desc是空的，因此顺序不变。
    #    后续的深度优先搜索依然从antichain_0开始
    # states即为装着antinode的list
    states = antichain_gr.topological_sort() # out-of-place
    if verbose: 
        print("\nstates (sorted AntichainNodes):")
        for node in states:
            print(str(node))
        print("\nTotal number of states (sorted AntichainNodes): %d" % len(states))
    
    # Treat each node (of original graph) as an vlayer
    # Follow sorting index (of AntiChainNodes) to assign node's vlayer id (vLayer id)
    # (Results might have multiple nodes in the same vlayer, but still works.)
    # (Results of vlayer ids follows topological ordering.)
    # 为每个AntiChainNodes的vlayer_id属性赋值，即标记节点的深度
    partial_splits = range(1, len(states)+1)
    if verbose:
        print("\npartial_splits = {}".format(partial_splits))
        # > partial_splits=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 52, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 73, 78, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    start_point = 0
    vlayer_id = 0
    for split in partial_splits:
        if verbose:
            print("\tcurrent split = [{},{})".format(start_point, split))
        
        # 找到给定节点集合中所有节点的前驱节点（包括给定节点），并将它们存储在一个集合中返回，集合中装的是node
        predecessors = gr.all_predecessors(states[split-1].antichain) # inclusive
        if verbose:
            print("\t{}'s predecessors = {}".format(
                    str(states[split-1]), 
                    [predecessor.node_id for predecessor in predecessors] ))
        
        set_vlayer = False
        for predecessor in predecessors:
            # 若当前遍历到的前驱节点的 vlayer_id 属性还未被设置，对其进行赋值。该if语句的意义在于，前驱节点一定会被不断
            # 遍历到，有了该if就能避免相同节点的重复赋值
            # 📌看这就能理解为什么 all_predecessors 函数返回的node对象包含自身，因为这里为vlayer_id赋值的逻辑就是这样的
            # 并没有为当前node的vlayer_id直接赋值这样明显的操作，而是隐藏在为前驱node赋值的过程中
            if predecessor.vlayer_id is None:
                predecessor.set_vlayer_id(vlayer_id)
                if verbose:
                    print("\t\t{} set_vlayer_id to {}".format(
                            predecessor.node_id, vlayer_id))
                set_vlayer = True
        
        start_point = split
        # 若存在被赋值的前驱节点，深度+1
        if set_vlayer: # make vlayer_id continous
            vlayer_id += 1
    if verbose:
        print("Total number of vlayers: %d" % vlayer_id)
    
    # Set inputs as first vlayer; Add back removed inputs to graph
    # 重新建立输入层和其指向的node之间的边，并将输入层的vlauer_id属性赋值为0
    for source in nodes_to_remove:
        for out_node in nodes_to_remove[source]:
            # ❓输入层不和第一层同层了吗？
            source.set_vlayer_id(0)
            gr.add_edge(source, out_node)

    # Write result graph
    if output_dir is not None:
        graph.save_graph(gr, output_dir, graph.PAR_FILENAME)
    print("--- graph partitioned ---")

    return str(gr)
    
def sequentialize(par_graph, output_dir=None, verbose=False):
    
    gr = graph.Graph.from_str(par_graph) # graph.load_graph(input_dir, graph.PAR_FILENAME)
    if verbose:
        print("--- start sequentialize ---")
    
    # NOTE: different topological ordering results in different sequential Identity chains, which results in different performances.
    # Future work can be done to improve this performance by sorting differently.
    # 逐层检查每一层的所有节点，检查其是否存在跨层的连接，即扇出的node不在自己的下一层。若出现了这种node，则从当前检查到的
    # 节点开始，建立其到最远fan-out node所在层的identity chain，而后将和该最远fan-out node同层的identity node，即nw_node，
    # 连接到该fan-out node，如此形成一个层层连续的链
    gr.sequentialize_graph(verbose)
    if verbose:
        gr.print_ordered_vlayer_nodes()
    # all vlayers are now sequential
        
    # Write back
    if output_dir is not None:
        graph.save_graph(gr, output_dir, graph.SEQ_FILENAME)
    print("--- graph sequentialized ---")

    return str(gr)


    
