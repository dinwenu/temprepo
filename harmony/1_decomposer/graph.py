# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graphviz
import os
from collections import OrderedDict
from textwrap import shorten
import sys; sys.setrecursionlimit(5000)
# print("recursion limit={}".format(sys.getrecursionlimit()))

class Graph(object):
    def __init__(self, node=None):
        self.nodes = {} # { "node_id": node obj }
        if node is not None:
            self.nodes[node.node_id] = node
        self.edges = {} # { "node_id": [fan-out nodes] }
        self.in_edges = {} # { "node_id": [fan-in nodes] }

        self._predecessors = {}
        self._successors = {}
        self._augmented_antichains = {}
        self._deaugmented_augmented_antichains = {}
        self._next_antichains = {}
        self._antichain_dag = None

        self._colors = ['lightblue', 'green', 'grey', 'firebrick1',
                        'gold', 'chocolate1', 'beige']

        if node is not None:
            self.in_edges[node.node_id] = list()

    def copy(self):
        gr = Graph()
        for node_id in self.in_edges:
            for node2 in self.in_edges[node_id]:
                gr.add_edge(node2, self.nodes[node_id])
        # confirm fan-in ordering is kept
        for node_id, in_nodes in gr.in_edges.items(): # { "node_id": [fan-in nodes] }
            assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]]
        
        return gr

    # 找到graph的输入层(node)放在列表里返回
    def sources(self):
        sources = []
        for node_id in self.nodes:
            # 若该节点不存在扇入节点，或其扇入列表的长度为0，将该node加入到sources列表中
            if node_id not in self.in_edges or len(self.in_edges[node_id]) == 0:
                sources.append(self.nodes[node_id])
        return sources

    def add_node(self, node):
        self.nodes[node.node_id] = node

    # 将给定node、该node的所有边及其所有映射关系从整个graph中移除
    def remove_node(self, node):
        # 从图中删除该节点 id 到其本身的映射
        del self.nodes[node.node_id]
        # 若当前要删除的node存在扇出，则要把当前node从那些节点的扇入映射关系中移除
        if node.node_id in self.edges:
            # 暂存当前要删除的node指向的那些node
            out_nodes = self.edges[node.node_id]
            # 删除当前要删除的node在扇出字典中的映射关系
            del self.edges[node.node_id]
            # 移除这些扇入node关于当前node的扇入信息（从list中移除要删除的node）
            for out_node in out_nodes:
                self.in_edges[out_node.node_id].remove(node) # NOTE: can change the fan-in order
        # 若当前要删除的node存在扇入，则要把当前node从那些节点的扇出映射关系中移除
        if node.node_id in self.in_edges:
            # 暂存那些指向当前要删除的node的节点
            in_nodes = self.in_edges[node.node_id]
            # 删除当前要删除的node在扇入字典中的映射关系
            del self.in_edges[node.node_id]
            # 移除这些扇出node关于当前node的扇出信息（从list中移除要删除的node）
            for in_node in in_nodes:
                self.edges[in_node.node_id].remove(node) # NOTE: can change the fan-out order

    # 返回所有没有出度的node
    def sinks(self):
        sinks = []
        for node_id in self.nodes:
            if node_id not in self.edges or len(self.edges[node_id]) == 0:
                sinks.append(self.nodes[node_id])
        return sinks

    def reset(self):
        self._predecessors = {}
        self._successors = {}

    # 注意：若添加的node还没加入到self.nodes，将其添加进去
    def add_edge(self, node1, node2): # node1 -> node2
        if node1.node_id not in self.nodes:
            self.nodes[node1.node_id] = node1
        if node2.node_id not in self.nodes:
            self.nodes[node2.node_id] = node2

        if node2.node_id not in self.in_edges:
            self.in_edges[node2.node_id] = list()
        self.in_edges[node2.node_id].append(node1) # NOTE: always the last fan-in 
        if node1.node_id not in self.edges:
            self.edges[node1.node_id] = list()
        self.edges[node1.node_id].append(node2) # NOTE: always the last fan-out

    def remove_edge(self, node1, node2): # node1 -> node2
        self.edges[node1.node_id].remove(node2) # NOTE: can change the fan-out order
        self.in_edges[node2.node_id].remove(node1) # NOTE: can change the fan-in order
    
    # 删除旧node和目标node间边的关系，在新node目标node间建立边
    def replace_in_edge(self, old_src, dst, new_src):
        """ (old_src->dst) to (new_src->dst), where in_edge order of dst node is kept."""
        assert isinstance(old_src, Node) and isinstance(dst, Node) and isinstance(new_src, Node)
    
        self.edges[old_src.node_id].remove(dst)
        
        if new_src.node_id not in self.nodes:
            self.nodes[new_src.node_id] = new_src
        if new_src.node_id not in self.edges:
            self.edges[new_src.node_id] = list()
        self.edges[new_src.node_id].append(dst)
        
        # inplace replace in_edge of dst
        for i, in_node in enumerate(self.in_edges[dst.node_id]):
            if in_node.node_id == old_src.node_id:
                self.in_edges[dst.node_id][i] = new_src
    
    # 使用广度优先搜索（BFS）的方式，遍历并标记每个node的深度，即为其depth成员变量赋值（输入层深度为1）
    def populate_depths(self):
        # Helper method that annotates each node in the graph with its depth from the sink.
        sources = self.sources()
        # 将输入节点的深度标记为1
        sources[0].depth = 1
        queue = [sources[0]]
        while len(queue) > 0:
            # 从队列的末尾弹出一个元素
            node = queue.pop(-1)
            if node.node_id not in self.edges: continue
            # 对弹出的node的所有fan-out node，若其depth成员变量还没被赋值，或其depth < 当前node的深度+1，
            # 就更新fan-out node的深度为当前节点深度加 1，并将这些fan-out node加入队列中以便后续处理
            for out_node in self.edges[node.node_id]:
                if out_node.depth is None or out_node.depth < (node.depth + 1):
                    out_node.depth = node.depth + 1
                queue.append(out_node)

    # 1.遍历所有node，将所有 vlayer_id 放入集合中
    # 2.有序的为每一层建立一个子图，即为同层node建立子图，且同层之间若有指向关系，还需在两个node间建立边
    # 3.以列表的形式返回所有子图
    def partition_graph(self): # generate a list of isolated subgraphes(vlayers)
        vlayer_ids = set()
        # 遍历所有node，将所有 vlayer_id 放入集合中
        for node_id in self.nodes:
            vlayer_ids.add(self.nodes[node_id].vlayer_id)
        if len(vlayer_ids) == 1:
            return [self.copy()]
        subgraphs = []
        # 有序的为每一层建立一个子图，即为同层node建立子图，且同层之间若有指向关系，还需在两个node间建立边
        for vlayer_id in sorted(vlayer_ids):
            subgraphs.append(self.partition_graph_helper(vlayer_id))
        
        # confirm fan-in ordering is kept
        # for subgraph in subgraphs:
        #     for node_id, in_nodes in subgraph.in_edges.items(): # { "node_id": [fan-in nodes] }
        #         assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]], "[fan-in ordering not kept during partition_graph] {} v.s. {}".format([n.node_id for n in in_nodes], [n.node_id for n in self.in_edges[node_id]])
        
        # 以列表的形式返回所有子图
        return subgraphs

    # 将属于 vlayer_id 这一层的所有node加入到子图中，且同层之间若有指向关系，还需在两个node间建立边
    def partition_graph_helper(self, vlayer_id): # generate a copy of subgraph of vlayer_id; subgraphes are isolated from each other
        subgraph = Graph() # node and edge residing in this vlayer (excluding edging in/out from/to other vlayers)
        # traverse full-graph nodes to add my vlayer node

        # 遍历所有node，将所有在 vlayer_id 这一层的node加入到 subgraph 中
        for node_id in self.nodes:
            if self.nodes[node_id].vlayer_id == vlayer_id:
                subgraph.add_node(self.nodes[node_id]) 
        # traverse sub-graph nodes to add my vlayer edge
        # 遍历 subgraph 中的所有 node
        for node_id in subgraph.nodes:
            # 若当前node没有入边，继续
            if node_id not in self.in_edges: continue
            # 对当前node的所有入边node，若入边node和当前node在同一层，则在这两个node间建立边
            for in_node in self.in_edges[node_id]: # follow fan-in order
                if in_node.vlayer_id == vlayer_id:
                    subgraph.add_edge(in_node, self.nodes[node_id])
        return subgraph

    def chain_nodes(self):
        chain_nodes = list()
        for node in self.nodes.values():
            if node.node_id in self.edges and len(self.edges[node.node_id]) == 1 \
                and node.node_id in self.in_edges and len(self.in_edges[node.node_id]) == 1:
                chain_nodes.append(node)
        return chain_nodes

    def aggregate(self, sum_activations=False):
        forward_compute_time = 0.0
        backward_compute_time = 0.0
        parameter_size = 0.0
        activation_size = 0.0
        for node in self.nodes.values():
           forward_compute_time += node.forward_compute_time
           backward_compute_time += node.backward_compute_time
           parameter_size += node.parameter_size
           if sum_activations:
               activation_size += node.activation_size
           else:
               if node.node_id not in self.in_edges or len(self.in_edges[node.node_id]) == 0:
                   activation_size += node.activation_size
        return [forward_compute_time, backward_compute_time, parameter_size, activation_size]

    # 拓扑排序算法，用于对图数据结构中的节点进行拓扑排序，并返回排序后的节点列表
    def topological_sort(self):
        # Algorithm from https://en.wikipedia.org/wiki/Topological_sorting
        self.sorted_nodes = []
        self.marked_nodes = set()
        self.temporarily_marked_nodes = set()
        nodes = list(self.nodes.values())
        for node in nodes:
            print(node.node_id, node.node_desc)
        # 按照node的描述进行排序
        # ❓我不懂这里为什么要sort一下？
        nodes.sort(key=lambda x: x.node_desc)
        for node in nodes:
            print(node.node_id)
        for node in nodes:
            if node.node_id in self.marked_nodes:
                continue
            self.topological_sort_helper(node.node_id)
        return [self.nodes[node_id] for node_id in self.sorted_nodes]

    # 使用递归实现深度优先搜索进行拓扑排序
    def topological_sort_helper(self, node_id):
        if node_id in self.marked_nodes:
            return
        # 即出现走回到自身的情况
        if node_id in self.temporarily_marked_nodes:
            raise Exception("Graph has a cycle")
        self.temporarily_marked_nodes.add(node_id)
        # 若当前node指向其他node，获得其所有fan-out node并按desc（描述）排序，而后按照顺序递归的调用 topological_sort_helper
        if node_id in self.edges:
            out_nodes = list(self.edges[node_id])
            out_nodes.sort(key=lambda x: (x.node_desc, x.height))
            for out_node in out_nodes:
                self.topological_sort_helper(out_node.node_id)
        self.marked_nodes.add(node_id)
        self.temporarily_marked_nodes.remove(node_id)
        # 将当前节点插入到 sorted_nodes 列表的开头，也就是说，第一个遍历到的节点最后执行Insert操作，反而会位于链表的第一个
        self.sorted_nodes.insert(0, node_id)

    # 建给定node和其所有前驱节点间的映射关系，而后返回给定node的所有前驱节点
    def predecessors(self, node):
        if node in self._predecessors:
            return self._predecessors[node]
        predecessors = set()
        # 若node没有入度，直接返回一个空集合，即没有前驱节点
        if node not in self.in_edges:  # Source node
            return predecessors
        # 否则，将每个指向当前node的节点加入到集合中
        for in_node in self.in_edges[node]:
            predecessors.add(in_node)
            # 递归调用 predecessors 方法，获取入边节点 in_node 的前驱节点，并将其添加到前驱节点集合中
            predecessors.update(self.predecessors(in_node.node_id))
        # 建立当前节点和其所有前驱节点间的映射关系
        self._predecessors[node] = predecessors
        # 返回给定node的所有前驱节点
        return self._predecessors[node]

    # 找到给定节点集合中所有节点的前驱节点，并将它们存储在一个集合中返回
    def all_predecessors(self, antichain):
        all_predecessors = set()
        for antichain_node in antichain:
            # 获取当前antichain_node的所有前驱节点，并加入到集合中
            all_predecessors.update(self.predecessors(antichain_node))
            # 将节点 antichain_node 本身也添加到 all_predecessors 集合中
            all_predecessors.add(self.nodes[antichain_node])
        return all_predecessors

    # 返回node的所有后继节点
    def successors(self, node):
        if node in self._successors:
            return self._successors[node]
        successors = set()
        # 若该节点不存在出度，直接返回空集合
        if not node in self.edges:  # Sink node
            return successors
        # 对该节点指向的所有节点
        for out_node in self.edges[node]:
            # 将其添加到successors集合中
            successors.add(out_node)
            # 递归调用 successors 方法，获取出边节点 out_node 的后继节点，并将其添加到后继节点集合中
            successors.update(self.successors(out_node.node_id))
        # 建立当前节点和所有后继节点之间的映射关系
        self._successors[node] = successors
        # 返回其所有后继节点
        return self._successors[node]

    # 📌分析：感觉也是个保险操作，遍历antichain中所有node的前驱节点来找不和给定antichain存在路径的node，将其加入到antichain中，
    # 但是现在就是从网络的第一层开始建立antichain，这个函数目前没理解到存在的必要
    def augment_antichain(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._augmented_antichains:
            return self._augmented_antichains[antichain_key]
        extra_nodes = set()
        all_predecessors = set()
        for antichain_node in antichain:
            # 建立当前antichain_node和其所有前驱节点间的映射关系，而后返回其所有前驱节点
            predecessors = self.predecessors(antichain_node)
            # 并集操作
            all_predecessors = all_predecessors.union(predecessors)
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            # 提取出和当前antichain_node不相连的所有节点（即其前驱节点指向的其他节点，非前驱节点），加入到 extra_nodes 集合中
            for predecessor in predecessors:
                # 对于当前前驱的每一个出边节点
                for out_node in self.edges[predecessor.node_id]:
                    # 该出边节点不是任当前反链节点的前驱，显然无法到达当前反链节点。且该出边节点也不是当前反链节点自身，
                    # 则该节点可加入反链点集
                    if out_node not in predecessors and out_node.node_id != antichain_node:
                        extra_nodes.add(predecessor.node_id)
        # 将 extra_nodes 集合中的节点和原始反链合并成一个新的列表
        self._augmented_antichains[antichain_key] = list(extra_nodes) + antichain
        # 返回增广后的反链
        return self._augmented_antichains[antichain_key]

    # 📌分析：看起来不像是还原，而像是一个保险操作，即若存在两个节点之间的路径，例如node1->node2，则从反链中剔除掉node2
    # 剔除掉给定反链中存在的异常node，即从该node可以走到反链中的某个节点，返回正确的结果（一个list）
    # 假设的例子：old_node和other_node是augmented_antichain中的两个node，彼此没有直接的路径，此时肯定需要把other_node
    # 也删除掉
    # old_node -> new_node
    #                ^
    # other_node-----|
    def deaugment_augmented_antichain(self, augmented_antichain):
        augmented_antichain_key = tuple(sorted(augmented_antichain))
        if augmented_antichain_key in self._deaugmented_augmented_antichains:
            return self._deaugmented_augmented_antichains[augmented_antichain_key]
        nodes_to_remove = set()
        all_successors = set()
        # 对augmented_antichain中的每一个node
        for augmented_antichain_node in augmented_antichain:
            # 返回其所有后继节点
            successors = self.successors(augmented_antichain_node)
            # 若augmented_antichain中的其他节点是augmented_antichain中当前node的后继节点，将该节点加入到 nodes_to_remove 集合中
            for augmented_antichain_node_prime in augmented_antichain:
                if self.nodes[augmented_antichain_node_prime] in successors:
                    nodes_to_remove.add(augmented_antichain_node)
        antichain = list()
        # 对增广反链中的每一个node
        for augmented_antichain_node in augmented_antichain:
            # 若当前node不在 nodes_to_remove 中，也未加入 antichain 列表中，则加入 antichain
            if (augmented_antichain_node not in nodes_to_remove and \
                augmented_antichain_node not in antichain):
                antichain.append(augmented_antichain_node)
        self._deaugmented_augmented_antichains[augmented_antichain_key] = antichain
        return self._deaugmented_augmented_antichains[augmented_antichain_key]

    # 通过新节点的后继节点是否存在反链中的node，判断给定的新节点是否能构成下一个反链。new_node即新的反链中的节点，该节点
    # 显然不能直接走到当前反链中的任何一个节点。
    def is_next_antichain(self, augmented_antichain, new_node):
        # 返回new_node的所有后继节点
        successors = self.successors(new_node)
        augmented_antichain_set = set(augmented_antichain)
        # 对返回的每一个后继节点，若存在某个后继节点包含在 augmented_antichain 中，证明new_node和反链中的节点存在前后关系，
        # 返回false，表示当前反链和新节点不能构成下一个反链
        # 📌分析：这里的目的应该不是找后继节点，因为反链中节点的后继节点，只要不和其他反链节点有直连联系，即可与其他反链节点
        # 构成新的antichain，因此这里找的这么深，应该不是为了找新的反链节点，而是避免new_node是反链节点的前驱节点。这里我暂时
        # 认为没有意义
        # 📌分析：下一个反链中的元素不能走向当前反链中的任何一个元素，这么理解该函数的逻辑就通了
        for successor in successors:
            if successor.node_id in augmented_antichain_set:
                return False
        # 如果所有后继节点都不在增广后的反链中，则返回 True，表示新节点可以构成下一个反链
        return True

    # 1.建立新的antichain，新的antichain即排除augmented_antichain中与new_node有直接前后关系的old_node，剩余与new_nod结合为新的antichain
    # 2.保险操作：剔除掉给定反链中存在的异常node，即从该node可以走到反链中的某个节点
    def construct_antichain(self, augmented_antichain, old_node, new_node):
        # 将 augmented_antichain 中的 old_node 替换为 new_node
        # 新的antichain即排除augmented_antichain中与new_node有直接前后关系的old_node，剩余与new_nod结合为新的antichain
        new_antichain = [x if x != old_node else new_node for x in augmented_antichain]
        # 剔除掉给定反链中存在的异常node，即从该node可以走到反链中的某个节点，返回正确的结果（一个list）
        #
        # 📌分析：从当前gpt2的例子来看，这个方法貌似没有存在的必要，上面建好的new_antichain不可能存在某个节点到另一个节点的
        # 直接路径，因为 1.所有的层都是单向且只有一条路径的，new_antichain中的node不可能走到结尾又从头开始走，即不可能存在某个node
        # 是另一个node的后继；2.上面的语句把old_node剔除掉了/
        # 猜测这个语句可能在存在多条分支的神经网络中会产生作用
        # 假设的例子：old_node和other_node是augmented_antichain中的两个node，彼此没有直接的路径，此时肯定需要把other_node
        # 也删除掉
        # old_node -> new_node
        #                ^
        # other_node-----|
        return self.deaugment_augmented_antichain(new_antichain)

    # 1.执行保险操作，通过遍历antichain中所有node的前驱节点来找不和给定antichain存在路径的node，将其加入到antichain中，
    # 2.返回基于当前antichain构建的所有new_antichain，即对当前antichain中所有的node执行一次以下逻辑：删除当前node，并添加当前
    #   node的后继节点，而后再删除掉当前antichain中其他能走向该node的node
    def next_antichains(self, antichain):
        antichain_key = tuple(sorted(antichain))
        if antichain_key in self._next_antichains:
            return self._next_antichains[antichain_key]

        next_antichains = []
        antichain_set = set(antichain)
        # 感觉是个保险操作，遍历antichain中所有node的前驱节点来找不和给定antichain存在路径的node，将其加入到antichain中，
        # 但是现在就是从网络的第一层开始建立antichain，这个函数目前没理解到存在的必要
        augmented_antichain = self.augment_antichain(antichain)
        # 将不包含augmented_antichain_node（antichain中当前node）的new_antichain（即next_antichain）加入到链表中保存
        for augmented_antichain_node in augmented_antichain:
            # 若augmented_antichain_node存在出度，取出它所有指向的节点
            next_nodes = self.edges[augmented_antichain_node] if augmented_antichain_node in self.edges else []
            # 对当前augmented_antichain_node的每一个后继节点
            for next_node in next_nodes:
                if next_node.node_id in antichain_set:
                    continue
                # 通过新节点的后继节点是否存在反链中的node，判断给定的新节点是否能构成下一个反链。new_node即新的反链中的节点，该节点
                # 显然不能直接走到当前反链中的任何一个节点。
                if self.is_next_antichain(augmented_antichain, next_node.node_id):
                    # 1.建立新的antichain，新的antichain即排除augmented_antichain中与new_node有直接前后关系的old_node，剩余与new_nod结合为新的antichain
                    # 2.保险操作：剔除掉给定反链中存在的异常node，即从该node可以走到反链中的某个节点（应该特指可以走到next_node）
                    next_antichain = self.construct_antichain(augmented_antichain,
                                                              augmented_antichain_node,
                                                              next_node.node_id)
                    next_antichains.append(next_antichain)
        # 建立当前antichain和所有new_antichain之间的映射，即保存操作
        self._next_antichains[antichain_key] = next_antichains
        # 返回基于当前antichain构建的所有new_antichain，即对当前antichain中所有的node执行一次以下逻辑：删除当前node，并添加当前
        # node的后继节点，而后再删除掉当前antichain中其他能走向该node的node
        return self._next_antichains[antichain_key]

    # 从网络的第一层开始不断建立反链(一个装着node_id的list)，新的反链是基于旧的反链建立的，并为每个反链实例化一个AntichainNode，
    # 同时建立旧AntichainNode到新AntichainNode之间的边，最终构成由反链节点构成的有向无环图
    # 就gpt2这个例子来说，每个AntichainNode就是一个单独的节点，与原来的图没有任何区别
    def antichain_dag(self):
        if self._antichain_dag is not None:
            return self._antichain_dag

        antichain_dag = Graph()
        antichain_id = 0
        # sources：找到graph的输入层(node)放在列表里返回
        # 由于在调用该函数前已经删除了输入层，这里返回的应该是网络的第一层，即['node2']
        antichain = [self.sources()[0].node_id]
        # print(antichain)
        # 此时antichain就是一个装着node_id的列表
        # 建立AntichainNode，其中一个成员是增广后的反链
        # AntichainNode就是一个反链id 加一个 该反链包含的节点(装在list中)
        source_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(antichain))# augment_antichain：保险操作，对给定的反链（antichain）进行增广操作，并返回增广后的反链，返回的结果也是一个装着node_id的list
        # 为antichain_dag这个图添加一个属性
        antichain_dag.source = source_node
        antichain_queue = [antichain]
        # 建立 原始反链 和其 增广反链 之间的映射关系
        antichain_mapping = {tuple(sorted(antichain)): source_node}

        # 宽度优先遍历antichain，根据遍历到的antichain建立新的antichain，即next_antichain。并为新的antichain实例化
        # AntichainNode，而后在当前AntichainNode和新的AntichainNode之间建立边。最后将新的antichain入队
        while len(antichain_queue) > 0:
            antichain = antichain_queue.pop(0)
            antichain_key = tuple(sorted(antichain))
            if antichain_key in self._next_antichains:
                continue
            # 1.执行保险操作，通过遍历antichain中所有node的前驱节点来找不和给定antichain存在路径的node，将其加入到antichain中，
            # 2.返回基于当前antichain构建的所有new_antichain，即对当前antichain中所有的node执行一次以下逻辑：删除当前node，并
            #   添加当前node的后继节点，而后再删除掉当前antichain中其他能走向该node的node
            next_antichains = self.next_antichains(antichain)
            # 检查next_antichain是否已经加入到 antichain_mapping 中，没加入则是新的反链，为其建立一个AntichainNode
            # 为其实例化一个AntichainNode，并建立AntichainNode之间的边
            for next_antichain in next_antichains:
                next_antichain_key = tuple(sorted(next_antichain))
                # 若当前遍历到的next_antichain还没有建立 AntichainNode ，则实例化
                if next_antichain_key not in antichain_mapping:
                    antichain_id += 1
                    next_antichain_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(next_antichain))
                    antichain_mapping[next_antichain_key] = next_antichain_node
                # 在两个 AntichainNode 之间建立边
                antichain_dag.add_edge(antichain_mapping[antichain_key],
                                       antichain_mapping[next_antichain_key])
                # 将下个 next_antichain 入队
                antichain_queue.append(next_antichain)

        self._antichain_dag = antichain_dag
        
        # confirm fan-in ordering is kept (maybe too strong here)
        # for node_id, in_nodes in antichain_dag.in_edges.items(): # { "node_id": [fan-in nodes] }
        #     assert [n.node_id for n in in_nodes] == [n.node_id for n in self.in_edges[node_id]] 
        
        return antichain_dag
    
    def __str__(self): # graph.txt
        strs = []
        # 输出所有点的信息
        for node in self.nodes.values():
            strs.append(str(node))
        # 输出所有边的信息
        for node in self.nodes.values():
            # 若节点不存在入边，略过（不存在入边没发建立指向关系）
            if node.node_id not in self.in_edges:
                continue
            # 从第一个存在入边的node，开始建立node之间的指向关系
            for in_node in self.in_edges[node.node_id]: # fan-in order kept
                strs.append("\t%s -- %s" % (in_node.node_id, node.node_id))
        return "\n".join(strs)

    # 根据保存的txt文件，恢复graph
    # 1.将输入按行分割
    # 2.遍历每一行图形字符串，根据行的格式，将节点添加到图形对象的节点字典中，或者添加边到图形对象的边字典中
    #   2.1.如果行不以制表符 \t 开头，则将行解析为节点，并将节点添加到图形对象（gr）的节点字典中
    #   2.2.如果行以制表符 \t 开头，则将行解析为边，并将边添加到图形对象的边字典和入边字典中
    @staticmethod
    def from_str(graph_str): # graph.txt
        gr = Graph()
        # strip() 方法去除字符串开始和结尾的空白字符
        # 将输入按行分割
        graph_str_lines = graph_str.strip().split('\n')
        # print(type(graph_str_lines), graph_str_lines)
        # exit(0)
        # 遍历每一行图形字符串，根据行的格式，将节点添加到图形对象的节点字典中，或者添加边到图形对象的边字典中
        for graph_str_line in graph_str_lines:
            # 如果行不以制表符 \t 开头，则将行解析为节点，并将节点添加到图形对象（gr）的节点字典中
            if not graph_str_line.startswith('\t'):
                # 从字符串中提取出 node_id、node_desc、node_metadata，进一步的从node_metadata中提取数据并转为float类型
                # 最后利用提取的信息新建一个Node并返回
                node = Node.from_str(graph_str_line.strip())
                # 将该节点加入到gr中
                gr.nodes[node.node_id] = node
            # 如果行以制表符 \t 开头，则将行解析为边，并将边添加到图形对象的边字典和入边字典中（\t会被解释为一个制表符，占用一个固定的空格宽度）
            else:
                [in_node_id, node_id] = graph_str_line.strip().split(" -- ")
                if node_id not in gr.in_edges:
                    gr.in_edges[node_id] = [gr.nodes[in_node_id]]
                else: # fan-in order kept
                    gr.in_edges[node_id].append(gr.nodes[in_node_id])
                if in_node_id not in gr.edges:
                    gr.edges[in_node_id] = [gr.nodes[node_id]]
                else:
                    gr.edges[in_node_id].append(gr.nodes[node_id])
        return gr
    
    def to_dot(self, arch): # graph.dot.pdf
        dot = graphviz.Digraph()
        # 1.创建节点
        # 对于self.nodes中的每一个值，即node对象
        for node in self.nodes.values():
            # 若当前这个node存在扇入节点，则取出它所有扇入节点的node_id
            in_node_ids = ", ".join([in_node.node_id for in_node in self.in_edges[node.node_id]]) if node.node_id in self.in_edges else ""
            # in代表该node的所有扇入节点
            node_desc = "[in: %s]\n"%(in_node_ids)
            # node_desc += str(shorten("%s -- %s"%(node.node_id,node.node_desc), width=50, placeholder="..."))
            node_desc += "%s -- %s"%(node.node_id, node.node_desc)
            if node.vlayer_id is not None:
                node_desc += " (vLayer %d)" % node.vlayer_id
            # node_desc = shorten(node_desc, width=64, placeholder="...")
            if node.vlayer_id is not None:
                color = self._colors[node.vlayer_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        # 2.创建节点之间的边
        # 对于self.nodes中的每一个值，即node对象
        for node in self.nodes.values():
            # 若该node不存在扇入节点，继续循环
            if node.node_id not in self.in_edges:
                continue
            # 否则，对该节点的每一个扇入节点：
            for in_node in self.in_edges[node.node_id]: # fan-in order kept
                # 创建扇入节点到该node的边：start node -> end node
                dot.edge(in_node.node_id, node.node_id) # NOTE: can not show ordering
        # 3.保存并渲染图形
        print(arch)
        dot.render(arch)

    def to_dot_legacy(self, arch): # graph.dot.legacy.pdf
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            node_desc = "%s\n[forward_compute_time=%.3f,backward_compute_time=%.3f,activation_size=%s,parameter_size=%.1f]" % (
                node.node_desc, node.forward_compute_time, node.backward_compute_time,
                node.activation_size, node.parameter_size)
            if node.vlayer_id is not None:
                color = self._colors[node.vlayer_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.edges:
                continue
            for out_node in self.edges[node.node_id]:
                dot.edge(node.node_id, out_node.node_id)
        dot.render(arch)

    # 返回一个有序的字典，字典的键为vlayer_id，值为排好序的装着node_id的list，其中的顺序按新旧排序，旧节点排在前面，
    # 新节点排在后面。新旧各自按照节点序号从小到大排序
    def get_ordered_vlayer_node_ids(self):
        vlayer_node_ids = {} # { vlayer_id: [node_id, node_id, ..., new_node_id] }
        # 将vlayer_id属性相同的层放在一个list中，并建立vlayer_id和其对应list的映射
        for node in self.nodes.values():
            assert node.vlayer_id is not None, "graph node must be vlayered"
            if node.vlayer_id not in vlayer_node_ids:
                vlayer_node_ids[node.vlayer_id] = []
            vlayer_node_ids[node.vlayer_id].append(node.node_id)
        ordered_vlayer_node_ids = OrderedDict() # { ordered vlayer_id: ordered [node_id, ..., new_node_id] }
        # 将上面处理好的node_id重新排列，即新建立一个映射，将每个vlayer_list对应list中的node排序。其中的顺序按新旧排序，旧节点排在前面，
        # 新节点排在后面。新旧各自按照节点序号从小到大排序
        for vlayer_id in sorted(list(vlayer_node_ids.keys())):        
            # oem node_id
            nids = sorted([int(node_id.split("node")[-1]) for node_id in vlayer_node_ids[vlayer_id] if node_id.startswith("node")])
            # new_node_id
            new_nids = sorted([int(node_id.split("node")[-1]) for node_id in vlayer_node_ids[vlayer_id] if node_id.startswith("nw_node")])
            # recreate two
            ordered_vlayer_node_ids[vlayer_id] = \
            ["node%d"%nid for nid in nids] + ["nw_node%d"%nid for nid in new_nids]
        return ordered_vlayer_node_ids
    
    def print_ordered_vlayer_nodes(self):
        vlayer_node_ids = self.get_ordered_vlayer_node_ids()
        print("[vlayer_id : nodes] =")
        for vlayer_id, node_ids in vlayer_node_ids.items():
            print("{} : {}".format(vlayer_id,node_ids))
            for node_id in node_ids:
                print("\t{} -- {}".format(node_id, self.nodes[node_id].node_desc))
   
    # 逐层检查每一层的所有节点，检查其是否存在跨层的连接，即扇出的node不在自己的下一层。若出现了这种node，则从当前检查到的
    # 节点开始，建立其到最远fan-out node所在层的identity chain，而后将和该最远fan-out node同层的identity node，即nw_node，
    # 连接到该fan-out node，如此形成一个层层连续的链
    def sequentialize_graph(self, verbose=False):                
        # 返回一个有序的字典，字典的键为vlayer_id，值为排好序的装着node_id的list，其中的顺序按新旧排序，旧节点排在前面，
        # 新节点排在后面。新旧各自按照节点序号从小到大排序
        # 所谓的新节点，即nw_node，就是改方法内要创建的节点，即用于连接跨层的删除节点，使node与node之间是层层顺序连接的
        vlayer_node_ids = self.get_ordered_vlayer_node_ids() # { ordered vlayer_id: ordered [node_id, ...] }
        
        # find branch outs and seqentialize with Identiy nodes
        new_node_id = 1 # max([int(node_id.split("node")[-1]) for node_id in self.nodes.keys()]) + 1
        # 逐层检查每一层的所有节点，检查其是否存在跨层的连接，即扇出的node不在自己的下一层。若出现了这种node，则从当前检查到的
        # 节点开始，建立其到最远fan-out node所在层的identity chain，而后将和该最远fan-out node同层的identity node，即nw_node，
        # 连接到该fan-out node，如此形成一个层层连续的链
        for vlayer_id, node_ids in vlayer_node_ids.items():
            for node_id in node_ids:
                # 若当前node不是new_node（即identity node），且存在扇出，即指向其他节点
                if ("nw" not in node_id) and (node_id in self.edges): # not identity node && fan-out exists
                    # record current node's fan-outs
                    # 1.建立fan-out node的vlayer_id到其node_id的映射，即属于同一层级的fan-out node会被存到一个list中
                    #   该字典的键即为其指向的所有node的层号(vlayer_id)
                    out_vlayer_node_ids = {} # { out_vlayer_id: [out_node_ids] }
                    for out_node in self.edges[node_id]:
                        if out_node.vlayer_id not in out_vlayer_node_ids:
                            out_vlayer_node_ids[out_node.vlayer_id] = []
                        out_vlayer_node_ids[out_node.vlayer_id].append(out_node.node_id)
                    # leave only out vlayers
                    # 2.若当前遍历到 vlayer_id 已经存在于 out_vlayer_node_ids 中。说明当前node的出边连接到的node和自己在同一层
                    if vlayer_id in out_vlayer_node_ids:
                        # ❓删除当前节点连接的同一层的记录，因为在这里不允许节点直接连接到相同的层
                        del out_vlayer_node_ids[vlayer_id]
                    # 3.检查剩余的fan-out node所在的层，即out_vlayer_id，检查其是否大于当前遍历到的vlayer_id。若小于，
                    #   说明当前层指向的层是自己上面的层，这是不允许的，直接退出程序
                    for out_vlayer_id in out_vlayer_node_ids.keys():
                        assert out_vlayer_id > vlayer_id # no circular vlayers
                    
                    # distinct_vlayer_ids = set()
                    # for out_node in self.edges[node_id]:
                    #     distinct_vlayer_ids.add(out_node.vlayer_id)
                    # distinct_vlayer_ids.discard(node.node_id) # pure out vlayer_ids
                    # distinct_vlayer_ids = set(out_vlayer_node_ids.keys()).discard(node_id) # pure out vlayer_ids
                    # confirm no circular fan-out
                    # for distinct_vlayer_id in distinct_vlayer_ids:
                    #     assert distinct_vlayer_id > vlayer_id:
                    
                    # check whether all fan-out are sequential
                    seq = True
                    # 4.遍历每一个出边node所在的层（即vlayer_id），若所在层的id不等于当前层的id+1，说明出现了跨层连接，即node之
                    #   间的连接不是层层顺序连接的
                    #   直接退出当前循环，执行后续的if else语句
                    for out_vlayer_id in out_vlayer_node_ids.keys():
                        if out_vlayer_id != vlayer_id + 1:
                            seq = False
                            break
                    # 若当前node与其出边node是顺序的，即没跨层，continue
                    if seq:
                        continue # next node
                    else: # non-sequential fan-out exists                 
                        if verbose:
                            print("non-sequential fan-out on {}. seqentializing.".format(node_id))
                        
                        # create an Identity chain from current node to the farest vlayer
                        vlayer_new_node = {} # {vlayer: new node}
                        prev_node = self.nodes[node_id]
                        # 从当前层的下一层到当前node连接到的最远的那一层，在每一层建立一个identity node，并从当前node开始，在所有identity
                        # node之间顺序的建立边
                        for identity_vlayer_id in range(vlayer_id+1, max(out_vlayer_node_ids.keys())+1):
                            new_node = Node("nw_node%d" % new_node_id,
                                            node_desc="Identity",
                                            vlayer_id=identity_vlayer_id)
                            new_node_id += 1
                            self.add_edge(prev_node, new_node)
                            prev_node = new_node
                            vlayer_new_node[identity_vlayer_id] = new_node 
                        
                        # replace edges (current node -> out node) to (Identity node -> out node)
                        # out_vlayer_id 即该fan-out node所在的层，
                        # ❓为什么使用和删除节点相同层的identity node指向删除节点，不应该用上一层的吗？
                        # 目前来看，貌似就是这样的，但感觉这样凭空多了一层出来啊？
                        for out_vlayer_id, out_node_ids in out_vlayer_node_ids.items():
                            for out_node_id in out_node_ids:
                                # 删除旧node（参数1）和目标node（参数2）间边的关系，在新node（参数3）和目标node间建立边
                                self.replace_in_edge(self.nodes[node_id], self.nodes[out_node_id], vlayer_new_node[out_vlayer_id])
                        
                        # # remove edges of (current node -> out nodes) 
                        # for out_node_ids in out_vlayer_node_ids.values():
                        #     for out_node_id in out_node_ids:
                        #         self.remove_edge(self.nodes[node_id], self.nodes[out_node_id])
                        # # add edge for (Identity node -> out node) 
                        # for out_vlayer_id, out_node_ids in out_vlayer_node_ids.items():
                        #     for out_node_id in out_node_ids:
                        #         self.add_edge(vlayer_new_node[out_vlayer_id], self.nodes[out_node_id])
                     
    def track_source_of_new_node_chain(self, cur_node_id, verbose=False):
        """ 'source' node -> new node: Identity -> new node: Identity -> ... -> curent new node Identity """
        if cur_node_id.startswith("node"):
            return cur_node_id
        else: # current is new node
            node_id = cur_node_id
            while True: # loop through new nodes
                if verbose: print("[track source] node_id = {}".format(node_id))
                if node_id.startswith("node"):
                    break
                assert self.nodes[node_id].node_desc.startswith("Identity")
                assert len(self.in_edges[node_id])==1
                node_id = self.in_edges[node_id][0].node_id
            return node_id

class Node(object):
    def __init__(self, node_id, node_desc="", forward_compute_time=0.0,
                 backward_compute_time=0.0, activation_size=0.0, parameter_size=0.0,
                 vlayer_id=None): # TODO: remove forward_compute_time/backward_compute_time/activation_size/parameter_size
        self.node_id = node_id
        self.node_desc = node_desc
        self.forward_compute_time = forward_compute_time
        self.backward_compute_time = backward_compute_time
        self.activation_size = activation_size
        self.parameter_size = parameter_size
        self.vlayer_id = vlayer_id
        self.depth = None
        self.height = None

    def set_vlayer_id(self, vlayer_id):
        self.vlayer_id = vlayer_id

    def __str__(self):
        vlayer_id_str = " -- vlayer_id=%d" % self.vlayer_id if self.vlayer_id is not None else ""
        # 将节点的 node_desc 属性中的换行符替换为空字符串
        node_desc = self.node_desc.replace('\n', "")
        # 将节点的 activation_size 属性转换为字符串
        activation_size = ("%s" % self.activation_size).replace(", ", "; ")
        return "%s -- %s -- forward_compute_time=%.3f, backward_compute_time=%.3f, activation_size=%s, parameter_size=%.3f%s" % (
            self.node_id, node_desc, self.forward_compute_time, self.backward_compute_time,
            activation_size, self.parameter_size, vlayer_id_str)

    # 从字符串中提取出 node_id、node_desc、node_metadata，进一步的从node_metadata中提取数据并转为float类型
    # 最后利用提取的信息新建一个Node并返回
    @staticmethod
    def from_str(node_str):
        node_str_tokens = node_str.strip().split(" -- ")
        node_id = node_str_tokens[0]
        node_desc = node_str_tokens[1]
        node_metadata = node_str_tokens[2]
        vlayer_id = None
        if len(node_str_tokens) > 3:
            vlayer_id = int(node_str_tokens[3].split("=")[1])
        [forward_compute_time, backward_compute_time, activation_size, parameter_size] = node_metadata.split(", ")
        forward_compute_time = float(forward_compute_time.split("=")[1])
        backward_compute_time = float(backward_compute_time.split("=")[1])
        if "[" in activation_size:
            activation_size = activation_size.split("=")[1]
            activation_size = sum([float(x) for x in activation_size.lstrip("[").rstrip("]").split("; ")])
        else:
            activation_size = float(activation_size.split("=")[1])
        parameter_size = float(parameter_size.split("=")[1])
        return Node(node_id, node_desc, forward_compute_time=forward_compute_time,
                    backward_compute_time=backward_compute_time, activation_size=activation_size,
                    parameter_size=parameter_size, vlayer_id=vlayer_id)

class AntichainNode(Node):
    def __init__(self, node_id, antichain, node_desc=""):
        self.antichain = antichain
        self.output_activation_size = 0.0
        super(AntichainNode, self).__init__(node_id, node_desc)

    def __str__(self):
        return "%s -- %s" % (self.node_id, self.antichain)


GRAPH_FILENAME = "graph"
PAR_FILENAME = "par_graph"
SEQ_FILENAME = "seq_graph"

# 1.渲染一个graphviz图
# 2.将当前保存了node和node入边出边新的graph实例字符串化并写入到txt文件
def save_graph(graph, path_dir, fname="graph", base_dir="graph", verbose=True): 
    assert isinstance(graph, Graph)
    assert os.path.exists(path_dir)
    # 创建完整的保存地址
    fname = fname.split(".")[0] # strip off extension
    if base_dir is None:
        full_dir = path_dir
    else:
        full_dir = os.path.join(path_dir, base_dir)
        os.makedirs(full_dir, exist_ok=True)

    # graph.to_dot_legacy(os.path.join(full_dir, fname + ".dot.legacy")) 
    # 1.创建节点
    # 2.创建边
    # 3.保存并渲染
    graph.to_dot(os.path.join(full_dir, fname + ".dot"))

    # 将当前保存了node和node入边出边新的graph实例字符串化并写入到txt文件
    with open(os.path.join(full_dir, fname + ".txt"), 'w') as f:
        f.write(str(graph))

    if verbose: print("graph saved to: {}/{}".format(full_dir, fname + ".txt"))

# 读取并返回上面那个方法保存的图(是一个txt文件)，根据保存的txt文件，恢复graph
def load_graph(path_dir, fname="graph.txt", base_dir="graph", verbose=True):
    if ".txt" not in fname:
        fname += ".txt"
    if base_dir is None:
        full_path = os.path.join(path_dir, fname)
    else:
        full_path = os.path.join(path_dir, base_dir, fname)
    assert os.path.exists(full_path)
    
    with open(full_path, 'r') as f:
        # 根据保存的txt文件，恢复graph
        # 1.将输入按行分割
        # 2.遍历每一行图形字符串，根据行的格式，将节点添加到图形对象的节点字典中，或者添加边到图形对象的边字典中
        #   2.1.如果行不以制表符 \t 开头，则将行解析为节点，并将节点添加到图形对象（gr）的节点字典中
        #   2.2.如果行以制表符 \t 开头，则将行解析为边，并将边添加到图形对象的边字典和入边字典中
        graph = Graph.from_str(f.read())# f.read():读取文件中的所有内容并返回一个包含文件内容的字符串
    if verbose: print("graph loaded from: {}".format(full_path))
    
    return graph
