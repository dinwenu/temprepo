# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import functools
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.autograd import Function

import graph

import inspect
# 很明显，输入参数是module
def description(obj):
    """ Get one-line description of an Torch Module object.
        Format: <ClassName>(Arg1=Val1, Arg2=Val2) 
        E.g.  : Linear(in_features=512, out_features=1000, bias=True)
    """
    assert isinstance(obj, torch.nn.Module)
    if obj.extra_repr(): # most torch Modules
        return obj.__repr__()
    else: # customized Modules
        # 如果 obj 是自定义的模块，则获取其 __init__ 方法的参数签名（signature）
        sig = inspect.signature(obj.__class__.__init__)
        # 从参数签名中获取所有参数的名称
        names = [param.name for param in sig.parameters.values()]
        if "self" in names:
            names.remove("self")
        values = []
        for n in names: # in strict definition order
            # 若当前参数名是'config'，直接将该参数名加入到values列表中
            if n in ['config']: # BERT, GPT2
                values.append(n)
            else:
                assert hasattr(obj, n)
                # 获取对象 obj 中名称为 n 的属性的值
                v = getattr(obj, n)
                # 若获取的值是一个元组，将该值加入到value类表中
                if isinstance(v, (int, float, bool, None)):
                    values.append(v)
                else:
                    print("[warning] untested argument type: {}.__init__({}={})".format(obj.__class__.__name__, n, v))
                    values.append(v)
        main_str = obj._get_name() + '('
        # make simple one-liner info as most builtin Modules will use
        # e.g.
        #       'in_features={}, out_features={}, bias={}'.format(
        #        self.in_features, self.out_features, self.bias is not None )
        main_str += ", ".join(["{}={}".format(n,v) for n,v in zip(names,values)])
        main_str += ')'
        return main_str
    # ref: 
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/module.html#Module
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/linear.html#Linear
    # - https://pytorch.org/docs/1.5.0/_modules/torch/nn/modules/conv.html#Conv2d
    # - https://docs.python.org/3.8/library/inspect.html#introspecting-callables-with-the-signature-object
    # - https://stackoverflow.com/questions/36849837/introspecting-arguments-from-the-constructor-function-init-in-python

object_id = 0

class TensorWrapper(object):
    def __init__(self, tensor, node_desc, graph_creator, activation_size=None):
        self.tensor = tensor
        global object_id
        self.object_id = object_id
        print(f"node id:{object_id}, node_desc:{node_desc}")
        object_id += 1
        self.node_desc = node_desc

        self._node = graph.Node("node%d" % object_id, node_desc=node_desc)
        self.graph_creator = graph_creator

    def size(self, dim=None):
        if dim is None:
            result = self.tensor.size()
            dim_str = ""
        else:
            result = self.tensor.size(dim)
            dim_str = "(%d)" % dim
        wrapped_result = TensorWrapper(result, "Size%s" % dim_str, self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def dim(self):
        return self.tensor.dim()

    def view(self, *wrapped_sizes):
        sizes = []
        in_edges = []
        for wrapped_size in wrapped_sizes:
            if isinstance(wrapped_size, TensorWrapper):
                sizes.append(wrapped_size.tensor)
                in_edges.append(wrapped_size)
            else:
                sizes.append(wrapped_size)
        result = self.tensor.view(*sizes)

        if len(sizes) == 1:
            wrapped_result = TensorWrapper(result, "View", self.graph_creator)
        else:
            wrapped_result = TensorWrapper(result,
                                           "View(%s)" % ", ".join([str(size) for size in sizes[1:]]),
                                           self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        for in_edge in in_edges:
            self.graph_creator.graph.add_edge(in_edge.node(), wrapped_result.node())
        return wrapped_result

    def __gt__(self, other):
        return self.tensor.__gt__(other)

    def __lt__(self, other):
        return self.tensor.__lt__(other)

    def __add__(self, other):
        
        if isinstance(other, TensorWrapper):
            result_tensor = self.tensor + other.tensor
        else:
            result_tensor = self.tensor + other
        wrapped_result = TensorWrapper(result_tensor, "Add", self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        if isinstance(other, TensorWrapper):
            self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __iadd__(self, other):
        wrapped_result = TensorWrapper(self.tensor, "Add(inplace)", self.graph_creator)
        self.tensor += other.tensor
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __mul__(self, other):
        result = self.tensor * other.tensor
        wrapped_result = TensorWrapper(result, "Mul", self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        self.graph_creator.graph.add_edge(other.node(), wrapped_result.node())
        return wrapped_result

    def __getitem__(self, key):
        """ NOTE: slice is underdevelopment 
        see: 
            https://www.geeksforgeeks.org/implementing-slicing-in-__getitem__/
            https://stackoverflow.com/questions/43627405/understanding-getitem-method
            https://stackoverflow.com/questions/2936863/implementing-slicing-in-getitem
        """
        result_tensor = self.tensor[key]
        wrapped_result = TensorWrapper(result_tensor, "__getitem__(%s)" % key, self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def transpose(self, *args):
        result_tensor = self.tensor.transpose(*args)
        args_str = ", ".join([str(arg) for arg in args])
        wrapped_result = TensorWrapper(result_tensor, "Transpose(%s)" % args_str,
                                       self.graph_creator)
        self.graph_creator.graph.add_edge(self._node, wrapped_result.node())
        return wrapped_result

    def unsqueeze(self, *args):
        return self.tensor.unsqueeze(*args)

    def node(self):
        return self._node


def cat(wrapped_tensors, dim):
    tensors = []
    all_unwrapped_tensors = True
    graph_creator = None
    for wrapped_tensor in wrapped_tensors:
        if isinstance(wrapped_tensor, TensorWrapper):
            tensors.append(wrapped_tensor.tensor)
            graph_creator = wrapped_tensor.graph_creator
            all_unwrapped_tensors = False
        else:
            tensors.append(wrapped_tensor)
    # Simplifying assumption: if all tensors are "unwrapped", then we're not profiling,
    # and default to torch implementation.
    if all_unwrapped_tensors:
        return torch.cat(tensors, dim)
    result = torch.cat(tensors, dim)
    wrapped_result = TensorWrapper(result, "Concat(%d)" % dim, graph_creator)
    for wrapped_tensor in wrapped_tensors:
        if not isinstance(wrapped_tensor, TensorWrapper):
            wrapped_tensor = TensorWrapper(wrapped_tensor, "Input", graph_creator)
        graph_creator.graph.add_edge(wrapped_tensor.node(), wrapped_result.node())
    return wrapped_result


class GraphCreator(object):
    def __init__(self, model, module_whitelist=[], summary=[]):
        """
        Recursively create graph nodes from top to leaf module.
        Args:
            model: the top-level module
            module_whitelist: optional, contains module class names, if recursion hits this whitelist, just build the hit module as a graph node, without further recursion to leaf modules. Otherwise (miss this whitelist), recursion to leafs. 
            summary: optional, contains size and time of each module node
        """
        if isinstance(model, torch.nn.Module) is False:
            raise Exception("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.module_whitelist = module_whitelist
        self.summary = copy.deepcopy(summary)
        self.forward_original_methods = {}
        self.graph = graph.Graph()
        self.inputs = {} # {原始tensor：包装后的tensor}

    # 将给定module的每一层，若其没有子层或其名字在白名单中（就是各种GPT2layer），替换其前向传播方法
    # 这样在执行前向传播期间，会自动的建立每层对应的node，并在node之间建立边的关系（通过邻接表建立）
    def hook_modules(self, module, root=False):
        this_creator = self
        # __dict__ 属性是一个字典，包含了一个对象的所有属性和方法
        sub_modules = module.__dict__['_modules']

        # Wrapper function to "forward()", keeping track of dependencies.

        # 备注：“*”表示参数可以接受任意数量的位置参数，并且在函数内部可以将它们当作一个元组进行处理
        #
        # forward()的包装函数，功能为追踪依赖，即执行原本前向计算功能的同时，通过TensorWrapper中保存的node构建层(node)之间
        # 的依赖关系。（每一层的输出都会被包裹成TensorWrapper，该类中的一个成员变量就是node，即产生该输出的层）
        # 两种情况
        # 1.模型刚开始执行时，输入当前forward的数据是tensor，在对模型的输入进行 TensorWrapper 的实例化的过程中，创建对应
        #   输入的node，即“输入层”。而后得到当前层的输出，并对输出进行TensorWrapper的实例化，这个过程中建立了对应执行forward
        #   的module的node。而后建立输入node和当前node之间的连接。返回的 TensorWrapper 实例将直接用于后续module执行
        #   forward_wrapper函数时node之间的连接
        # 2.若输入的tensor已经是包裹后的tesnor，TensorWrapper，说明输入当前层的数据是上一层的输出，其中的node成员变量
        #   必然保存了输出该输出的node。直接执行当前module的forward函数得到输出，建立当前module的node，并在两个module之间
        #   建立连接
        #
        # 📌分析：刚看会有点绕，因为会把TensorWrapper实例化的过程与tensor联系在一起，感觉就是为了存放tensor。其实实例化的目的
        # 是为了创建node，站在node的角度就好理解了。绕的原因在于，并不是非常明显的直接对当前Module创建node，而是在当前Module输出
        # 一个结果后，以该结果做为参数之一进行实例化，这样看起来实例化的过程貌似只是跟tensor相关的，建立node的过程很隐晦。
        def forward_wrapper(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            # 将wrapped_inputs_list中的tesnor全部替换为包裹后的tensor，执行完毕后，input这个列表包含所有原始的tesnor
            #
            # 分析：经过module forward执行后的结果一定是TensorWrapper类型的，因此在for循环中只会执行if语句。换句话说，只有模型
            # 刚开始执行时会进入else，显式的传入node描述，即在实例化TensorWrapper的过程中创建“输入层”
            for i in range(len(wrapped_inputs_list)):
                # 2.若输入的tensor已经是包裹后的tesnor，TensorWrapper，说明输入当前层的数据是上一层的输出，其中的node成员变量
                #   必然保存了输出该输出的node，因此无需进入else语句通过实例化TensorWrapper来创建node。
                #   故可直接将该类中保存的tesnor加入到input列表中，直接用于输入给当前层的forward函数
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    # 该key即 wrapped_inputs 中的一个tensor
                    key = wrapped_inputs_list[i]

                    # 感觉这就是个保险操作，暂时理解为废操作
                    # 若inputs这个字典中存在这个key，直接把对应的值拿出来原地替换掉wrapped_inputs_list在当前位置的tensor
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]

                    # 1.若不存在，说明输入的tensor是模型的输入，即训练数据。显然，该逻辑仅在数据刚进入模型时执行一次。
                    #   即在对模型的输入进行TensorWrapper的实例化的过程中，创建对应该输入的“输入层（输入node）”
                    else:
                        # 模型刚开始执行，j=0. 所以这个描述（第2个参数）是 Input0。可见在对模型的输入进行TensorWrapper
                        # 的实例化的过程中，创建的对应输入的node为“输入层”
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        # 把 {原始tensor：包装后的tensor} 加入到 inputs 这个字典中
                        this_creator.inputs[key] = wrapped_inputs_list[i]

                    # input中装的是原始的tensor
                    input.append(wrapped_inputs_list[i].tensor)

            # 备注：* 操作符，表示将 input 中的元素解包并作为参数传递给函数调用
            # forward_original_methods[self] 就是得到一个forward方法，这个self就是sub_module，并非hook_modules的self
            # 使用调用该方法的sub_module原本的forward方法处理输入的tensor
            result = this_creator.forward_original_methods[self](*input)
            
            # 对上面输出的结果，即当前module forward后的结果进行包裹，即创建对应当前module的node。
            # 注意，第二个参数的self指的是module，也就是说，对该输出的描述是针对这一层的描述
            #
            # 📌分析：刚看可能觉得是对层的输出的描述，这是不对的，该描述用来创建一个node，因此当前函数建立的节点就是调
            # 用该forward_wrapper函数的层本身。只是在计算该层输出的同时建立了该层的节点，这个描述和该层的输出之间并无关系，
            # 就是用来描述当前层的
            wrapped_result = TensorWrapper(result, description(self), this_creator) 
            
            # 对每一个输入（TensorWrapper）：
            for wrapped_input in wrapped_inputs_list:
                # 使用邻接表建立节点之间的指向关系
                # def add_edge(self, node1, node2): # node1 -> node2
                # 📌可见，只是通过输入来获取输出该输入的node（层），建立的还是层之间的关系，不是建立输入输出之间的关系
                # 📌此外，尽管在实例化TensorWrapper的过程中创建了node，但在执行add_edge函数时，node才被加入到graph中
                this_creator.graph.add_edge(wrapped_input.node(), wrapped_result.node())

            return wrapped_result

        # Wrapper function to "forward()", keeping track of dependencies.
        # (without creating self node)
        def forward_wrapper_root(self, *wrapped_inputs):
            input = []
            wrapped_inputs_list = list(wrapped_inputs)
            for i in range(len(wrapped_inputs_list)):
                if isinstance(wrapped_inputs_list[i], TensorWrapper):
                    input.append(wrapped_inputs_list[i].tensor)
                else:
                    key = wrapped_inputs_list[i]
                    if key in this_creator.inputs:
                        wrapped_inputs_list[i] = this_creator.inputs[key]
                    else:
                        j = len(this_creator.inputs)
                        wrapped_inputs_list[i] = TensorWrapper(wrapped_inputs_list[i],
                                                               "Input%d" % j, this_creator)
                        this_creator.inputs[key] = wrapped_inputs_list[i]
                    input.append(wrapped_inputs_list[i].tensor)
            result = this_creator.forward_original_methods[self](*input)

            return result

        # name 是模块名称，sub_module 是对应的模块对象
        for name, sub_module in sub_modules.items():
            # nn.Module is the only thing we care about.
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_module_name = sub_module.__class__.__name__
            print("sub_module_name: ",sub_module_name)
            sub_sub_modules = sub_module.__dict__['_modules']
            print("len of sub_sub_modules", len(sub_sub_modules))
            print("------")
            # 如果当前子模块没有子模块，或者当前子模块在白名单中，则执行以下操作
            if len(sub_sub_modules) == 0 or sub_module_name in self.module_whitelist:
                """
                In 'pre_hook.patch' for the 'torchprofiler'
                +    def reset_hooks(self):
                +        self._backward_hooks = OrderedDict()
                +        self._backward_pre_hooks = OrderedDict() # profiler specific
                +        self._forward_hooks = OrderedDict()
                +        self._backward_hooks = OrderedDict()
                """
                """
                To be patch free, we manually reset hooks:
                    sub_module._backward_hooks = OrderedDict()
                    sub_module._forward_hooks = OrderedDict()
                [optional] sub_module._forward_pre_hooks = OrderedDict()
                """
                # sub_module.reset_hooks()
                if hasattr(sub_module, 'reset_hooks'): # patched for 'torchprofiler'
                    sub_module.reset_hooks()
                else: # patch free
                    from collections import OrderedDict
                    sub_module._backward_hooks = OrderedDict()
                    sub_module._forward_hooks = OrderedDict()    
                #
                # Hook leaf nn.Module with no descendants, or just in whitelist.
                #

                # Replace "forward" with "wrapped_forward".
                # 若 forward_original_methods 这个字典中不存在 sub_module 这个key，说明该层还没处理过，进行以下逻辑：
                if sub_module not in this_creator.forward_original_methods:
                    # 将当前子模块及其 forward 方法添加到 forward_original_methods 这个字典中，即保存好原始的forward方法
                    this_creator.forward_original_methods.update({sub_module:
                                                                   sub_module.forward})
                    # __get__ ：一个描述符方法，用于绑定方法或属性到一个对象上
                    # 将当前子模块的 forward 方法替换为 forward_wrapper 方法
                    sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)

            # 若当前层有子层且当前子层的名字不在白名单中,先递归的为子层执行上面的逻辑
            if len(sub_sub_modules) > 0 and sub_module_name not in self.module_whitelist:
                #
                # Recursively visit this module's descendants, if not in whitelist
                #
                self.hook_modules(sub_module)
        
        if root:
            this_creator.forward_original_methods.update({module: module.forward})
            module.forward = forward_wrapper_root.__get__(module, module.__class__)

    def unhook_modules(self):
        for sub_module in self.forward_original_methods:
            sub_module.forward = self.forward_original_methods[sub_module]

    def persist_graph(self, directory): 
        # 1.渲染一个graphviz图
        # 2.将当前保存了node和node入边出边新的graph实例字符串化并写入到txt文件
        graph.save_graph(self.graph, directory, graph.GRAPH_FILENAME)
