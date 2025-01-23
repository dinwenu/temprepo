# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import json
from collections import OrderedDict as ODict

import torch

import sys
sys.path.append("../../../model_lib")
sys.path.append("../../../util_lib")

sys.path.append("../../2_profiler")

def add_args(parser):
    ### GPT2
    parser.add_argument("--gpt2_config_path", type=str, 
                        default="../../../model_lib/gpt2_configs/gpt2-xl-config.json", 
                        help="")
    return parser

# 根据input和label的形状与类型，分别为这俩创建一个值全为 1 的张量。返回这俩tensor的名字和创建的张量
def synthetic_data(args): # "gpt2_huggingface" and "gpt2_2bw"
    data_names = ["input0"]
    target_names = ["labels"]
    
    print(f"args.config.n_positions:{args.config.n_positions}")
    tensor_shapes = {"input0": [args.config.n_positions], "labels": [args.config.n_positions]}
    tensor_dtypes = {"input0": torch.int64, "labels": torch.int64}

    data_tensors = []
    for name in data_names:
        # 根据对应的形状和类型，创建一个值全为 1 的张量
        data_tensors.append(torch.ones(tuple(tensor_shapes[name]), dtype=tensor_dtypes[name]))
    
    target_tensors = []
    for name in target_names:
        # 根据对应的形状和类型，创建一个值全为 1 的张量
        target_tensors.append(torch.ones(tuple(tensor_shapes[name]), dtype=tensor_dtypes[name]))   

    return data_names, data_tensors, target_names, target_tensors

# 1.使用Importliob模块，把第1步拆分层的代码导入进来，赋值给module
# 2.从参数 args 中指定的配置文件中加载 GPT-2 模型的配置信息，创建一个 GPT2Config 对象并赋值给 config
# 3.创建了一个交叉熵损失函数 CrossEntropyLoss 的实例
# 4.实例化已被拆分的model
# 5.为args添加新的值，即模型的config实例
# 返回model
def create_model(args):
    sys.path.append(args.module_dir)
    # 使用Importliob模块，把第1步拆分层的代码导入进来，赋值给module
    print("args.module_name:", args.module_name)
    import importlib; module = importlib.import_module(args.module_name + ".code")

    from gpt2_huggingface import GPT2Config
    # 从参数 args 中指定的配置文件中加载 GPT-2 模型的配置信息，创建一个 GPT2Config 对象并赋值给 config
    config = GPT2Config.from_json_file(args.gpt2_config_path)
    # print("gpt2config({}): num_hidden_layers={}, hidden_size={}, num_attention_heads={}, max_seq_length={}".format(
    #     os.path.basename(args.gpt2_config_path), config.num_hidden_layers,config.hidden_size,config.num_attention_heads,config.n_positions))
    
    # 创建了一个交叉熵损失函数 CrossEntropyLoss 的实例
    criterion = torch.nn.CrossEntropyLoss()
    # 实例化已被拆分的模型
    model = module.model(config, criterion)
    # print("model type:", type(model)) # 一个list
    # print("直接输出model：", model)
    print("model的类型是：", type(model))
    # for element in model:
    #     print(f"元素 {element} 的类名是: {type(element).__name__}", "tuple第一个元素的类名为：", type(element[0]).__name__)

    # 为args添加新的值，即模型的config实例
    args.config = config

    return model

# 为model的每一层创建一个AdamW优化器，返回一个优化器列表
def create_optimizer(model):
    optimizer = []
    from gpt2_huggingface.optimization2 import AdamW
    for vlayer, _, _ in model:
        if len(list(vlayer.parameters())) == 0:
            optimizer.append(None)
        else:
            # 参数只能在cpu上
            for param in vlayer.parameters():
                assert not param.is_cuda
            optim = AdamW(vlayer.parameters(), lr=3e-5, weight_decay=0.0)
            optimizer.append(optim)
    return optimizer

# 计算交叉熵损失
# named_tensors：模型最后一层的输出
# named_targets：目标值
# ❓为啥要删掉最后一个元素？
def compute_loss(last_vlayer, named_tensors, X_names, named_targets):
    # ...即在该维度上取所有元素
    # :-1即在中间这个维度上，删掉最后一个元素
    output = named_tensors[X_names[0]][..., :-1, :].contiguous()
    print("named_tensors删掉中间维度的最后一个元素后的形状为：", output.shape)
    print(output)
    # 将 output 张量进行形状变换，将其视作一个二维张量，第一维的大小自动计算，第二维的大小等于原始张量的最后一维大小
    output = output.view(-1, output.size(-1))
    print("named_tensors形状变换后的形状为：", output.shape)
    # 在最后一个维度上删掉第一个元素
    shift_labels = named_targets["labels"][..., 1:].contiguous()
    print("target_tensors删掉中间维度的第一个元素后的形状为：", shift_labels.shape)
    # 调用损失函数计算loss
    print("target_tensors形状变换后的形状为：",shift_labels.view(-1).shape)
    loss = last_vlayer(output, shift_labels.view(-1))
    print("loss为：", loss)
    return [loss]

if __name__ == "__main__":    
    import prof_args
    args = prof_args.initialize_args(custom_args_provider=add_args)
    print("main函数收到的args为:", args)

    import profiler
    # 在cuda:0上进行profile，将收集的信息保存到.pickle文件中
    # 1.model、输入数据、label数据、Profilerlei的初始化
    # 2.通过倍增法探测最大 microbatch 的大小（1，2，4，8）
    # 3.使用刚刚得到的最大micro batch执行探测：
    #   3.1.FWDBWD：收集每一层的前后向(的平均)执行时间、内存占用(最大内存使用量)、输入该层的数据的元数据、label tensor的元数据
    #   3.2.UDP：收集每一层在cpu上的(平均)参数更新时间、参数相关的元数据、buffer相关元数据、优化器状态相关元数据
    profiler.run(args, synthetic_data, create_model, create_optimizer, compute_loss=compute_loss)
