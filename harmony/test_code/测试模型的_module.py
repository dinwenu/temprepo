import torch
import torch.nn as nn
from collections import OrderedDict

class CustomModule(nn.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 创建一个模块实例
module = CustomModule()

# 访问 _modules 属性
sub_modules = module.__dict__['_modules']

# 打印子模块
print(sub_modules)