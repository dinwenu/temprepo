import torch.nn as nn

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleModel()

# 获取模型中的所有可学习参数
# params = list(model.parameters())
# print(params)

# print("named_parameters()")
# for param in model.named_parameters():
#     print(param)

print(len(model._parameters))
for key, param in model._parameters.items():
    print("key")

# 会输出整个模型
for m in model.modules():
    print(f"m:{m}, m's type:{type(m)}, len of m:{len(m._parameters)}")
    for key, param in m._parameters.items():
        print(f"key:{key}, param:{param}")

print("输出buffer")
print(model.named_buffers())
for name, buf in model.named_buffers():
    print(f"name:{name}, buffer:{buf}")
    if buf is None:
        print("不存在buffer")
    else:
        print("存在buffer")