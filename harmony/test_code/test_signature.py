import inspect

def example_function(a, b=10, *args, **kwargs):
    pass

# 使用 inspect.signature 获取函数参数签名
signature = inspect.signature(example_function)

# 输出参数信息
for param_name, param in signature.parameters.items():
    print(param_name, param.default)

for param in signature.parameters.values():
    print(param, type(param), param.name)

# 输出参数的默认值和其他信息
print(signature)
