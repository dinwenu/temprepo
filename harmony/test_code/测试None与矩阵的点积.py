import torch
 
# 创建两个矩阵
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
 
# 方法1: 使用torch.mm()
C = torch.mm(A, B)
 
# 方法2: 使用@运算符
D = A @ B
 
print(C)  # 输出C，即点积结果
print(D)  # 输出D，同样是点积结果

# 1,2   5,6
# 3,4   7,8

C = None
D = [[7,8],[9,10]]
print(C @ D) 