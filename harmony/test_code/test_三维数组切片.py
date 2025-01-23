import torch

# 创建一个3维张量作为例子
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]])
print("原始张量:")
print(tensor)
print("原始张量形状:", tensor.shape)

# 使用切片操作去除最后一个维度的最后一个元素
# sliced_tensor = tensor[:, 1:-1]
# print("\n切片后的张量:")
# print(sliced_tensor)
# print("切片后的张量形状:", sliced_tensor.shape)

sliced_tensor = tensor[..., :-1, :]
print("\n切片后的张量:")
print(sliced_tensor)
print("切片后的张量形状:", sliced_tensor.shape)

t2 = torch.tensor([[1,2,3],[4,5,6]])
sliced_t2 = t2[..., 1:]
print(sliced_t2.contiguous())

print("只输出删除的维度：")
sliced_tensor = tensor[..., -1, :]
print(sliced_tensor)