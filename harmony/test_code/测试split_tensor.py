import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

chunks = torch.split(x, split_size_or_sections=2, dim=0)

print(type(chunks))
print(chunks)