import torch
import math

# 参数定义
batch_size = 2
seq_len = 3
feature_len = 4 # 在Transformer中这个feature_len就是d_model
eps = 1e-5 # 一个极小的常数，防止分母为0

# 数据定义
x = torch.arange(24).reshape(2,3,4).float()
print(x)

ln_mean = x.mean(dim=-1, keepdim=True) 
print(ln_mean)
ln_std = x.std(dim=-1, unbiased=False, keepdim=True)
print(ln_std)
verify_ln_y = (x - ln_mean) / (ln_std + 1e-5)

print(verify_ln_y)