import torch
from torch import einsum

a = torch.ones(3, 4)
b = torch.ones(4, 5)
c = torch.ones(6, 7, 8)
d = torch.ones(3, 4)
x, y = torch.randn(5), torch.randn(5)

print(einsum("ij->i", a))  # tensor([4., 4., 4.])  行求和
