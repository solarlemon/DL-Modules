import torch
import torch.nn as nn

ps = nn.PixelShuffle(3)
# 创建一个1*9*4*4的张量作为输入
input = torch.randn(1, 9, 4, 4)
output = ps(input)
print(output.size())  # torch.Size([1, 1, 12, 12])
