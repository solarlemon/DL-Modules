import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads=5):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_o(output)

        return output


# 准备数据
# 对于CrossAttention，它需要输入两个三维张量x1和x2，分别代表(batch_size, seq_len1, in_dim1)和(batch_size, seq_len2, in_dim2)
# 其中seq_len1和seq_len2可以不相同。
num_hiddens, num_heads = 100, 5

# 准备数据
input_a = torch.ones(16, 36, 192)  # 输入序列A，大小为(batch_size, seq_len_a, input_dim_a)
input_b = torch.ones(16, 192, 36)  # 输入序列B，大小为(batch_size, seq_len_b, input_dim_b)
input_dim_a = input_a.shape[-1]  # 192
input_dim_b = input_b.shape[-1]  # 36

# 定义模型
cross_attention = CrossAttention(input_dim_a, input_dim_b, num_hiddens, num_heads)
cross_attention.eval()

print(cross_attention(input_a, input_b).shape)  # torch.Size([16, 36, 192])
