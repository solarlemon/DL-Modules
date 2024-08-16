import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, merge,rank=16,lora_alpha=16,dropout_rate=0.5):
        """
        in_features (int): 输入特征的数量。
        out_features (int): 输出特征的数量。
        merge (bool): 是否合并输入特征。如果为True，则将输入特征直接作为输出特征的一部分；如果为False，则通过模型进行转换。
        rank (int, optional): LORA（Low-Rank Adaptation）的秩。默认为16。
        lora_alpha (int, optional): LORA的alpha值。决定了权重矩阵中可训练部分的规模。默认为16。
        dropout (float, optional): dropout层的丢弃率。默认为0.5。
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.merge = merge
        self.rank = rank
        
        self.lora_alpha = lora_alpha
        self.dropout = dropout_rate
        
        self.lin = nn.Linear(in_features, out_features)
        if rank > 0: # 安全性声明
            self.lora_b = nn.parameter(torch.zeros(out_features,rank))
            self.lora_a= nn.parameter(torch.zeros(rank, in_features))
            self.scale=self.lora_alpha/self.rank # 权重系数
            self.linear.weight.requires_grad = False # 关闭权重更新
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity() # 防止dropout层被调用，不应用dropout
        
        self.initial_weigth()
        
    def initial_weigth(self):
        """初始化权重矩阵。"""
        nn.init.zeros_(self.lora_b)
        if self.rank > 0: # 安全性声明
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
    
    def forward(self,x):
        if self.rank > 0 and self.merge:
            output=F.linear(x,self.linear.weight,self.lora_b @ self.lora_a*self.scale,self.linear.bias) # 注意是out_features*in_features
            output=self.dropout(output)
            return output
        else:
            return self.dropout(self.linear(x))
            