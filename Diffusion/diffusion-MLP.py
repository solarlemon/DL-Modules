import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLoss(nn.Module):
    # 自定义一个损失函数类，因为有两种损失函数，所以需要继承nn.Module
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, targ, weight=1.0):
        """
        :param pred: 预测值 targ: 目标值 形状为[batch_size, action_dim] 目标的噪声-预测的噪声
        """
        losses = self._loss(pred, targ)
        WeightedLoss = (losses*weight).mean()
        return WeightedLoss


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="None")


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2
}


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class SinusoidalPosEmb(nn.Module):
    # 自定义一个正弦位置编码类
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):  # 状态维度、动作维度、隐藏层、时间维度
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),  # 激活函数
            nn.Linear(t_dim*2, t_dim)
        )
        input_dim = state_dim + action_dim + t_dim

        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weight()

    def init_weight(self):
        # 遍历mid_layer中的每一层，如果某一层是nn.Linear类型
        # 那么就使用Xavier均匀分布初始化该层的权重（layer.weight）
        # 并将偏置项（layer.bias）初始化为0。
        for layer in self.mid_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class Diffusion(nn.Module):
    # 损失类型、beta的更新方式（linear离散）、对预测的噪声进行CLIP操作
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, predict_epsilon=True, **kwargs):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs["obs_dim"]  # 读取参数
        self.action_dim = kwargs["act_dim"]
        self.hidden_dim = kwargs["hid_dim"]
        self.T = kwargs["T"]
        self.device = kwargs["device"]
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == "linear":
            betas = torch.linspace(0.001, 0.02, self.T, dtype=torch.float32)  # 生成一个从0.001到0.02的线性分布（得到beta_t）
        alphas = 1.0 - betas  # 得到alpha_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算alpha_t的累积乘积 [1,2,3] -> [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.ones(1,), alphas_cumprod[:-1]])  # [1,...alpha_{t-1}]

        self.register_buffer("alphas_cumprod", alphas_cumprod)  # 注册为buffer
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)

        # 前向过程
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_1m_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # 反向过程
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp(min=1e-20)))

        # 已知x_t，求解x_0
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0/alphas_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0/alphas_cumprod-1))
        # 后两项参数
        self.register_buffer("posterior_mean_coef1", betas*torch.sqrt(alphas_cumprod_prev)/(1.0-alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0-alphas_cumprod_prev)*torch.sqrt(alphas)/(1.0-alphas_cumprod))

        # 构建loss
        self.loss_fn = Losses[loss_type]()

    def q_poster(self, x_start, x, t):
        # 计算后验概率
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        """
        :param x: 状态 [bat_size, action_dim]
        :param t: 步数 [bat_size,]
        :param pred_noise: 噪声 [bat_size, action_dim]
        """
        return (extract(self.sqrt_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise)

    def p_mean_variance(self, x, t, s):
        pred_noise = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1, 1)  # 直接在x_recon上做clip，而不是在pred_noise上做clip

        # 已知x_t和x_0，求解x_t-1,
        model_mean, posterior_variance, posterior_log_variance = self.model(x_recon, x, t)
        return model_mean, posterior_log_variance

    def p_sample(self, x, t, state):
        """
        :param x: 状态 [bat_size, action_dim]
        :param t: 步数 [bat_size,]
        :param state: 状态 [bat_size, state_dim]
        """
        b, *_, device = *x.shape, self.device
        modal_mean, modal_log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)  # 生成一个标准正态分布，不需要反向过程的
        # t>1才加noise
        nonezero_mask = (1-(t == 0).float()).reshape(b, *((1,)*(len(x.shape)-1)))
        return modal_mean + nonezero_mask * (0.5*modal_log_variance.exp() * noise)

    def p_sample_loop(self, state, shape, *args, **kwargs):
        """
        :param state: 状态 [bat_size, state_dim]
        :param shape: 形状 [bat_size, action_dim]
        """
        device = self.device
        # 采样过程
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)  # 生成一个标准正态分布，不需要反向过程的梯度
        for i in reversed(range(self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)  # 生成一个从0到T-1的序列 反向传播过程的第几步
            x = self.p_sample(x, t, state)
        return x

    def sample(self, state, *args, **kwargs):
        """
        :param state: 状态 [bat_size, state_dim]
        """
        # 采样过程
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)  # 通过DDPM反向不断扩散的过程得到action
        return action.clamp_(-1, 1)

    def q_sample(self, x_start, t, noise):
        """
        :param x_start: 状态 [bat_size, action_dim]
        :param t: 步数 [bat_size,]
        :param noise: 噪声 [bat_size, action_dim]
        """
        # 前向过程
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_recipm_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weigths=1.0):
        noise = torch.randn_like(x_start)  # 维度相同
        # 生成噪声标签
        x_noise = self.q_sample(x_start, t, noise)  # 前向过程 已知x_0和第t步，求解x_t
        x_recon = self.model(x_noise, t, state)  # 神经网络预测的噪声
        loss = self.loss_fn(x_recon, noise, weigths)
        return loss

    def loss(self, x_start, state, weigths=1.0):
        """
        :param x_start: x_0 
        """
        batch_size = len(x_start)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x_start, state, t, weigths)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


if __name__ == "__main__":
    device = "cpu"  # cuda
    x = torch.randn(256, 2).to(device)  # Batch, action_dim
    state = torch.randn(256, 11).to(device)  # Batch, state_dim
    model = Diffusion(loss_type="l2", obs_dim=11, act_dim=2, hid_dim=256, T=100, device=device)
    result = model(state)  # Sample result

    loss = model.loss(x, state)

    print(f"loss: {loss.item()}")
