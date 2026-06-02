from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from dataclasses import dataclass

_SUPPORTED_ACTIVATIONS = {"elu", "relu", "leaky_relu", "tanh", "silu"}
_SUPPORTED_OPTIMIZERS = {"adam", "adamw", "sgd"}


def _get_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "elu": return nn.ELU()
    if act == "relu": return nn.ReLU()
    if act == "leaky_relu": return nn.LeakyReLU(negative_slope=0.2)
    if act == "tanh": return nn.Tanh()
    if act == "silu": return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int, activation: str, last_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [in_dim, *hidden_dims]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(_get_activation(activation))
    layers.append(nn.Linear(dims[-1], out_dim))
    if last_activation:
        layers.append(_get_activation(activation))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


@dataclass
class CENetConfig:
    """Configuration strictly matched to DreamWaQ CENet paper."""
    obs_dim: int = 1
    history_length: int = 1
    latent_dim: int = 16  # z_t dimension
    encoder_hidden_dims: tuple[int, ...] = (128, 64)  # 参考论文架构图
    decoder_hidden_dims: tuple[int, ...] = (64, 128)  # 参考论文架构图
    activation: str = "elu"  # RL领域常用ELU
    
    beta: float = 1.0  # beta-VAE 的 KL 惩罚系数
    term_dims: tuple[int, ...] = (0,)
    
    logvar_min: float = -10.0
    logvar_max: float = 10.0
    
    # 优化器配置
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    optimizer_type: str = "adam"
    weight_decay: float = 1e-5

    def __post_init__(self) -> None:
        self.obs_dim = int(self.obs_dim)
        self.history_length = int(self.history_length)
        self.latent_dim = int(self.latent_dim)
        self.activation = self.activation.lower()


# ---------------------------------------------------------
# 数据类：修正了论文中的输出结构
# z 是随机变量(带mu, logvar), v 是确定性估计(仅 v_est)
# ---------------------------------------------------------
@dataclass
class CENetState:
    z: torch.Tensor          # Context vector sample (z_t)
    z_mu: torch.Tensor       # Prior mean
    z_logvar: torch.Tensor   # Prior log variance
    v_est: torch.Tensor      # Body linear velocity estimation (v_t)


@dataclass
class CENetMetrics:
    loss_ce: float           # L_CE (Total Loss)
    loss_est: float          # L_est (Velocity MSE)
    loss_vae: float          # L_VAE (Recons + Beta * KL)
    loss_recons: float       # Next obs reconstruction MSE
    loss_kld: float          # KL Divergence of z
    valid_ratio: float


class CENet(nn.Module):
    """
    Context Estimation Network (CENet) exactly as described in the DreamWaQ paper.
    Contains explicit velocity estimation and implicit context inference via beta-VAE.
    """

    def __init__(self, cfg: CENetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.latent_dim
        self.vel_dim = 3  # 线速度 v_t 固定为3维 (x, y, z)
        
        encoder_in_dim = cfg.obs_dim * cfg.history_length
        # 共享编码器的输出特征维度，随意定一个中间值
        encoder_out_dim = 64 
        
        self.encoder = _build_mlp(
            in_dim=encoder_in_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            out_dim=encoder_out_dim,
            activation=cfg.activation,
            last_activation=True,
        )
        
        # 💡 核心修正 1：输出由 (mu_z, logvar_z, v_est) 组成
        # z_dim * 2 (给隐变量 z 的均值和对数方差) + vel_dim (确定性的速度预测)
        self.shared_head = nn.Linear(encoder_out_dim, 2 * self.z_dim + self.vel_dim)
        
        # 解码器输入：根据架构图，z_t 和 v_t 拼接后送入 Decoder
        self.decoder = _build_mlp(
            in_dim=self.z_dim + self.vel_dim,
            hidden_dims=cfg.decoder_hidden_dims,
            out_dim=cfg.obs_dim,
            activation=cfg.activation,
        )
        
        valid_obs_dims = [i for i in range(cfg.obs_dim) if i not in cfg.term_dims]
        self.register_buffer("recons_valid_idx", torch.tensor(valid_obs_dims, dtype=torch.long))

        self.apply(_orthogonal_init)

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        raise ValueError(f"Unsupported optimizer")

    def encode(self, obs_history: torch.Tensor) -> CENetState:
        """编码观测历史，分离出隐变量 z 的分布和确定的速度估计 v_t"""
        features = self.encoder(obs_history.flatten(start_dim=1))
        
        # 按维度截断：z_mu, z_logvar, v_est
        z_mu, z_logvar, v_est = self.shared_head(features).split(
            [self.z_dim, self.z_dim, self.vel_dim], dim=-1
        )
        
        z_logvar = torch.clamp(z_logvar, self.cfg.logvar_min, self.cfg.logvar_max)
        
        # 仅对 z 进行重参数化采样，v_est 直接输出
        z = z_mu + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mu)
        
        return CENetState(z=z, z_mu=z_mu, z_logvar=z_logvar, v_est=v_est)

    def decode(self, z: torch.Tensor, v_est: torch.Tensor) -> torch.Tensor:
        """联合 z_t 和 v_t 重建下一时刻的观测 O_{t+1}"""
        return self.decoder(torch.cat([z, v_est], dim=-1))

    # --- 暴露给 RL 的接口 ---
    
    def inference(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout 使用：返回采样的上下文 z_t 和速度估计 v_t"""
        state = self.encode(obs_history)
        return state.z, state.v_est
        
    def evaluate(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """测试阶段使用：取分布均值，消除随机性"""
        state = self.encode(obs_history)
        return state.z_mu, state.v_est

    # 💡 核心修正 3：加入 AdaBoot (自适应引导) 的计算公式
    @staticmethod
    def compute_adaboot_prob(episode_rewards: torch.Tensor, eps: float = 1e-5) -> float:
        """
        计算论文中的自适应引导概率 p_boot = 1 - tanh(CV(R))
        :param episode_rewards: 当前 batch/环境池 中的回合奖励 (Tensor)
        :return: 使用 bootstrap (估计状态) 的概率
        """
        if episode_rewards.numel() <= 1:
            return 1.0  # 默认完全引导
            
        mean_r = episode_rewards.mean()
        std_r = episode_rewards.std()
        
        # 变异系数 Coefficient of Variation
        cv_r = std_r / (mean_r.abs() + eps) 
        
        # tanh 将上限平滑地变为 1
        p_boot = 1.0 - torch.tanh(cv_r).item()
        
        # 返回概率约束在 [0, 1] 之间
        return max(0.0, min(1.0, p_boot))

    # ---------------------------------------------------------
    # 💡 核心修正 2：损失函数完全对齐论文公式
    # ---------------------------------------------------------
    def update_step(
        self,
        optimizer: torch.optim.Optimizer,
        obs_history: torch.Tensor,
        next_obs: torch.Tensor,
        target_velocity: torch.Tensor,
        dones: torch.Tensor | None = None,
    ) -> CENetMetrics:
        
        self.train()

        # 1. 前向传播
        state = self.encode(obs_history)
        
        # 注意：依据架构图，Decoder 是接受估计的速度 (v_est) 和 z_t 作为输入的
        next_obs_pred = self.decode(state.z, state.v_est)

        # 2. 计算各部分损失
        
        # 公式: L_est = MSE(\tilde{v}_t, v_t)
        loss_est = nn.functional.mse_loss(state.v_est, target_velocity, reduction="none").mean(dim=-1)
        
        # 公式: MSE(\tilde{o}_{t+1}, o_{t+1}) (剔除特权信息)
        loss_recons = nn.functional.mse_loss(
            next_obs_pred[..., self.recons_valid_idx], 
            next_obs[..., self.recons_valid_idx], 
            reduction="none"
        ).mean(dim=-1)
        
        # 公式: KL( q(z_t | o_H) || p(z_t) )，先验 p(z) 为标准正态分布
        loss_kld = -0.5 * torch.sum(1.0 + state.z_logvar - state.z_mu.pow(2) - state.z_logvar.exp(), dim=-1)
        
        # 公式: L_VAE = MSE_recons + \beta * D_KL
        loss_vae = loss_recons + self.cfg.beta * loss_kld
        
        # 公式: L_CE = L_est + L_VAE
        loss_ce = loss_est + loss_vae

        # 3. 处理掩码并计算均值 (消除无效 done step)
        if dones is not None:
            mask = (dones.reshape(-1) == 0).float()
        else:
            mask = torch.ones_like(loss_ce)
            
        mask_sum = mask.sum()
        valid_ratio = (mask_sum / mask.numel()).item()

        if mask_sum < 1e-5:
            return CENetMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # 加权求均值
        optimize_loss = (loss_ce * mask).sum() / mask_sum

        # 4. 反向传播更新
        optimizer.zero_grad(set_to_none=True)
        optimize_loss.backward()
        if self.cfg.max_grad_norm is not None and self.cfg.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), self.cfg.max_grad_norm)
        optimizer.step()

        # 5. 日志数据
        return CENetMetrics(
            loss_ce=optimize_loss.detach().item(),
            loss_est=((loss_est * mask).sum() / mask_sum).detach().item(),
            loss_vae=((loss_vae * mask).sum() / mask_sum).detach().item(),
            loss_recons=((loss_recons * mask).sum() / mask_sum).detach().item(),
            loss_kld=((loss_kld * mask).sum() / mask_sum).detach().item(),
            valid_ratio=valid_ratio
        )