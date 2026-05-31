# from dataclasses import dataclass
# from typing import Dict, Iterable, Optional, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # 这是一个可重用的CENet风格VAE模块，输入是obs_history、next_obs和target_velocity，输出是重构的next_obs和预测的velocity，以及潜在变量z。它包含编码器和解码器网络，并提供了计算损失和更新参数的函数。
# # 论文 DreamWaQ

# def _get_activation(name: str) -> nn.Module:
#     act = name.lower()
#     if act == "elu":
#         return nn.ELU()
#     if act == "relu":
#         return nn.ReLU()
#     if act == "leaky_relu":
#         return nn.LeakyReLU(negative_slope=0.2)
#     if act == "tanh":
#         return nn.Tanh()
#     if act == "silu":
#         return nn.SiLU()
#     raise ValueError(f"Unsupported activation: {name}")


# def _build_mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int, activation: str) -> nn.Sequential:
#     layers = []
#     dims = [in_dim, *hidden_dims]
#     for i in range(len(dims) - 1):
#         layers.append(nn.Linear(dims[i], dims[i + 1]))
#         layers.append(_get_activation(activation))
#     layers.append(nn.Linear(dims[-1], out_dim))
#     return nn.Sequential(*layers)


# def _orthogonal_init(module: nn.Module) -> None:
#     if isinstance(module, nn.Linear):
#         nn.init.orthogonal_(module.weight)
#         if module.bias is not None:
#             nn.init.constant_(module.bias, 0.0)


# @dataclass
# class CENetVAEConfig:
#     obs_dim: int = 1
#     history_length: int = 1
#     latent_dim: int = 16
#     encoder_hidden_dims: Tuple[int, ...] = (512, 256)
#     decoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
#     activation: str = "elu"
#     beta: float = 1.0
#     learning_rate: float = 1e-3
#     max_grad_norm: float = 1.0
#     decode_with_target_velocity: bool = True
#     velocity_loss_use_sample: bool = True
#     term_dims: Tuple[int, ...] = (0,)
    
#     optimizer_type: str = "adam"   # 可选 "adam", "adamw", "sgd"
#     weight_decay: float = 1e-5      # 推荐给 AdamW 设置 1e-4 或 1e-5

# class CENetVAE(nn.Module):
#     """
#     Reusable CENet-style VAE module.

#     Inputs:
#       obs_history: [B, T, obs_dim]
#       next_obs: [B, obs_dim]
#       target_velocity: [B, 3]
#     """

#     def __init__(self, cfg: CENetVAEConfig) -> None:
#         super().__init__()
#         self.cfg = cfg
#         encoder_in_dim = cfg.obs_dim * cfg.history_length
#         encoder_out_dim = cfg.latent_dim * 4

#         self.encoder = _build_mlp(
#             in_dim=encoder_in_dim,
#             hidden_dims=cfg.encoder_hidden_dims,
#             out_dim=encoder_out_dim,
#             activation=cfg.activation,
#         )
#         self.latent_mu = nn.Linear(encoder_out_dim, cfg.latent_dim)
#         self.latent_logvar = nn.Linear(encoder_out_dim, cfg.latent_dim)
#         self.vel_mu = nn.Linear(encoder_out_dim, 3)
#         self.vel_logvar = nn.Linear(encoder_out_dim, 3)

#         self.decoder = _build_mlp(
#             in_dim=cfg.latent_dim + 3,
#             hidden_dims=cfg.decoder_hidden_dims,
#             out_dim=cfg.obs_dim,
#             activation=cfg.activation,
#         )
#         self.apply(_orthogonal_init)

#     def create_optimizer(self) -> torch.optim.Optimizer:
#         opt_type = self.cfg.optimizer_type.lower()
        
#         if opt_type == "adam":
#             return torch.optim.Adam(
#                 self.parameters(), 
#                 lr=self.cfg.learning_rate, 
#                 weight_decay=self.cfg.weight_decay
#             )
#         elif opt_type == "adamw":
#             return torch.optim.AdamW(
#                 self.parameters(), 
#                 lr=self.cfg.learning_rate, 
#                 weight_decay=self.cfg.weight_decay
#             )
#         elif opt_type == "sgd":
#             return torch.optim.SGD(
#                 self.parameters(), 
#                 lr=self.cfg.learning_rate, 
#                 weight_decay=self.cfg.weight_decay,
#                 momentum=0.9
#             )
#         else:
#             raise ValueError(f"Unsupported optimizer type: {opt_type}")

#     @staticmethod
#     def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode(self, obs_history: torch.Tensor) -> Dict[str, torch.Tensor]:
#         bs = obs_history.shape[0]
#         features = self.encoder(obs_history.reshape(bs, -1))
#         z_mu = self.latent_mu(features)
#         z_logvar = self.latent_logvar(features)
#         vel_mu = self.vel_mu(features)
#         vel_logvar = self.vel_logvar(features)
#         return {
#             "z_mu": z_mu,
#             "z_logvar": z_logvar,
#             "vel_mu": vel_mu,
#             "vel_logvar": vel_logvar,
#         }

#     def decode(self, z: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
#         return self.decoder(torch.cat([z, velocity], dim=-1))

#     def predict(
#         self,
#         obs_history: torch.Tensor,
#         deterministic: bool = False,
#         decode_next_obs: bool = False,
#         decode_velocity: Optional[torch.Tensor] = None,
#     ) -> Dict[str, torch.Tensor]:
#         latent = self.encode(obs_history)
#         if deterministic:
#             z = latent["z_mu"]
#             vel = latent["vel_mu"]
#         else:
#             z = self.reparameterize(latent["z_mu"], latent["z_logvar"])
#             vel = self.reparameterize(latent["vel_mu"], latent["vel_logvar"])

#         out = {
#             "z": z,
#             "velocity": vel,
#             **latent,
#         }
#         if decode_next_obs:
#             vel_for_decode = decode_velocity if decode_velocity is not None else vel
#             out["next_obs_pred"] = self.decode(z, vel_for_decode)
#         return out

#     def compute_loss(
#         self,
#         obs_history: torch.Tensor,
#         next_obs: torch.Tensor,
#         target_velocity: torch.Tensor,
#         beta: Optional[float] = None,
#     ) -> Dict[str, torch.Tensor]:
#         if beta is None:
#             beta = self.cfg.beta

#         pred = self.predict(obs_history, deterministic=False, decode_next_obs=False)
#         z = pred["z"]
#         vel_sample = pred["velocity"]
#         z_mu = pred["z_mu"]
#         z_logvar = pred["z_logvar"]
#         vel_mu = pred["vel_mu"]

#         if self.cfg.decode_with_target_velocity:
#             next_obs_pred = self.decode(z, target_velocity)
#         else:
#             next_obs_pred = self.decode(z, vel_sample)

#         vel_for_loss = vel_sample if self.cfg.velocity_loss_use_sample else vel_mu

#         recons_loss = F.mse_loss(next_obs_pred, next_obs, reduction="none").mean(dim=-1)
#         vel_loss = F.mse_loss(vel_for_loss, target_velocity, reduction="none").mean(dim=-1)
#         kld_loss = -0.5 * torch.sum(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)

#         total_loss = recons_loss + vel_loss + beta * kld_loss
#         return {
#             "loss": total_loss,
#             "recons_loss": recons_loss,
#             "vel_loss": vel_loss,
#             "kld_loss": kld_loss,
#         }

#     def update_step(
#         self,
#         optimizer: torch.optim.Optimizer,
#         obs_history: torch.Tensor,
#         next_obs: torch.Tensor,
#         target_velocity: torch.Tensor,
#         dones: Optional[torch.Tensor] = None,
#         beta: Optional[float] = None,
#         max_grad_norm: Optional[float] = None,
#     ) -> Dict[str, float]:
#         self.train()
#         if max_grad_norm is None:
#             max_grad_norm = self.cfg.max_grad_norm

#         loss_dict = self.compute_loss(
#             obs_history=obs_history,
#             next_obs=next_obs,
#             target_velocity=target_velocity,
#             beta=beta,
#         )

#         if dones is not None:
#             valid = (dones.reshape(-1) == 0)
#             if valid.any():
#                 optimize_loss = loss_dict["loss"][valid].mean()
#                 metrics = {
#                     "loss": optimize_loss.detach().item(),
#                     "recons_loss": loss_dict["recons_loss"][valid].mean().detach().item(),
#                     "vel_loss": loss_dict["vel_loss"][valid].mean().detach().item(),
#                     "kld_loss": loss_dict["kld_loss"][valid].mean().detach().item(),
#                     "valid_ratio": valid.float().mean().detach().item(),
#                 }
#             else:
#                 return {
#                     "loss": 0.0,
#                     "recons_loss": 0.0,
#                     "vel_loss": 0.0,
#                     "kld_loss": 0.0,
#                     "valid_ratio": 0.0,
#                 }
#         else:
#             optimize_loss = loss_dict["loss"].mean()
#             metrics = {
#                 "loss": optimize_loss.detach().item(),
#                 "recons_loss": loss_dict["recons_loss"].mean().detach().item(),
#                 "vel_loss": loss_dict["vel_loss"].mean().detach().item(),
#                 "kld_loss": loss_dict["kld_loss"].mean().detach().item(),
#                 "valid_ratio": 1.0,
#             }

#         optimizer.zero_grad(set_to_none=True)
#         optimize_loss.backward()
#         if max_grad_norm is not None and max_grad_norm > 0.0:
#             nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
#         optimizer.step()
#         return metrics

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 这是一个可重用的CENet风格VAE模块，输入是obs_history、next_obs和target_velocity，输出是重构的next_obs和预测的velocity，以及潜在变量z。它包含编码器和解码器网络，并提供了计算损失和更新参数的函数。
# 论文 DreamWaQ

def _get_activation(name: str) -> nn.Module:
    act = name.lower()
    if act == "elu":
        return nn.ELU()
    if act == "relu":
        return nn.ReLU()
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if act == "tanh":
        return nn.Tanh()
    if act == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")

# 🚀 [修改点 1]：增加 last_activation 参数，支持在最后一层后添加激活函数
def _build_mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int, activation: str, last_activation: bool = False) -> nn.Sequential:
    layers = []
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
class CENetVAEConfig:
    obs_dim: int = 1
    history_length: int = 1
    latent_dim: int = 16
    encoder_hidden_dims: Tuple[int, ...] = (512, 256)
    decoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "silu"
    beta: float = 1.0
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    decode_with_target_velocity: bool = True
    velocity_loss_use_sample: bool = True
    term_dims: Tuple[int, ...] = (0,)
    
    optimizer_type: str = "adam"   # 可选 "adam", "adamw", "sgd"
    weight_decay: float = 1e-5      # 推荐给 AdamW 设置 1e-4 或 1e-5

class CENetVAE(nn.Module):
    """
    Reusable CENet-style VAE module.

    Inputs:
      obs_history: [B, T, obs_dim]
      next_obs: [B, obs_dim]
      target_velocity: [B, 3]
    """

    def __init__(self, cfg: CENetVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_in_dim = cfg.obs_dim * cfg.history_length
        encoder_out_dim = cfg.latent_dim * 4

        # 🚀 [修改点 2]：传入 last_activation=True，打破连续的 Linear-Linear 冗余
        self.encoder = _build_mlp(
            in_dim=encoder_in_dim,
            hidden_dims=cfg.encoder_hidden_dims,
            out_dim=encoder_out_dim,
            activation=cfg.activation,
            last_activation=True, 
        )
        self.latent_mu = nn.Linear(encoder_out_dim, cfg.latent_dim)
        self.latent_logvar = nn.Linear(encoder_out_dim, cfg.latent_dim)
        self.vel_mu = nn.Linear(encoder_out_dim, 3)
        self.vel_logvar = nn.Linear(encoder_out_dim, 3)

        self.decoder = _build_mlp(
            in_dim=cfg.latent_dim + 3,
            hidden_dims=cfg.decoder_hidden_dims,
            out_dim=cfg.obs_dim,
            activation=cfg.activation,
            last_activation=False, # 解码器最后输出就是物理数值，不需要激活
        )
        self.apply(_orthogonal_init)

    # 🚀 [修改点 3]：补充 PyTorch 标准的 forward 方法
    def forward(self, obs_history: torch.Tensor, deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        标准前向传播函数，规范化 Module 结构。
        默认进行确定性推理 (deterministic=True)，抛弃噪声采样。
        """
        return self.predict(obs_history, deterministic=deterministic, decode_next_obs=False)

    def create_optimizer(self) -> torch.optim.Optimizer:
        opt_type = self.cfg.optimizer_type.lower()
        
        if opt_type == "adam":
            return torch.optim.Adam(
                self.parameters(), 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay
            )
        elif opt_type == "adamw":
            return torch.optim.AdamW(
                self.parameters(), 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay
            )
        elif opt_type == "sgd":
            return torch.optim.SGD(
                self.parameters(), 
                lr=self.cfg.learning_rate, 
                weight_decay=self.cfg.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, obs_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs = obs_history.shape[0]
        features = self.encoder(obs_history.reshape(bs, -1))
        z_mu = self.latent_mu(features)
        z_logvar = self.latent_logvar(features)
        vel_mu = self.vel_mu(features)
        vel_logvar = self.vel_logvar(features)
        return {
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "vel_mu": vel_mu,
            "vel_logvar": vel_logvar,
        }

    def decode(self, z: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, velocity], dim=-1))

    def predict(
        self,
        obs_history: torch.Tensor,
        deterministic: bool = False,
        decode_next_obs: bool = False,
        decode_velocity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = self.encode(obs_history)
        if deterministic:
            z = latent["z_mu"]
            vel = latent["vel_mu"]
        else:
            z = self.reparameterize(latent["z_mu"], latent["z_logvar"])
            vel = self.reparameterize(latent["vel_mu"], latent["vel_logvar"])

        out = {
            "z": z,
            "velocity": vel,
            **latent,
        }
        if decode_next_obs:
            vel_for_decode = decode_velocity if decode_velocity is not None else vel
            out["next_obs_pred"] = self.decode(z, vel_for_decode)
        return out

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        next_obs: torch.Tensor,
        target_velocity: torch.Tensor,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        if beta is None:
            beta = self.cfg.beta

        pred = self.predict(obs_history, deterministic=False, decode_next_obs=False)
        z = pred["z"]
        vel_sample = pred["velocity"]
        z_mu = pred["z_mu"]
        z_logvar = pred["z_logvar"]
        vel_mu = pred["vel_mu"]

        if self.cfg.decode_with_target_velocity:
            next_obs_pred = self.decode(z, target_velocity)
        else:
            next_obs_pred = self.decode(z, vel_sample)

        vel_for_loss = vel_sample if self.cfg.velocity_loss_use_sample else vel_mu

        recons_loss = F.mse_loss(next_obs_pred, next_obs, reduction="none").mean(dim=-1)
        vel_loss = F.mse_loss(vel_for_loss, target_velocity, reduction="none").mean(dim=-1)
        kld_loss = -0.5 * torch.sum(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)

        total_loss = recons_loss + vel_loss + beta * kld_loss
        return {
            "loss": total_loss,
            "recons_loss": recons_loss,
            "vel_loss": vel_loss,
            "kld_loss": kld_loss,
        }

    def update_step(
        self,
        optimizer: torch.optim.Optimizer,
        obs_history: torch.Tensor,
        next_obs: torch.Tensor,
        target_velocity: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        self.train()
        if max_grad_norm is None:
            max_grad_norm = self.cfg.max_grad_norm

        loss_dict = self.compute_loss(
            obs_history=obs_history,
            next_obs=next_obs,
            target_velocity=target_velocity,
            beta=beta,
        )

        if dones is not None:
            valid = (dones.reshape(-1) == 0)
            if valid.any():
                optimize_loss = loss_dict["loss"][valid].mean()
                metrics = {
                    "loss": optimize_loss.detach().item(),
                    "recons_loss": loss_dict["recons_loss"][valid].mean().detach().item(),
                    "vel_loss": loss_dict["vel_loss"][valid].mean().detach().item(),
                    "kld_loss": loss_dict["kld_loss"][valid].mean().detach().item(),
                    "valid_ratio": valid.float().mean().detach().item(),
                }
            else:
                return {
                    "loss": 0.0,
                    "recons_loss": 0.0,
                    "vel_loss": 0.0,
                    "kld_loss": 0.0,
                    "valid_ratio": 0.0,
                }
        else:
            optimize_loss = loss_dict["loss"].mean()
            metrics = {
                "loss": optimize_loss.detach().item(),
                "recons_loss": loss_dict["recons_loss"].mean().detach().item(),
                "vel_loss": loss_dict["vel_loss"].mean().detach().item(),
                "kld_loss": loss_dict["kld_loss"].mean().detach().item(),
                "valid_ratio": 1.0,
            }

        optimizer.zero_grad(set_to_none=True)
        optimize_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return metrics