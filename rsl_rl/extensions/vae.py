# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from dataclasses import dataclass


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


def _build_mlp(
    in_dim: int,
    hidden_dims: Iterable[int],
    out_dim: int,
    activation: str,
    last_activation: bool = False,
) -> nn.Sequential:
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
class CENetVAEConfig:
    """Configuration for the DreamWaQ CENet-style VAE."""

    obs_dim: int = 1
    history_length: int = 1
    latent_dim: int = 16
    encoder_hidden_dims: tuple[int, ...] = (512, 256)
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "silu"
    beta: float = 1.0
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    decode_with_target_velocity: bool = True
    velocity_loss_use_sample: bool = True
    term_dims: tuple[int, ...] = (0,)
    optimizer_type: str = "adam"
    weight_decay: float = 1e-5


class CENetVAE(nn.Module):
    """Reusable CENet-style VAE used by DreamWaQ.

    Inputs:
        obs_history: Tensor shaped ``[batch, history_length, obs_dim]``.
        next_obs: Tensor shaped ``[batch, obs_dim]``.
        target_velocity: Tensor shaped ``[batch, 3]``.
    """

    def __init__(self, cfg: CENetVAEConfig) -> None:
        """Initialize encoder, latent heads, velocity heads, and decoder."""
        super().__init__()
        self.cfg = cfg

        encoder_in_dim = cfg.obs_dim * cfg.history_length
        encoder_out_dim = cfg.latent_dim * 4
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
        )
        self.apply(_orthogonal_init)

    def forward(self, obs_history: torch.Tensor, deterministic: bool = True) -> dict[str, torch.Tensor]:
        """Run deterministic or stochastic inference and return latent and velocity predictions."""
        return self.predict(obs_history, deterministic=deterministic, decode_next_obs=False)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer requested by the VAE config."""
        opt_type = self.cfg.optimizer_type.lower()
        if opt_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if opt_type == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        if opt_type == "sgd":
            return torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, obs_history: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode observation history into latent and velocity distributions."""
        batch_size = obs_history.shape[0]
        features = self.encoder(obs_history.reshape(batch_size, -1))
        return {
            "z_mu": self.latent_mu(features),
            "z_logvar": self.latent_logvar(features),
            "vel_mu": self.vel_mu(features),
            "vel_logvar": self.vel_logvar(features),
        }

    def decode(self, z: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Decode latent state and velocity into the next observation."""
        return self.decoder(torch.cat([z, velocity], dim=-1))

    def predict(
        self,
        obs_history: torch.Tensor,
        deterministic: bool = False,
        decode_next_obs: bool = False,
        decode_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict latent state, velocity, and optionally the next observation."""
        latent = self.encode(obs_history)
        if deterministic:
            z = latent["z_mu"]
            velocity = latent["vel_mu"]
        else:
            z = self.reparameterize(latent["z_mu"], latent["z_logvar"])
            velocity = self.reparameterize(latent["vel_mu"], latent["vel_logvar"])

        out = {"z": z, "velocity": velocity, **latent}
        if decode_next_obs:
            velocity_for_decode = decode_velocity if decode_velocity is not None else velocity
            out["next_obs_pred"] = self.decode(z, velocity_for_decode)
        return out

    def compute_loss(
        self,
        obs_history: torch.Tensor,
        next_obs: torch.Tensor,
        target_velocity: torch.Tensor,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction, velocity, KL, and total VAE losses per sample."""
        if beta is None:
            beta = self.cfg.beta

        pred = self.predict(obs_history, deterministic=False, decode_next_obs=False)
        z = pred["z"]
        velocity_sample = pred["velocity"]
        z_mu = pred["z_mu"]
        z_logvar = pred["z_logvar"]
        velocity_mu = pred["vel_mu"]

        if self.cfg.decode_with_target_velocity:
            next_obs_pred = self.decode(z, target_velocity)
        else:
            next_obs_pred = self.decode(z, velocity_sample)

        velocity_for_loss = velocity_sample if self.cfg.velocity_loss_use_sample else velocity_mu
        recons_loss = nn.functional.mse_loss(next_obs_pred, next_obs, reduction="none").mean(dim=-1)
        vel_loss = nn.functional.mse_loss(velocity_for_loss, target_velocity, reduction="none").mean(dim=-1)
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
        dones: torch.Tensor | None = None,
        beta: float | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float]:
        """Run one standalone VAE optimization step."""
        self.train()
        if max_grad_norm is None:
            max_grad_norm = self.cfg.max_grad_norm

        loss_dict = self.compute_loss(obs_history, next_obs, target_velocity, beta)
        valid = dones.reshape(-1) == 0 if dones is not None else torch.ones_like(loss_dict["loss"], dtype=torch.bool)

        if not valid.any():
            return {"loss": 0.0, "recons_loss": 0.0, "vel_loss": 0.0, "kld_loss": 0.0, "valid_ratio": 0.0}

        optimize_loss = loss_dict["loss"][valid].mean()
        metrics = {
            "loss": optimize_loss.detach().item(),
            "recons_loss": loss_dict["recons_loss"][valid].mean().detach().item(),
            "vel_loss": loss_dict["vel_loss"][valid].mean().detach().item(),
            "kld_loss": loss_dict["kld_loss"][valid].mean().detach().item(),
            "valid_ratio": valid.float().mean().detach().item(),
        }

        optimizer.zero_grad(set_to_none=True)
        optimize_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return metrics
