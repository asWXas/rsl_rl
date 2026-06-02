# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

_SUPPORTED_ACTIVATIONS = {"elu", "relu", "leaky_relu", "tanh", "silu"}
_SUPPORTED_OPTIMIZERS = {"adam", "adamw", "sgd"}


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


class CENetVAEDistribution(TypedDict):
    """Encoder distribution parameters."""

    z_mu: torch.Tensor
    z_logvar: torch.Tensor
    vel_mu: torch.Tensor
    vel_logvar: torch.Tensor


class CENetVAEBaseOutput(CENetVAEDistribution):
    """Common VAE prediction output."""

    z: torch.Tensor
    velocity: torch.Tensor


class CENetVAEOutput(CENetVAEBaseOutput, total=False):
    """VAE prediction output with optional decoded next observation."""

    next_obs_pred: torch.Tensor


class CENetVAELoss(TypedDict):
    """Per-sample VAE loss tensors."""

    loss: torch.Tensor
    recons_loss: torch.Tensor
    vel_loss: torch.Tensor
    kld_loss: torch.Tensor
    z_kld_loss: torch.Tensor
    vel_kld_loss: torch.Tensor


class CENetVAEMetrics(TypedDict):
    """Scalar metrics returned by one standalone VAE update."""

    loss: float
    recons_loss: float
    vel_loss: float
    kld_loss: float
    z_kld_loss: float
    vel_kld_loss: float
    valid_ratio: float


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
    velocity_kld_weight: float = 0.0
    learning_rate: float = 1e-3
    max_grad_norm: float | None = 1.0
    decode_with_target_velocity: bool = True
    velocity_loss_use_sample: bool = True
    term_dims: tuple[int, ...] = (0,)
    optimizer_type: str = "adam"
    weight_decay: float = 1e-5
    logvar_min: float = -10.0
    logvar_max: float = 10.0

    def __post_init__(self) -> None:
        """Validate and normalize config values early."""
        self.obs_dim = int(self.obs_dim)
        self.history_length = int(self.history_length)
        self.latent_dim = int(self.latent_dim)
        self.encoder_hidden_dims = tuple(int(dim) for dim in self.encoder_hidden_dims)
        self.decoder_hidden_dims = tuple(int(dim) for dim in self.decoder_hidden_dims)
        self.term_dims = tuple(int(dim) for dim in self.term_dims)
        self.activation = self.activation.lower()
        self.optimizer_type = self.optimizer_type.lower()

        if self.obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {self.obs_dim}.")
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive, got {self.history_length}.")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}.")
        if any(dim <= 0 for dim in self.encoder_hidden_dims):
            raise ValueError(f"encoder_hidden_dims must contain only positive values, got {self.encoder_hidden_dims}.")
        if any(dim <= 0 for dim in self.decoder_hidden_dims):
            raise ValueError(f"decoder_hidden_dims must contain only positive values, got {self.decoder_hidden_dims}.")
        if any(dim < 0 for dim in self.term_dims):
            raise ValueError(f"term_dims must contain only non-negative values, got {self.term_dims}.")
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.optimizer_type not in _SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}.")
        if self.velocity_kld_weight < 0.0:
            raise ValueError(f"velocity_kld_weight must be non-negative, got {self.velocity_kld_weight}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if self.max_grad_norm is not None and self.max_grad_norm < 0.0:
            raise ValueError(f"max_grad_norm must be non-negative or None, got {self.max_grad_norm}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if self.logvar_min > self.logvar_max:
            raise ValueError(f"logvar_min ({self.logvar_min}) must be <= logvar_max ({self.logvar_max}).")


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
        self.z_dim = cfg.latent_dim  # 16
        self.vel_dim = 3
        self.total_param_dim = 2 * self.z_dim + 2 * self.vel_dim  # 38
        self.shared_head = nn.Linear(encoder_out_dim, self.total_param_dim)

        self.decoder = _build_mlp(
            in_dim=cfg.latent_dim + 3,
            hidden_dims=cfg.decoder_hidden_dims,
            out_dim=cfg.obs_dim,
            activation=cfg.activation,
        )
        self.apply(_orthogonal_init)

    def forward(self, obs_history: torch.Tensor, deterministic: bool = True) -> CENetVAEOutput:
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

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(logvar, min=self.cfg.logvar_min, max=self.cfg.logvar_max)

    def _flatten_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        if obs_history.dim() == 3:
            expected_shape = (self.cfg.history_length, self.cfg.obs_dim)
            if obs_history.shape[1:] != expected_shape:
                raise ValueError(
                    f"Expected obs_history shape [batch, {expected_shape[0]}, {expected_shape[1]}], "
                    f"got {tuple(obs_history.shape)}."
                )
            return obs_history.reshape(obs_history.shape[0], -1)

        if obs_history.dim() == 2:
            expected_width = self.cfg.history_length * self.cfg.obs_dim
            if obs_history.shape[-1] != expected_width:
                raise ValueError(
                    f"Expected flattened obs_history width {expected_width}, got {obs_history.shape[-1]}."
                )
            return obs_history

        raise ValueError(f"Expected 2D or 3D obs_history, got shape {tuple(obs_history.shape)}.")

    @staticmethod
    def _validate_last_dim(tensor: torch.Tensor, name: str, expected_dim: int) -> None:
        if tensor.dim() == 0 or tensor.shape[-1] != expected_dim:
            raise ValueError(f"Expected {name} last dimension to be {expected_dim}, got shape {tuple(tensor.shape)}.")

    def encode(self, obs_history: torch.Tensor) -> CENetVAEDistribution:
        """Encode observation history into latent and velocity distributions."""
        features = self.encoder(self._flatten_history(obs_history))
        params = self.shared_head(features)
        z_mu, z_logvar, vel_mu, vel_logvar = params.split(
            [self.z_dim, self.z_dim, self.vel_dim, self.vel_dim], dim=-1
        )
        return {
            "z_mu": z_mu,
            "z_logvar": self._clamp_logvar(z_logvar),
            "vel_mu": vel_mu,
            "vel_logvar": self._clamp_logvar(vel_logvar),
        }

    def decode(self, z: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Decode latent state and velocity into the next observation."""
        self._validate_last_dim(z, "z", self.z_dim)
        self._validate_last_dim(velocity, "velocity", self.vel_dim)
        return self.decoder(torch.cat([z, velocity], dim=-1))

    def infer(
        self,
        obs_history: torch.Tensor,
        decode_next_obs: bool = False,
        decode_velocity: torch.Tensor | None = None,
    ) -> CENetVAEOutput:
        """Run deterministic inference for policy-time latent and velocity prediction."""
        return self.predict(
            obs_history,
            deterministic=True,
            decode_next_obs=decode_next_obs,
            decode_velocity=decode_velocity,
        )

    def sample(
        self,
        obs_history: torch.Tensor,
        decode_next_obs: bool = False,
        decode_velocity: torch.Tensor | None = None,
    ) -> CENetVAEOutput:
        """Run stochastic inference using reparameterized latent and velocity samples."""
        return self.predict(
            obs_history,
            deterministic=False,
            decode_next_obs=decode_next_obs,
            decode_velocity=decode_velocity,
        )

    def predict(
        self,
        obs_history: torch.Tensor,
        deterministic: bool = True,
        decode_next_obs: bool = False,
        decode_velocity: torch.Tensor | None = None,
    ) -> CENetVAEOutput:
        """Predict latent state, velocity, and optionally the next observation."""
        latent = self.encode(obs_history)
        if deterministic:
            z = latent["z_mu"]
            velocity = latent["vel_mu"]
        else:
            z = self.reparameterize(latent["z_mu"], latent["z_logvar"])
            velocity = self.reparameterize(latent["vel_mu"], latent["vel_logvar"])

        out: CENetVAEOutput = {
            "z": z,
            "velocity": velocity,
            "z_mu": latent["z_mu"],
            "z_logvar": latent["z_logvar"],
            "vel_mu": latent["vel_mu"],
            "vel_logvar": latent["vel_logvar"],
        }
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
    ) -> CENetVAELoss:
        """Compute reconstruction, velocity, KL, and total VAE losses per sample."""
        if beta is None:
            beta = self.cfg.beta

        self._validate_last_dim(next_obs, "next_obs", self.cfg.obs_dim)
        self._validate_last_dim(target_velocity, "target_velocity", self.vel_dim)
        pred = self.sample(obs_history, decode_next_obs=False)
        z = pred["z"]
        velocity_sample = pred["velocity"]
        z_mu = pred["z_mu"]
        z_logvar = pred["z_logvar"]
        velocity_mu = pred["vel_mu"]
        velocity_logvar = pred["vel_logvar"]

        if self.cfg.decode_with_target_velocity:
            next_obs_pred = self.decode(z, target_velocity)
        else:
            next_obs_pred = self.decode(z, velocity_sample)

        velocity_for_loss = velocity_sample if self.cfg.velocity_loss_use_sample else velocity_mu
        recons_loss = nn.functional.mse_loss(next_obs_pred, next_obs, reduction="none").mean(dim=-1)
        vel_loss = nn.functional.mse_loss(velocity_for_loss, target_velocity, reduction="none").mean(dim=-1)
        z_kld_loss = -0.5 * torch.sum(1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)
        vel_kld_loss = -0.5 * torch.sum(
            1.0 + velocity_logvar - velocity_mu.pow(2) - velocity_logvar.exp(), dim=-1
        )
        kld_loss = z_kld_loss + self.cfg.velocity_kld_weight * vel_kld_loss
        total_loss = recons_loss + vel_loss + beta * kld_loss
        return {
            "loss": total_loss,
            "recons_loss": recons_loss,
            "vel_loss": vel_loss,
            "kld_loss": kld_loss,
            "z_kld_loss": z_kld_loss,
            "vel_kld_loss": vel_kld_loss,
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
    ) -> CENetVAEMetrics:
        """Run one standalone VAE optimization step."""
        self.train()
        if max_grad_norm is None:
            max_grad_norm = self.cfg.max_grad_norm

        loss_dict = self.compute_loss(obs_history, next_obs, target_velocity, beta)
        valid = dones.reshape(-1) == 0 if dones is not None else torch.ones_like(loss_dict["loss"], dtype=torch.bool)

        if not valid.any():
            return {
                "loss": 0.0,
                "recons_loss": 0.0,
                "vel_loss": 0.0,
                "kld_loss": 0.0,
                "z_kld_loss": 0.0,
                "vel_kld_loss": 0.0,
                "valid_ratio": 0.0,
            }

        optimize_loss = loss_dict["loss"][valid].mean()
        metrics: CENetVAEMetrics = {
            "loss": optimize_loss.detach().item(),
            "recons_loss": loss_dict["recons_loss"][valid].mean().detach().item(),
            "vel_loss": loss_dict["vel_loss"][valid].mean().detach().item(),
            "kld_loss": loss_dict["kld_loss"][valid].mean().detach().item(),
            "z_kld_loss": loss_dict["z_kld_loss"][valid].mean().detach().item(),
            "vel_kld_loss": loss_dict["vel_kld_loss"][valid].mean().detach().item(),
            "valid_ratio": valid.float().mean().detach().item(),
        }

        optimizer.zero_grad(set_to_none=True)
        optimize_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        optimizer.step()
        return metrics
