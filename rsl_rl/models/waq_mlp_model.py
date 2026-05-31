# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState
from rsl_rl.utils import unpad_trajectories


class WaqMLPModel(MLPModel):
    """MLP model that appends DreamWaQ latent features to the actor input.

    The base observation groups are normalized exactly like :class:`MLPModel`. The extra DreamWaQ features
    ``latent_z`` and ``pred_v`` are appended after normalization and are not included in observation-normalization
    statistics.
    """

    def __init__(
        self,
        waq_input_dim: int,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the base MLP with extra DreamWaQ feature capacity."""
        self.waq_input_dim = waq_input_dim
        self.base_obs_dim = 0
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.base_obs_dim)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Return normalized base observations concatenated with DreamWaQ features."""
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        base_latent = torch.cat(obs_list, dim=-1)
        base_latent = self.obs_normalizer(base_latent)

        if "latent_z" in obs and "pred_v" in obs:
            return torch.cat([base_latent, obs["latent_z"], obs["pred_v"]], dim=-1)

        dummy_waq = torch.zeros(base_latent.shape[0], self.waq_input_dim, device=base_latent.device)
        return torch.cat([base_latent, dummy_waq], dim=-1)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass with support for padded feedforward trajectories."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Add the DreamWaQ feature width to the base observation dimension."""
        active_obs_groups, base_obs_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
        self.base_obs_dim = base_obs_dim
        return active_obs_groups, base_obs_dim + self.waq_input_dim

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly model for pre-concatenated DreamWaQ observations."""
        return _TorchWaqMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return an ONNX-friendly model for pre-concatenated DreamWaQ observations."""
        return _OnnxWaqMLPModel(self, verbose)


class _TorchWaqMLPModel(nn.Module):
    """Exportable DreamWaQ actor wrapper for TorchScript."""

    def __init__(self, model: WaqMLPModel) -> None:
        super().__init__()
        self.base_obs_dim = model.base_obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[..., : self.base_obs_dim]
        extra = x[..., self.base_obs_dim :]
        latent = torch.cat([self.obs_normalizer(base), extra], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxWaqMLPModel(nn.Module):
    """Exportable DreamWaQ actor wrapper for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: WaqMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.base_obs_dim = model.base_obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[..., : self.base_obs_dim]
        extra = x[..., self.base_obs_dim :]
        latent = torch.cat([self.obs_normalizer(base), extra], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
