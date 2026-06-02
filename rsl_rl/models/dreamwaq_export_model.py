# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn


class DreamWaQExportModel(nn.Module):
    """End-to-end DreamWaQ export wrapper from observation history to deterministic actions."""

    is_recurrent: bool = False

    def __init__(self, actor: nn.Module, vae: nn.Module) -> None:
        """Copy the VAE encoder and actor policy components needed for deployment."""
        super().__init__()
        self.history_dim = int(vae.cfg.obs_dim * vae.cfg.history_length)  # type: ignore[attr-defined]
        self.z_dim = int(vae.z_dim)  # type: ignore[attr-defined]
        self.vel_dim = int(vae.vel_dim)  # type: ignore[attr-defined]
        self.input_size = self.history_dim

        actor_base_obs_dim = int(actor.base_obs_dim)  # type: ignore[attr-defined]
        if actor_base_obs_dim != self.history_dim:
            raise ValueError(
                f"DreamWaQ export expects actor base_obs_dim ({actor_base_obs_dim}) to match "
                f"VAE history dimension ({self.history_dim})."
            )

        self.encoder = copy.deepcopy(vae.encoder)  # type: ignore[attr-defined]
        self.shared_head = copy.deepcopy(vae.shared_head)  # type: ignore[attr-defined]
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)  # type: ignore[attr-defined]
        self.actor_mlp = copy.deepcopy(actor.mlp)  # type: ignore[attr-defined]
        if actor.distribution is not None:  # type: ignore[attr-defined]
            self.deterministic_output = actor.distribution.as_deterministic_output_module()  # type: ignore[attr-defined]
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Run deterministic DreamWaQ inference from flattened or shaped observation history."""
        flat_history = obs_history.reshape(obs_history.shape[0], self.history_dim)
        params = self.shared_head(self.encoder(flat_history))
        z_mu = params[..., : self.z_dim]
        velocity_start = 2 * self.z_dim
        velocity_mu = params[..., velocity_start : velocity_start + self.vel_dim]
        actor_latent = torch.cat([self.obs_normalizer(flat_history), z_mu, velocity_mu], dim=-1)
        actor_output = self.actor_mlp(actor_latent)
        return self.deterministic_output(actor_output)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs_history"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for DreamWaQ feedforward exports)."""
        pass
