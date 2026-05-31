# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class WaqBatch(RolloutStorage.Batch):
    """Rollout batch extended with next observations for DreamWaQ VAE training."""

    def __init__(self, *args: object, next_observations: TensorDict | None = None, **kwargs: object) -> None:
        """Initialize a standard rollout batch plus optional next observations."""
        super().__init__(*args, **kwargs)
        self.next_observations: TensorDict | None = next_observations


class WaqRolloutStorage(RolloutStorage):
    """Rollout storage that also records next observations for DreamWaQ."""

    class Transition(RolloutStorage.Transition):
        """Transition container extended with the next observation target."""

        def __init__(self) -> None:
            """Initialize an empty DreamWaQ transition."""
            super().__init__()
            self.next_observations: TensorDict | None = None

    def __init__(
        self,
        *args: object,
        next_obs_shapes: dict[str, int] | int | None = None,
        next_obs_keys: list[str] | tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> None:
        """Allocate regular rollout buffers and next-observation buffers."""
        super().__init__(*args, **kwargs)

        if next_obs_shapes is None:
            self.next_observations = self.observations.clone().zero_()
            return

        if isinstance(next_obs_shapes, int):
            keys = list(next_obs_keys) if next_obs_keys is not None else list(self.observations.keys())
            next_obs_shapes = {key: next_obs_shapes for key in keys}

        self.next_observations = TensorDict(
            {
                key: torch.zeros(
                    self.num_transitions_per_env,
                    self.num_envs,
                    dim,
                    dtype=self.observations[key].dtype if key in self.observations else torch.float,
                    device=self.device,
                )
                for key, dim in next_obs_shapes.items()
            },
            batch_size=[self.num_transitions_per_env, self.num_envs],
            device=self.device,
        )

    def add_transition(self, transition: Transition) -> None:
        """Store a transition and any provided next-observation targets."""
        current_step = self.step
        super().add_transition(transition)

        if transition.next_observations is None:
            return

        for key, value in transition.next_observations.items():
            if key not in self.next_observations:
                raise KeyError(f"Next observation key '{key}' was not allocated in WaqRolloutStorage.")
            self.next_observations[key][current_step].copy_(value)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[WaqBatch, None, None]:
        """Yield shuffled feedforward RL mini-batches with next observations."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield WaqBatch(
                    observations=observations[batch_idx],  # type: ignore
                    next_observations=next_observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    dones=dones[batch_idx],
                )
