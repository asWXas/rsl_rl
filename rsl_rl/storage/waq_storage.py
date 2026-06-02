# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules import HiddenState
from rsl_rl.storage.rollout_storage import RolloutStorage


class WaqRolloutStorage(RolloutStorage):
    """Rollout storage with DreamWaQ auxiliary targets."""

    class Transition(RolloutStorage.Transition):
        """Single transition with next observation and velocity target."""

        def __init__(self) -> None:
            super().__init__()
            self.next_observations: TensorDict | None = None
            self.target_velocities: torch.Tensor | None = None

    class Batch(RolloutStorage.Batch):
        """Mini-batch with DreamWaQ auxiliary targets."""

        def __init__(
            self,
            observations: TensorDict | None = None,
            actions: torch.Tensor | None = None,
            values: torch.Tensor | None = None,
            advantages: torch.Tensor | None = None,
            returns: torch.Tensor | None = None,
            old_actions_log_prob: torch.Tensor | None = None,
            old_distribution_params: tuple[torch.Tensor, ...] | None = None,
            hidden_states: tuple[HiddenState, HiddenState] = (None, None),
            masks: torch.Tensor | None = None,
            privileged_actions: torch.Tensor | None = None,
            dones: torch.Tensor | None = None,
            next_observations: TensorDict | None = None,
            target_velocities: torch.Tensor | None = None,
        ) -> None:
            super().__init__(
                observations=observations,
                actions=actions,
                values=values,
                advantages=advantages,
                returns=returns,
                old_actions_log_prob=old_actions_log_prob,
                old_distribution_params=old_distribution_params,
                hidden_states=hidden_states,
                masks=masks,
                privileged_actions=privileged_actions,
                dones=dones,
            )
            self.next_observations = next_observations
            self.target_velocities = target_velocities

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        next_obs_shapes: dict[str, int] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        if next_obs_shapes is None:
            next_obs_tensors = {
                key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()
            }
        else:
            next_obs_tensors = {
                key: torch.zeros(num_transitions_per_env, num_envs, dim, device=device)
                for key, dim in next_obs_shapes.items()
            }
        self.next_observations = TensorDict(
            next_obs_tensors,
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.target_velocities = torch.zeros(num_transitions_per_env, num_envs, 3, device=self.device)

    def add_transition(self, transition: Transition) -> None:
        """Add one transition and its DreamWaQ auxiliary targets."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        if transition.next_observations is not None:
            self.next_observations[self.step].copy_(transition.next_observations)
        if transition.target_velocities is not None:
            self.target_velocities[self.step].copy_(transition.target_velocities)

        super().add_transition(transition)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """Yield shuffled feedforward RL mini-batches with DreamWaQ targets."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        target_velocities = self.target_velocities.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield WaqRolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    next_observations=next_observations[batch_idx],  # type: ignore
                    target_velocities=target_velocities[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
