# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from tensordict import TensorDict
from typing import Any

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import CENetVAE, CENetVAEConfig, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel, WaqMLPModel
from rsl_rl.storage import WaqBatch, WaqRolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups


class DreamWaQ(PPO):
    """DreamWaQ: PPO with an online CENet-style VAE for blind locomotion."""

    actor: WaqMLPModel
    critic: MLPModel
    storage: WaqRolloutStorage

    def __init__(
        self,
        actor: WaqMLPModel,
        critic: MLPModel,
        storage: WaqRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        waq_vae_cfg: dict | object | None = None,
    ) -> None:
        """Initialize PPO components and the optional DreamWaQ VAE."""
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.waq_vae: CENetVAE | None = None
        self.waq_vae_optimizer: torch.optim.Optimizer | None = None
        self.waq_history_group = self.actor.obs_groups[0]
        self.latest_frame_idx: torch.Tensor | None = None

        if waq_vae_cfg is not None:
            cfg = CENetVAEConfig(**_config_to_dict(waq_vae_cfg))
            self.waq_vae = CENetVAE(cfg).to(self.device)
            self.waq_vae_optimizer = self.waq_vae.create_optimizer()

            expected_actor_dim = cfg.obs_dim * cfg.history_length + cfg.latent_dim + 3
            if expected_actor_dim != actor.obs_dim:
                raise ValueError(
                    f"Actor model input_dim ({actor.obs_dim}) does not match DreamWaQ input_dim "
                    f"({expected_actor_dim}). Check obs_dim, history_length, latent_dim, and obs_groups."
                )

            self.latest_frame_idx = self._make_latest_frame_indices(cfg.term_dims, cfg.history_length, cfg.obs_dim)

        self._raw_actor = self.actor
        self._raw_critic = self.critic

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions after injecting VAE latent predictions into the actor observations."""
        actor_obs = self._augment_observations(obs)
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(actor_obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and store next-observation targets for the VAE."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        if self.waq_vae is not None:
            next_obs = TensorDict({}, batch_size=obs.batch_size, device=self.device)
            next_obs[self.waq_history_group] = self._extract_latest_frame(obs[self.waq_history_group]).clone()
            self.transition.next_observations = next_obs

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def update(self) -> dict[str, float]:
        """Run DreamWaQ VAE updates followed by PPO updates."""
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("DreamWaQ currently supports feedforward actor and critic models only.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_vae_loss = 0.0
        mean_vae_recons_loss = 0.0
        mean_vae_vel_loss = 0.0
        mean_vae_kld_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]  # type: ignore

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            if self.waq_vae is not None:
                vae_metrics = self._update_vae_from_batch(batch, original_batch_size)
                mean_vae_loss += vae_metrics["loss"]
                mean_vae_recons_loss += vae_metrics["recons_loss"]
                mean_vae_vel_loss += vae_metrics["vel_loss"]
                mean_vae_kld_loss += vae_metrics["kld_loss"]
                self._inject_vae_predictions(batch.observations)  # type: ignore

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()  # type: ignore

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()  # type: ignore
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if self.waq_vae is not None:
            loss_dict["vae_total_loss"] = mean_vae_loss / num_updates
            loss_dict["vae_recons_loss"] = mean_vae_recons_loss / num_updates
            loss_dict["vae_vel_loss"] = mean_vae_vel_loss / num_updates
            loss_dict["vae_kld_loss"] = mean_vae_kld_loss / num_updates

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        """Set learnable modules to training mode."""
        super().train_mode()
        if self.waq_vae is not None:
            self.waq_vae.train()

    def eval_mode(self) -> None:
        """Set learnable modules to evaluation mode."""
        super().eval_mode()
        if self.waq_vae is not None:
            self.waq_vae.eval()

    def save(self) -> dict:
        """Return a checkpoint dictionary including the DreamWaQ VAE."""
        saved_dict = super().save()
        if self.waq_vae is not None and self.waq_vae_optimizer is not None:
            vae_state = self.waq_vae.state_dict()
            saved_dict["waq_vae_encoder_state_dict"] = {
                key: value for key, value in vae_state.items() if not key.startswith("decoder.")
            }
            saved_dict["waq_vae_decoder_state_dict"] = {
                key: value for key, value in vae_state.items() if key.startswith("decoder.")
            }
            saved_dict["waq_vae_optimizer_state_dict"] = self.waq_vae_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load PPO state and, when requested, DreamWaQ VAE state."""
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "waq_vae": True,
            }

        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg.get("waq_vae") and self.waq_vae is not None and self.waq_vae_optimizer is not None:
            if "waq_vae_encoder_state_dict" in loaded_dict and "waq_vae_decoder_state_dict" in loaded_dict:
                vae_state = {}
                vae_state.update(loaded_dict["waq_vae_encoder_state_dict"])
                vae_state.update(loaded_dict["waq_vae_decoder_state_dict"])
                self.waq_vae.load_state_dict(vae_state, strict=strict)
                self.waq_vae_optimizer.load_state_dict(loaded_dict["waq_vae_optimizer_state_dict"])
            elif "waq_vae_state_dict" in loaded_dict:
                self.waq_vae.load_state_dict(loaded_dict["waq_vae_state_dict"], strict=strict)
                self.waq_vae_optimizer.load_state_dict(loaded_dict["waq_vae_optimizer_state_dict"])
            else:
                print("[Warning] No DreamWaQ VAE state_dict found in the checkpoint.")
        return load_iteration

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast policy, value, RND, and VAE parameters across distributed workers."""
        super().broadcast_parameters()
        if self.waq_vae is not None:
            model_params = [self.waq_vae.state_dict()]
            torch.distributed.broadcast_object_list(model_params, src=0)
            self.waq_vae.load_state_dict(model_params[0])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DreamWaQ:
        """Construct the DreamWaQ algorithm from an RSL-RL training config."""
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)

        alg_class: type[DreamWaQ] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class_name = cfg["actor"].pop("class_name", "WaqMLPModel")
        if actor_class_name in ("MLPModel", "WaqMLPModel"):
            actor_class: type[WaqMLPModel] = WaqMLPModel
        else:
            actor_class = resolve_callable(actor_class_name)  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        waq_vae_cfg = cfg["algorithm"].get("waq_vae_cfg", None)
        waq_cfg_dict = _config_to_dict(waq_vae_cfg) if waq_vae_cfg is not None else None
        waq_action_input_dim = (waq_cfg_dict["latent_dim"] + 3) if waq_cfg_dict is not None else 0

        actor: WaqMLPModel = actor_class(
            waq_action_input_dim, obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")

        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        history_group = cfg["obs_groups"]["actor"][0]
        next_obs_shapes = {history_group: waq_cfg_dict["obs_dim"]} if waq_cfg_dict is not None else None
        storage = WaqRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            next_obs_shapes=next_obs_shapes,
        )

        alg: DreamWaQ = alg_class(
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def _augment_observations(self, obs: TensorDict) -> TensorDict:
        if self.waq_vae is None:
            return obs
        augmented = obs.clone()
        self._inject_vae_predictions(augmented)
        return augmented

    def _inject_vae_predictions(self, obs: TensorDict) -> None:
        if self.waq_vae is None:
            return
        with torch.no_grad():
            obs_history = self._reshape_history(obs[self.waq_history_group])
            pred = self.waq_vae.predict(obs_history, deterministic=True)
            obs["latent_z"] = pred["z"]
            obs["pred_v"] = pred["velocity"]

    def _update_vae_from_batch(self, batch: WaqBatch, original_batch_size: int) -> dict[str, float]:
        if self.waq_vae is None or self.waq_vae_optimizer is None:
            return {"loss": 0.0, "recons_loss": 0.0, "vel_loss": 0.0, "kld_loss": 0.0}
        if batch.next_observations is None:
            raise ValueError("DreamWaQ requires next_observations in WaqRolloutStorage batches.")

        obs_history = self._reshape_history(batch.observations[self.waq_history_group][:original_batch_size])  # type: ignore
        next_obs = batch.next_observations[self.waq_history_group][:original_batch_size]
        target_velocity = self._get_target_velocity(batch.observations, original_batch_size)  # type: ignore
        dones = batch.dones[:original_batch_size].reshape(-1) if batch.dones is not None else None

        self.waq_vae.train()
        loss_dict = self.waq_vae.compute_loss(obs_history, next_obs, target_velocity, beta=self.waq_vae.cfg.beta)
        valid = dones == 0 if dones is not None else torch.ones_like(loss_dict["loss"], dtype=torch.bool)
        if not valid.any():
            return {"loss": 0.0, "recons_loss": 0.0, "vel_loss": 0.0, "kld_loss": 0.0}

        optimize_loss = loss_dict["loss"][valid].mean()
        metrics = {
            "loss": optimize_loss.detach().item(),
            "recons_loss": loss_dict["recons_loss"][valid].mean().detach().item(),
            "vel_loss": loss_dict["vel_loss"][valid].mean().detach().item(),
            "kld_loss": loss_dict["kld_loss"][valid].mean().detach().item(),
        }

        self.waq_vae_optimizer.zero_grad(set_to_none=True)
        optimize_loss.backward()
        if self.is_multi_gpu:
            self._reduce_gradients(self.waq_vae.parameters())
        if self.waq_vae.cfg.max_grad_norm is not None and self.waq_vae.cfg.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.waq_vae.parameters(), self.waq_vae.cfg.max_grad_norm)
        self.waq_vae_optimizer.step()
        return metrics

    def _reshape_history(self, obs_history: torch.Tensor) -> torch.Tensor:
        if self.waq_vae is None:
            raise RuntimeError("DreamWaQ VAE is not configured.")

        cfg = self.waq_vae.cfg
        if obs_history.dim() == 3:
            return obs_history
        if obs_history.dim() != 2:
            raise ValueError(f"Expected 2D or 3D DreamWaQ history observations, got shape {obs_history.shape}.")
        expected_flat_dim = cfg.history_length * cfg.obs_dim
        if obs_history.shape[-1] != expected_flat_dim:
            raise ValueError(
                f"DreamWaQ history observation width ({obs_history.shape[-1]}) does not match "
                f"history_length * obs_dim ({expected_flat_dim})."
            )
        return obs_history.view(obs_history.shape[0], cfg.history_length, cfg.obs_dim)

    def _extract_latest_frame(self, obs_history: torch.Tensor) -> torch.Tensor:
        if self.waq_vae is None:
            raise RuntimeError("DreamWaQ VAE is not configured.")

        cfg = self.waq_vae.cfg
        if obs_history.dim() == 3:
            return obs_history[:, -1, :]
        if obs_history.dim() != 2:
            raise ValueError(f"Expected 2D or 3D DreamWaQ history observations, got shape {obs_history.shape}.")
        if obs_history.shape[-1] == cfg.obs_dim:
            return obs_history
        if self.latest_frame_idx is not None:
            return obs_history[:, self.latest_frame_idx.to(obs_history.device)]
        return obs_history[:, -cfg.obs_dim :]

    def _get_target_velocity(self, observations: TensorDict, original_batch_size: int) -> torch.Tensor:
        velocity_group = "critic" if "critic" in observations else self.critic.obs_groups[0]
        critic_obs = observations[velocity_group][:original_batch_size]
        if critic_obs.shape[-1] < 3:
            raise ValueError("DreamWaQ expects the first three critic observation values to be base velocity.")
        return critic_obs[:, :3]

    def _make_latest_frame_indices(
        self, term_dims: Iterable[int], history_length: int, obs_dim: int
    ) -> torch.Tensor | None:
        dims = [int(dim) for dim in term_dims if int(dim) > 0]
        if not dims or sum(dims) != obs_dim:
            return None

        indices: list[int] = []
        current_offset = 0
        for dim in dims:
            chunk_size = dim * history_length
            indices.extend(range(current_offset + chunk_size - dim, current_offset + chunk_size))
            current_offset += chunk_size
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _reduce_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        params = [param for param in parameters if param.grad is not None]
        if not params:
            return
        grads = [param.grad.view(-1) for param in params]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in params:
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel


def _config_to_dict(cfg: dict | object) -> dict[str, Any]:
    """Convert dict-like and configclass objects to a plain dictionary."""
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()  # type: ignore
    return dict(vars(cfg))
