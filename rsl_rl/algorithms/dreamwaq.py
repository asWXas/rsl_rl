from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel,WaqMLPModel
from rsl_rl.storage import WaqRolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.extensions import CENet, CENetConfig

class DreamWaQ:

    actor: WaqMLPModel
    critic: MLPModel

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
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        
        vae_cfg: dict | None = None
        
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        
        
        # DreamWaQ VAE extension (Pure Proprioception / No Vision)
        self.adaboot_prob = 0.0
        if vae_cfg is not None:
            cfg = CENetConfig(**vae_cfg)
            self.waq_vae = CENet(cfg).to(self.device)
            
            self.waq_vae_optimizer = self.waq_vae.create_optimizer()
            
            history_features_dim = cfg.obs_dim * cfg.history_length  # 历史观测堆叠的维度
            expected_action_input_dim = history_features_dim + cfg.latent_dim + 3
            actual_action_input_dim = actor.obs_dim + actor.waq_input_dim
            if actual_action_input_dim != expected_action_input_dim:
                raise ValueError(
                    f"Actor model input_dim ({actual_action_input_dim}) does not match expected "
                    f"action_input_dim ({expected_action_input_dim}). Please check the configuration."
                )
            
            self.term_dims = cfg.term_dims 
            hist_len = cfg.history_length
            self.next_frame_idx = self.get_next_frame_indices(list(self.term_dims), hist_len, self.device)
            
            print("\n" + "="*60)
            print(f"🚀 [DreamWaQ初代 Pipeline] 纯盲行架构 | VAE Latent: {cfg.latent_dim}")
            print("="*60)
            print(f" 1. 历史输入 ──► obs_history [{cfg.history_length}帧 x {cfg.obs_dim}维]")
            print("                     │")
            print("                     ▼ [ VAE 核心压缩 ]")
            print(f"    提取出 ──► 隐变量 z [{cfg.latent_dim}维] + 预测速度 v [3维]")
            print("                     │")
            print(" ────────────────────┼────────────────────")
            print("                     ▼ [ 组合输入给策略网络 (无视觉) ]")
            print("    ┌──────────────────────────────────┐")
            print(f"    │  1. 当前单帧观测 (Obs) :  {cfg.obs_dim} 维    │")
            print(f"    │  2. VAE 环境隐变量 (z) :  {cfg.latent_dim} 维    │")
            print(f"    │  3. VAE 预测速度 (v)   :  3 维     │")
            print(f"    │  4. 历史状态堆叠(Hist) :  {history_features_dim} 维   │ (用于时序控制)")
            print("    └────────────────┬─────────────────┘")
            print("                     │")
            print("                     ▼  (拼接后总计: 235 维)")
            print("          ┌──────────┴──────────┐")
            print("          ▼                     ▼")
            print("    [ Actor Model ]       [ Critic Model ]")
            print(f"    (in: {expected_action_input_dim} ──► out: 12)  (in: {self.critic.obs_dim} ──► out: 1)")
            print("="*60 + "\n")
            

        

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = WaqRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        
        # =========================================================
        # [DreamWaQ 核心] VAE 前向推理，生成 z 和 v 并注入 obs
        # =========================================================
        if hasattr(self, "waq_vae"):
            with torch.no_grad():
                group_name = self.actor.obs_groups[0]
                obs_hist = obs[group_name]

                if obs_hist.dim() == 2:
                    obs_hist_3d = obs_hist.view(
                        obs_hist.shape[0],
                        self.waq_vae.cfg.history_length,
                        self.waq_vae.cfg.obs_dim,
                    )
                else:
                    obs_hist_3d = obs_hist

                z_t, v_est = self.waq_vae.inference(obs_hist_3d)
                obs["latent_z"] = z_t

                # 真实速度：具体取哪里看你的环境
                # 常见情况：critic obs 前 3 维就是 base linear velocity
                critic_group = self.critic.obs_groups[0]
                ground_truth_vel = obs[critic_group][:, :3]

                # AdaBoot: p_boot 越大，越多使用 CENet 估计速度
                use_est = torch.rand(
                    v_est.shape[0], 1, device=v_est.device
                ) < self.adaboot_prob
                velocity_for_actor = torch.where(use_est, v_est, ground_truth_vel)
                obs["pred_v"] = velocity_for_actor
        # =========================================================
        
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        # print(f"当前观测 obs 包含的组: {self.actor.obs_groups}")

        # for group_name in self.actor.obs_groups:
        #     tensor_data = obs[group_name]
        #     print(f"👉 组名: {group_name}")
        #     print(f"   形状: {tensor_data.shape}")
        #     print(f"   数据: {tensor_data}")
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        # print(obs["latent_z"])
        # print(obs["pred_v"])
        
        
        
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        if hasattr(self, "waq_vae"):
            actor_group = self.actor.obs_groups[0]
            next_observations = TensorDict({}, batch_size=obs.batch_size, device=self.device)
            next_observations[actor_group] = obs[actor_group][:, self.next_frame_idx].clone()
            self.transition.next_observations = next_observations
            critic_group = self.critic.obs_groups[0]
            self.transition.target_velocities = obs[critic_group][:, :3].clone()

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # DreamWaQ / CENet losses
        has_waq_vae = getattr(self, "waq_vae", None) is not None
        mean_waq_vae_loss = 0.0
        mean_waq_est_loss = 0.0
        mean_waq_recons_loss = 0.0
        mean_waq_kld_loss = 0.0

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)
            # ======================== DreamWaQ / CENet update ========================
            # Keep this block before the actor forward pass: PPO must recompute log-probs
            # with the latest latent_z and pred_v features.
            waq_vae = getattr(self, "waq_vae", None)
            waq_vae_optimizer = getattr(self, "waq_vae_optimizer", None)
            if waq_vae is not None and waq_vae_optimizer is not None:
                actor_group = self.actor.obs_groups[0]
                observations = batch.observations
                next_observations = getattr(batch, "next_observations", None)
                target_velocities = getattr(batch, "target_velocities", None)
                assert observations is not None
                assert next_observations is not None
                assert target_velocities is not None

                obs_history = observations[actor_group][:original_batch_size]
                if obs_history.dim() == 2:
                    obs_history = obs_history.view(
                        obs_history.shape[0],
                        waq_vae.cfg.history_length,
                        waq_vae.cfg.obs_dim,
                    )

                next_obs_target = next_observations[actor_group][:original_batch_size]
                target_velocity = target_velocities[:original_batch_size]
                dones = batch.dones[:original_batch_size] if batch.dones is not None else None

                waq_metrics = waq_vae.update_step(
                    optimizer=waq_vae_optimizer,
                    obs_history=obs_history,
                    next_obs=next_obs_target,
                    target_velocity=target_velocity,
                    dones=dones,
                )

                mean_waq_vae_loss += waq_metrics.loss_ce
                mean_waq_est_loss += waq_metrics.loss_est
                mean_waq_recons_loss += waq_metrics.loss_recons
                mean_waq_kld_loss += waq_metrics.loss_kld

                with torch.no_grad():
                    all_obs_history = observations[actor_group]
                    if all_obs_history.dim() == 2:
                        all_obs_history = all_obs_history.view(
                            all_obs_history.shape[0],
                            waq_vae.cfg.history_length,
                            waq_vae.cfg.obs_dim,
                        )
                    latent_z, pred_v = waq_vae.evaluate(all_obs_history)
                    observations["latent_z"] = latent_z
                    observations["pred_v"] = pred_v
            # ========================================================================

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if has_waq_vae:
            mean_waq_vae_loss /= num_updates
            mean_waq_est_loss /= num_updates
            mean_waq_recons_loss /= num_updates
            mean_waq_kld_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if has_waq_vae:
            loss_dict["waq_vae"] = mean_waq_vae_loss
            loss_dict["waq_est"] = mean_waq_est_loss
            loss_dict["waq_recons"] = mean_waq_recons_loss
            loss_dict["waq_kld"] = mean_waq_kld_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        waq_vae = getattr(self, "waq_vae", None)
        if waq_vae is not None:
            waq_vae.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        waq_vae = getattr(self, "waq_vae", None)
        if waq_vae is not None:
            waq_vae.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        waq_vae = getattr(self, "waq_vae", None)
        waq_vae_optimizer = getattr(self, "waq_vae_optimizer", None)
        if waq_vae is not None:
            saved_dict["waq_vae_state_dict"] = waq_vae.state_dict()
        if waq_vae_optimizer is not None:
            saved_dict["waq_vae_optimizer_state_dict"] = waq_vae_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "waq_vae": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        waq_vae = getattr(self, "waq_vae", None)
        waq_vae_optimizer = getattr(self, "waq_vae_optimizer", None)
        if load_cfg.get("waq_vae") and waq_vae is not None and "waq_vae_state_dict" in loaded_dict:
            waq_vae.load_state_dict(loaded_dict["waq_vae_state_dict"], strict=strict)
        if (
            load_cfg.get("optimizer")
            and waq_vae_optimizer is not None
            and "waq_vae_optimizer_state_dict" in loaded_dict
        ):
            waq_vae_optimizer.load_state_dict(loaded_dict["waq_vae_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DreamWaQ:
        """Construct the DreamWaQ algorithm."""
        # Resolve class callables
        alg_class: type[DreamWaQ] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[WaqMLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        waq_vae_cfg = cfg["algorithm"].get("vae_cfg", None)
        if waq_vae_cfg is not None:
            waq_action_input_dim = waq_vae_cfg.get("latent_dim", 0) + 3
            obs_device = obs.device if obs.device is not None else device
            obs["latent_z"] = torch.zeros(env.num_envs, waq_vae_cfg.get("latent_dim", 0), device=obs_device)
            obs["pred_v"] = torch.zeros(env.num_envs, 3, device=obs_device)
        else:
            waq_action_input_dim = 0

        # Initialize the policy
        actor: WaqMLPModel = actor_class(
            waq_input_dim=waq_action_input_dim,
            obs=obs,
            obs_groups=cfg["obs_groups"],
            obs_set="actor",
            output_dim=env.num_actions,
            **cfg["actor"],
        ).to(device)
        print(f"Actor Model: {actor}")
        
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        next_obs_shapes = None
        if waq_vae_cfg is not None:
            actor_group = cfg["obs_groups"]["actor"][0]
            next_obs_shapes = {actor_group: waq_vae_cfg.get("obs_dim", 0)}
        storage = WaqRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            next_obs_shapes=next_obs_shapes,
            device=device,
        )

        # Initialize the algorithm
        alg: DreamWaQ = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def update_adaboot_prob(self, episode_rewards: torch.Tensor) -> None:
        if hasattr(self, "waq_vae"):
            self.adaboot_prob = self.waq_vae.compute_adaboot_prob(episode_rewards)

    # next_observations 切片索引计算函数
    def get_next_frame_indices(self, term_dims: list[int], history_length: int, device: str = "cuda") -> torch.Tensor:
        """
        根据 IsaacLab 的堆叠规则，计算出最新一帧在扁平张量中的索引位置。
        """
        indices = []
        current_offset = 0
        for dim in term_dims:
            chunk_size = dim * history_length
            start_idx = current_offset + chunk_size - dim
            end_idx = current_offset + chunk_size
            indices.extend(range(start_idx, end_idx))
            current_offset += chunk_size
        return torch.tensor(indices, dtype=torch.long, device=device)
