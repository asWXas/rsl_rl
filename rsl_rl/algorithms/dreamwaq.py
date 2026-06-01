# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


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
from rsl_rl.extensions import CENetVAE, CENetVAEConfig

class DreamWaQ:

    actor: WaqMLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

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
        # DreamWaQ extensions
        waq_vae_cfg: dict | None = None
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
        
        
        # DreamWaQ VAE extension (Pure Proprioception / No Vision)
        if waq_vae_cfg is not None:
            cfg = CENetVAEConfig(**waq_vae_cfg)
            self.waq_vae = CENetVAE(cfg).to(self.device)
            
            self.waq_vae_optimizer = self.waq_vae.create_optimizer()
            
            # 初代 DreamWaQ 的特征计算
            history_features_dim = cfg.obs_dim * cfg.history_length  # 历史观测堆叠的维度
            # 策略网络输入维度 = 原历史状态堆叠 + VAE隐变量 + VAE预测速度
            action_input_dim = history_features_dim + cfg.latent_dim + 3
            
            # 校验 action_input_dim 是否与 actor 模型的输入维度匹配
            if action_input_dim != actor.obs_dim:
                raise ValueError(f"Actor model input_dim ({actor.obs_dim}) does not match expected action_input_dim ({action_input_dim}). Please check the configuration.")
            
            
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
            print(f"    (in: {action_input_dim} ──► out: 12)  (in: {self.critic.obs_dim} ──► out: 1)")
            print("="*60 + "\n")
            
            self.term_dims = self.waq_vae.cfg.term_dims 
            hist_len = self.waq_vae.cfg.history_length
            self.latest_frame_idx = self.get_latest_frame_indices(list(self.term_dims), hist_len, self.device)
            
            

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

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
        # Record the hidden states for recurrent policies
        
        # =========================================================
        # [DreamWaQ 核心] VAE 前向推理，生成 z 和 v 并注入 obs
        # =========================================================
        if hasattr(self, "waq_vae"):
            with torch.no_grad():
                # 【修复点】：正确提取基础观测张量
                # self.actor.obs_groups 通常是 ['policy']
                # 我们取列表的第 0 个元素，即 "policy"
                group_name = self.actor.obs_groups[0] 
                obs_hist = obs[group_name]
                
                # 如果 IsaacLab 传过来的是压扁的 2D 张量 [num_envs, history_len * obs_dim]
                # 这里会通过 view 将其展开成 3D 张量 [num_envs, history_len, obs_dim]
                if obs_hist.dim() == 2:
                    obs_hist = obs_hist.view(
                        obs_hist.shape[0], # B: num_envs (并行环境数量)
                        self.waq_vae.cfg.history_length, 
                        self.waq_vae.cfg.obs_dim
                    )
                
                # 调用 VAE 预测，并行处理所有环境
                pred = self.waq_vae.predict(obs_hist, deterministic=True)
                
                # 动态注入 z [num_envs, latent_dim] 和 v [num_envs, 3] 
                obs["latent_z"] = pred["z"]
                obs["pred_v"] = pred["velocity"]
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

        # =========================================================
        # 🚀 [DreamWaQ 核心路由] 将传进来的 o_{t+1} 挂载到 transition
        # =========================================================
        group_name_a = self.actor.obs_groups[0]
        group_name_c = self.critic.obs_groups[0]
        next_obs_dict = TensorDict({}, batch_size=obs.batch_size, device=self.device)
        
        # 极简！直接且只提取 policy 组的最新一帧，舍弃 critic 等无关数据
        next_obs_dict[group_name_a] = obs[group_name_a][:, self.latest_frame_idx].clone()
        self.transition.next_observations = next_obs_dict        

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
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        # RND loss
        mean_rnd_loss = 0.0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0.0 if self.symmetry else None
        
        # 🚀 增加完整的 VAE Loss 监控记录 (已消灭 Pylance 警告)
        mean_vae_loss = 0.0
        mean_vae_recons_loss = 0.0
        mean_vae_vel_loss = 0.0
        mean_vae_kld_loss = 0.0

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        group_name = self.actor.obs_groups[0] 

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0] # type: ignore

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)
                
            # =========================================================
            # [DreamWaQ 核心] 1. VAE 更新与表征刷新 (必须放在 Actor 前向之前)
            # =========================================================
            if hasattr(self, "waq_vae"):
                # 1. 取出 actor 观测历史喂给 Encoder
                obs_hist = batch.observations[group_name][:original_batch_size]  # type: ignore

                # 如果是压扁的 2D 张量，恢复成 3D 张量 [B, T, obs_dim]
                if obs_hist.dim() == 2:
                    obs_hist = obs_hist.view(
                        obs_hist.shape[0], 
                        self.waq_vae.cfg.history_length, 
                        self.waq_vae.cfg.obs_dim
                    )

                # ---------------------------------------------------
                # 🚀 构造监督信号 (Ground Truth)
                # ---------------------------------------------------
                next_obs_target = batch.next_observations[group_name][:original_batch_size]  # type: ignore
                critic_obs = batch.observations["critic"][:original_batch_size]  # type: ignore
                target_vel = critic_obs[:, 0:3]  # type: ignore
                
                # 获取终止标志，维度对其 [B, 1] -> [B]
                b_dones = batch.dones[:original_batch_size].squeeze(-1) # type: ignore

                # --- 核心 A: 执行 VAE 的前向与反向传播 (更新 VAE 权重) ---
                vae_metrics = self.waq_vae.update_step(
                    optimizer=self.waq_vae_optimizer,
                    obs_history=obs_hist,          
                    next_obs=next_obs_target,      
                    target_velocity=target_vel,    
                    dones=b_dones,
                    beta=self.waq_vae.cfg.beta
                )

                # 记录指标
                mean_vae_loss += vae_metrics["loss"] 
                mean_vae_recons_loss += vae_metrics["recons_loss"]
                mean_vae_vel_loss += vae_metrics["vel_loss"]
                mean_vae_kld_loss += vae_metrics["kld_loss"]

                # --- 核心 B: 用刚更新好的 VAE 重新预测 z 和 v ---
                with torch.no_grad():
                    # fresh_pred = self.waq_vae.predict(obs_hist, deterministic=True)
                    fresh_pred = self.waq_vae.forward(obs_hist, deterministic=True)
                    batch.observations["latent_z"] = fresh_pred["z"]  # type: ignore
                    batch.observations["pred_v"] = fresh_pred["velocity"] # type: ignore
            # =========================================================
                
            # Recompute actions log prob and entropy for current batch of transitions
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

            # Compute KL divergence and adapt the learning rate
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
                rnd_loss.backward() # type: ignore

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
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item() # type: ignore
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # ==============================================================
        # 🚀 最终的平均计算与装载 (你之前漏掉的地方)
        # ==============================================================
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # 🚀 1. 把 VAE 的损失求平均
        if hasattr(self, "waq_vae"):
            mean_vae_loss /= num_updates
            mean_vae_recons_loss /= num_updates
            mean_vae_vel_loss /= num_updates
            mean_vae_kld_loss /= num_updates

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
            
        # 🚀 2. 把 VAE 的损失放入 loss_dict 返回给日志系统
        if hasattr(self, "waq_vae"):
            loss_dict["vae_total_loss"] = mean_vae_loss
            loss_dict["vae_recons_loss"] = mean_vae_recons_loss
            loss_dict["vae_vel_loss"] = mean_vae_vel_loss
            loss_dict["vae_kld_loss"] = mean_vae_kld_loss

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
            
        if hasattr(self, "waq_vae"):
            self.waq_vae.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
            
        if hasattr(self, "waq_vae"):
            self.waq_vae.eval()

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
        

        if hasattr(self, "waq_vae"):
            # 先拿到完整的 VAE 字典
            full_vae_sd = self.waq_vae.state_dict()
            
            # 利用字典推导式，把带有 "decoder" 的分到一边，其他的(包括 encoder, shared_head 等)分到另一边
            encoder_sd = {k: v for k, v in full_vae_sd.items() if not k.startswith("decoder.")}
            decoder_sd = {k: v for k, v in full_vae_sd.items() if k.startswith("decoder.")}
            
            # 分开存入 checkpoint！
            saved_dict["waq_vae_encoder_state_dict"] = encoder_sd
            saved_dict["waq_vae_decoder_state_dict"] = decoder_sd
            saved_dict["waq_vae_optimizer_state_dict"] = self.waq_vae_optimizer.state_dict()


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
        
        
        if load_cfg.get("waq_vae") and hasattr(self, "waq_vae"):
            # 1. 兼容加载分开保存的新格式 (Encoder + Decoder)
            if "waq_vae_encoder_state_dict" in loaded_dict and "waq_vae_decoder_state_dict" in loaded_dict:
                # 创在一个空字典，把两部分重新拼装起来
                assembled_vae_sd = {}
                assembled_vae_sd.update(loaded_dict["waq_vae_encoder_state_dict"])
                assembled_vae_sd.update(loaded_dict["waq_vae_decoder_state_dict"])
                
                self.waq_vae.load_state_dict(assembled_vae_sd, strict=strict)
                self.waq_vae_optimizer.load_state_dict(loaded_dict["waq_vae_optimizer_state_dict"])
                
            elif "waq_vae_state_dict" in loaded_dict:
                self.waq_vae.load_state_dict(loaded_dict["waq_vae_state_dict"], strict=strict)
                self.waq_vae_optimizer.load_state_dict(loaded_dict["waq_vae_optimizer_state_dict"])
                
            else:
                print("[Warning] No VAE state_dict found in the checkpoint. VAE will use random initialization.")
        
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
        cfg["actor"].pop("class_name", None)
        actor_class = WaqMLPModel  # 固定使用 WaqMLPModel 作为 Actor 模型
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

        waq_vae_cfg = cfg["algorithm"].get("waq_vae_cfg", None)
        if waq_vae_cfg is not None:
            waq_action_input_dim = waq_vae_cfg.get("latent_dim", 0) + 3
        else:
            waq_action_input_dim = 0

        # Initialize the policy
        actor: WaqMLPModel = actor_class(
            waq_action_input_dim, obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
            
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        next_obs_dim = waq_vae_cfg.get("obs_dim", 0) #type: ignore
        storage = WaqRolloutStorage(
            "rl", env.num_envs, 
            cfg["num_steps_per_env"], obs, 
            [env.num_actions], device,
            next_obs_shapes=next_obs_dim
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
            
            
        if hasattr(self, "waq_vae"):
            model_params.append(self.waq_vae.state_dict())
            
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])


        idx = 2
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[idx])
            idx += 1
            
        # 🚀 [新增] 接收并更新其他显卡传来的 VAE 权重
        if hasattr(self, "waq_vae"):
            self.waq_vae.load_state_dict(model_params[idx])



    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
            
        if hasattr(self, "waq_vae"):
            all_params = chain(all_params, self.waq_vae.parameters())
            
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

    # next_observations 切片索引计算函数
    def get_latest_frame_indices(self, term_dims: list[int], history_length: int, device: str = "cuda") -> torch.Tensor:
        """
        根据 IsaacLab 的堆叠规则，计算出最新一帧在扁平张量中的索引位置。
        """
        indices = []
        current_offset = 0
        
        for dim in term_dims:
            # 该特征在历史堆叠后占据的总长度
            chunk_size = dim * history_length
            
            # 最新一帧永远在这个 chunk 的最末尾
            start_idx = current_offset + chunk_size - dim
            end_idx = current_offset + chunk_size
            
            # 将这部分索引加入列表
            indices.extend(range(start_idx, end_idx))
            
            # 偏移量推到下一个特征块的开头
            current_offset += chunk_size
            
        return torch.tensor(indices, dtype=torch.long, device=device)