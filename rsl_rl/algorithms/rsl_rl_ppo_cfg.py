# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from typing import Tuple


@configclass
class DreamWaQCfg:
    obs_dim: int = 3
    history_length: int = 4
    term_dims: Tuple[int, ...] = (3,)
    latent_dim: int = 16
    encoder_hidden_dims: Tuple[int, ...] = (512, 256)
    decoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    beta: float = 1.0
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    decode_with_target_velocity: bool = True
    velocity_loss_use_sample: bool = True
    
    optimizer_type: str = "adam"   # 可选 "adam", "adamw", "sgd"
    weight_decay: float = 1e-5      # 推荐给 AdamW 设置 1e-4 或 1e-5
    
@configclass
class RslRlDreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DreamWaQ"
    waq_vae_cfg: DreamWaQCfg | None = None
    
@configclass
class RslRlDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name: str = "DreamWaQRunner"


@configclass
class UnitreeA1RoughDreamWaQRunnerCfg(RslRlDreamWaQRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "a1_dreamwaq"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation=True,
        #     use_mirror_loss=True,
        #     data_augmentation_func=online_symmetry.compute_symmetric_states,
        #     mirror_loss_coeff=0.01,
        # ),
        waq_vae_cfg=DreamWaQCfg(
            obs_dim= 9,
            history_length= 4,
            term_dims=(3,3,3),
            latent_dim= 16,
            encoder_hidden_dims=(512, 256),
            decoder_hidden_dims=(512, 256, 128),
            activation="elu",
            beta=1.0,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            decode_with_target_velocity=True,
            velocity_loss_use_sample=True,
            
            optimizer_type= "adam" ,  # 可选 "adam", "adamw", "sgd"
            weight_decay= 1e-5      # 推荐给 AdamW 设置 1e-4 或 1e-5
        )
    )







@configclass
class UnitreeA1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_a1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeA1FlatPPORunnerCfg(UnitreeA1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__() # type: ignore

        self.max_iterations = 300
        self.experiment_name = "unitree_a1_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
