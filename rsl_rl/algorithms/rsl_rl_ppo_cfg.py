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

    optimizer_type: str = "adam"
    weight_decay: float = 1e-5


@configclass
class RslRlDreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DreamWaQ"
    vae_cfg: DreamWaQCfg | None = None


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

        vae_cfg=DreamWaQCfg(
            obs_dim=45,
            history_length=3,
            term_dims=(3, 3, 3,12,12,12),
            latent_dim=16,
            encoder_hidden_dims=(512, 256),
            decoder_hidden_dims=(512, 256, 128),
            activation="elu",
            beta=1.0,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            optimizer_type="adam",
            weight_decay=1e-5,
        ),
    )
