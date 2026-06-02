# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from online_lab.tasks.manager_based.asf.mdp.symmetry import online_symmetry


@configclass
class DreamWaQCfg:
    """Configuration for the online DreamWaQ CENet VAE."""

    obs_dim: int = 45
    history_length: int = 3
    latent_dim: int = 16
    encoder_hidden_dims: tuple[int, ...] = (256, 128)
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "silu"
    beta: float = 1.0
    velocity_kld_weight: float = 0.0
    learning_rate: float = 1.0e-3
    max_grad_norm: float | None = 1.0
    decode_with_target_velocity: bool = True
    velocity_loss_use_sample: bool = True
    term_dims: tuple[int, ...] = (3, 3, 3, 12, 12, 12)
    optimizer_type: str = "adam"
    weight_decay: float = 1.0e-5
    logvar_min: float = -10.0
    logvar_max: float = 10.0


@configclass
class RslRlDreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config extended with DreamWaQ VAE training."""

    class_name: str = "DreamWaQ"
    waq_vae_cfg: DreamWaQCfg | None = None


@configclass
class RslRlDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for DreamWaQ."""

    class_name: str = "DreamWaQRunner"


@configclass
class UnitreeGo1RoughDreamWaQRunnerCfg(RslRlDreamWaQRunnerCfg):
    """DreamWaQ config for the rough Go1 locomotion task."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 1000
    experiment_name = "unitree_as2_rough_dreamwaq"
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=online_symmetry.compute_symmetric_states,
            mirror_loss_coeff=0.01,
        ),
        waq_vae_cfg=DreamWaQCfg(),
    )


@configclass
class UnitreeGo1FlatDreamWaQRunnerCfg(UnitreeGo1RoughDreamWaQRunnerCfg):
    """DreamWaQ config for the flat Go1 locomotion task."""

    def __post_init__(self) -> None:
        """Specialize the rough DreamWaQ defaults for flat-terrain training."""
        super().__post_init__()  # type: ignore

        self.max_iterations = 1500
        self.experiment_name = "unitree_go1_flat_dreamwaq"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.init_noise_std = 0.5
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.entropy_coef = 0.005
