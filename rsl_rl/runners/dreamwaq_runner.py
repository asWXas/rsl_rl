from __future__ import annotations


import copy
import os
import time
import torch
import torch.nn as nn
from typing import Any

from rsl_rl.algorithms import DreamWaQ
from rsl_rl.env import VecEnv
from rsl_rl.models import WaqMLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

class DreamWaQRunner():
    alg: DreamWaQ

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[DreamWaQ] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        
        
        
        self.adaboot_cur_returns = torch.zeros(self.env.num_envs, 1, device=self.device)
        self.adaboot_return_buffer: list[torch.Tensor] = []

        

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            if hasattr(self.alg, "waq_vae") and len(self.adaboot_return_buffer) > 0:
                episode_rewards = torch.cat(self.adaboot_return_buffer)
                self.alg.update_adaboot_prob(episode_rewards)
                self.adaboot_return_buffer.clear()
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    
                    if hasattr(self.alg, "waq_vae"):
                        self.adaboot_cur_returns += rewards.view(-1, 1)

                        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                        if done_ids.numel() > 0:
                            self.adaboot_return_buffer.append(
                                self.adaboot_cur_returns[done_ids].detach().clone().flatten()
                            )
                            self.adaboot_cur_returns[done_ids] = 0.0
                    
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> WaqMLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
        
        
    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        waq_vae = getattr(self.alg, "waq_vae", None)
        if waq_vae is None:
            jit_model = self.alg.get_policy().as_jit()
        else:
            jit_model = _DreamWaQExportPolicy(self.alg.get_policy(), waq_vae, verbose=False)
        jit_model.to("cpu")
        jit_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        waq_vae = getattr(self.alg, "waq_vae", None)
        if waq_vae is None:
            onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        else:
            onnx_model = _DreamWaQExportPolicy(self.alg.get_policy(), waq_vae, verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )


class _DreamWaQExportPolicy(nn.Module):
    """Exportable CENet + actor policy without TensorDict inputs."""

    is_recurrent: bool = False

    def __init__(self, actor: WaqMLPModel, waq_vae: Any, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_dim = waq_vae.cfg.obs_dim
        self.history_length = waq_vae.cfg.history_length
        self.latent_dim = waq_vae.cfg.latent_dim
        self.input_size = self.obs_dim * self.history_length

        self.waq_encoder = copy.deepcopy(waq_vae.encoder)
        self.waq_shared_head = copy.deepcopy(waq_vae.shared_head)
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        if actor.distribution is not None:
            self.deterministic_output = actor.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Run CENet inference and deterministic actor inference."""
        obs_history = obs_history.reshape(obs_history.shape[0], self.input_size)
        waq_features = self.waq_encoder(obs_history)
        waq_head = self.waq_shared_head(waq_features)
        latent_z = waq_head[..., : self.latent_dim]
        pred_v = waq_head[..., 2 * self.latent_dim : 2 * self.latent_dim + 3]

        actor_obs = self.obs_normalizer(obs_history)
        actor_input = torch.cat([actor_obs, latent_z, pred_v], dim=-1)
        out = self.mlp(actor_input)
        return self.deterministic_output(out)

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
        """Reset recurrent export state (no-op for MLP exports)."""
        pass
