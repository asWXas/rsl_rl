import torch
from collections.abc import Generator
from tensordict import TensorDict

# 从官方库显式引入原版基类
from rsl_rl.storage.rollout_storage import RolloutStorage


class WaqBatch(RolloutStorage.Batch):
    """继承原版 Batch，安全扩展 next_observations 属性"""
    def __init__(
        self,
        *args,
        next_observations: TensorDict | None = None,
        **kwargs
    ) -> None:
        # 父类初始化原有的所有 PPO 属性
        super().__init__(*args, **kwargs)
        # 绑定子类独有的未来观测
        self.next_observations: TensorDict | None = next_observations


class WaqRolloutStorage(RolloutStorage):
    """继承原版 RolloutStorage，实现数据双轨制路由"""

    class Transition(RolloutStorage.Transition):
        """继承原版 Transition，增加临时挂载槽位"""
        def __init__(self) -> None:
            super().__init__()
            self.next_observations: TensorDict | None = None

    def __init__(self, *args, next_obs_shapes: int | None = None, **kwargs) -> None:
        # 1. 引导父类完成所有的常规显存分配
        super().__init__(*args, **kwargs)
        
        # 2. [子类扩展] 动态开辟未来观测池
        if next_obs_shapes is not None:
            self.next_observations = TensorDict(
                {
                    key: torch.zeros(self.num_transitions_per_env, self.num_envs, next_obs_shapes, device=self.device) 
                    for key in self.observations.keys()
                },
                batch_size=[self.num_transitions_per_env, self.num_envs],
                device=self.device,
            )
        else:
            self.next_observations = self.observations.clone().zero_()


    def add_transition(self, transition: Transition) -> None:
        """重写落盘逻辑，将未来观测一同写入"""
        # 1. 拦截当前指针位置（因为 super().add_transition 内部会让 self.step += 1）
        current_step = self.step
        
        # 2. 调用父类，处理所有常规 PPO 数据的落盘和 step 步进
        super().add_transition(transition)
        
        # 3. [子类扩展] 将 next_obs 写回刚才的 current_step 位置
        if hasattr(transition, "next_observations") and transition.next_observations is not None:
            self.next_observations[current_step].copy_(transition.next_observations)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[WaqBatch, None, None]:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
            
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Flatten the data
        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1) 
        
        # 🚀 【新增】：把 dones 也拍扁准备切片！
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
                    observations=observations[batch_idx], # type: ignore
                    next_observations=next_observations[batch_idx], # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    
                    # 🚀 【新增】：把这一批的 dones 喂给 Batch！
                    dones=dones[batch_idx],
                )