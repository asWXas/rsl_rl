# VAE 模组（可独立复用）

该目录提供一个可单独复用的 `CENetVAE` 模组，目标是把 DreamWaQ 里的 VAE/估计器部分抽离出来，方便接入其他 RL 或控制框架。

代码文件：

- `cenet_vae.py`
- `__init__.py`

---

## 1. 论文对应关系（DreamWaQ, arXiv:2301.10602v2）

本文实现对齐论文 II-C 的 CENet 思路：

- 输入：时间窗口观测 `o_t^H`。
- 输出：`z_t`（环境上下文 latent）与 `v_t`（机体速度估计）。
- 解码：用 `[z_t, v_t]` 重构 `o_{t+1}`。
- 训练目标（论文式 (5)(6)(7)）：
  - `L_CE = L_est + L_VAE`
  - `L_est = MSE(v_t_hat, v_t)`
  - `L_VAE = MSE(o_t+1_hat, o_t+1) + beta * KL(q(z_t|o_t^H) || p(z_t))`

---

## 2. 与当前 DreamWaQ 工程实现的对应


本模组默认保持与现工程一致的两个关键细节：

1. 重构分支默认用 `target_velocity`（真实速度）参与解码。  
2. 速度监督默认用采样速度（`vel_sample`）与真实速度做 MSE。  

可通过配置切换：

- `decode_with_target_velocity=False`：改成用估计速度解码。
- `velocity_loss_use_sample=False`：改成用 `vel_mu` 做监督。

---

## 3. 接口设计（预测 + 更新）

## 3.1 创建模块

```python
from DreamWaQ.modules.vae import CENetVAE, CENetVAEConfig

cfg = CENetVAEConfig(
    obs_dim=45,
    history_len=5,
    latent_dim=16,
    beta=1.0,
    learning_rate=1e-3,
)
vae = CENetVAE(cfg).to(device)
optimizer = vae.create_optimizer()
```

## 3.2 预测接口（部署或策略前向）

```python
pred = vae.predict(obs_history, deterministic=True)
z = pred["z"]               # [B, latent_dim]
vel = pred["velocity"]      # [B, 3]
```

可选重构输出：

```python
pred = vae.predict(
    obs_history,
    deterministic=False,
    decode_next_obs=True,
)
next_obs_pred = pred["next_obs_pred"]   # [B, obs_dim]
```

## 3.3 更新接口（训练）

```python
metrics = vae.update_step(
    optimizer=optimizer,
    obs_history=obs_history,       # [B, T, obs_dim]
    next_obs=next_obs,             # [B, obs_dim]
    target_velocity=base_vel,      # [B, 3]
    dones=dones,                   # [B] 或 [B,1]，可选
    beta=1.0,                      # 可选，覆盖 cfg.beta
)
```

返回：

- `loss`
- `recons_loss`
- `vel_loss`
- `kld_loss`
- `valid_ratio`（`dones==0` 的样本占比）

---

## 4. 在其他算法框架中的接入方式

最小接入流程：

1. 框架每步维护 `obs_history`（长度 `T`）。  
2. 策略前向时调用 `vae.predict(obs_history)`，把 `z`、`velocity` 拼到 actor 输入。  
3. rollout 存 `next_obs`、`target_velocity`、`dones`。  
4. 更新阶段在每个 mini-batch 调 `update_step(...)`。  

典型 actor 输入改造：

```python
pred = vae.predict(obs_history, deterministic=False)
actor_input = torch.cat([obs, pred["z"], pred["velocity"]], dim=-1)
action = actor(actor_input)
```

---

## 5. 与 PPO 联合训练建议

- 给 VAE 单独优化器（与策略优化器分离）。  
- 每个 PPO mini-batch 后做 `K` 次 VAE 子步（DreamWaQ 原实现即此策略）。  
- VAE 更新时建议过滤 `dones==1` 的样本，避免跨 episode 重构噪声。  
- `beta` 过大时 latent 更规整但可能削弱重构精度；建议从 `1.0` 开始网格调参。  

---

## 6. 张量约定

- `obs_history`: `[B, T, obs_dim]`
- `next_obs`: `[B, obs_dim]`
- `target_velocity`: `[B, 3]`
- `dones`: `[B]` 或 `[B,1]`，其中 `1` 表示终止样本

---

## 7. 迁移检查清单

接入新框架前请确认：

1. 观测已归一化（论文假设零均值单位方差更利于 `N(0, I)` prior）。  
2. `obs_history` 时间顺序一致（旧到新或新到旧必须统一）。  
3. `target_velocity` 与 `obs/next_obs` 对齐同一时间步。  
4. rollout 切分后 batch 中 `dones` 语义未反转。  
