# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# CMoE PPO — faithful port of the official CMoE cmoe_ppo.py, adapted to the Isaac Lab rsl_rl fork.
#
# 与官方 (CMoE-main/rsl_rl/rsl_rl/algorithms/cmoe_ppo.py) 一致:
#   * 单一优化器 = Adam(actor.params + critic.params)。actor 含 experts/gating/projectors/prototypes
#     以及两个估计器;估计器在 PPO loss 路径上始终 no_grad, 因此 PPO 优化器对其为 no-op
#     (这与官方 optim.Adam(actor_critic.parameters()) 把估计器一并纳入、但实际由估计器自带 Adam
#      单独训练的行为完全相同)。
#   * 更新顺序(逐项对齐官方 update):
#       策略/值前向 -> KL 自适应 lr -> update_estimators(各自独立 Adam)
#       -> compute_contrastive_loss -> surrogate/value
#       -> zero_grad -> backward -> [multi-gpu reduce] -> clip(全部 PPO 参数) -> step
#   * loss = surrogate + value_loss_coef*value - entropy_coef*entropy + contrastive_loss_coef*contrastive
#            (官方 contrastive 无系数即系数 1.0; 这里用 contrastive_loss_coef, 配置设 1.0 即与官方一致)
#   * 门控末层含 Softmax, critic detach 复用 actor 的 gate_weights(概率)。
#   * 估计器速度目标用"下一步"base_lin_vel(Appendix A), 重建目标用"下一步"proprio。

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class CMoERolloutStorage(RolloutStorage):
    """RolloutStorage + 估计器训练所需的"下一步"目标。

    next_proprio    : o_{t+1} 的最后一帧 (β-VAE 重建目标, actor 每帧 proprio)
    next_critic_vel : s_{t+1} 的 base_lin_vel (速度估计目标 = critic 观测前 3 维)  [Appendix A]

    已知近似(2026-07-18 备注): 官方 runner L92-93 会把**终止回合**的 next 观测替换成复位前的
    terminal obs(compute_termination_observations); 这里终止步存的是复位后的新回合首帧。
    每回合只有 1 个终止步(约占样本 0.5%), 只影响估计器损失的边界样本, 不影响 PPO 本身。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.observations["actor"].shape[-1]          # 每帧 proprio 维度
        self.next_proprio = torch.zeros(self.num_transitions_per_env, self.num_envs, C, device=self.device)
        self.next_critic_vel = torch.zeros(self.num_transitions_per_env, self.num_envs, 3, device=self.device)

    def add_next_proprio(self, next_obs):
        # 在 add_transition 之后调用; step 已自增, 刚加的 transition 在 step-1
        self.next_proprio[self.step - 1].copy_(next_obs["actor"][:, -1])
        self.next_critic_vel[self.step - 1].copy_(next_obs["critic"][:, :3])

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mb = batch_size // num_mini_batches
        idx_all = torch.randperm(num_mini_batches * mb, device=self.device)
        obs = self.observations.flatten(0, 1)
        act = self.actions.flatten(0, 1)
        val = self.values.flatten(0, 1)
        ret = self.returns.flatten(0, 1)
        alp = self.actions_log_prob.flatten(0, 1)
        adv = self.advantages.flatten(0, 1)
        dp = tuple(p.flatten(0, 1) for p in self.distribution_params)
        nxt = self.next_proprio.flatten(0, 1)
        nxtv = self.next_critic_vel.flatten(0, 1)
        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                idx = idx_all[i * mb:(i + 1) * mb]
                b = RolloutStorage.Batch(
                    observations=obs[idx], actions=act[idx], values=val[idx],
                    advantages=adv[idx], returns=ret[idx], old_actions_log_prob=alp[idx],
                    old_distribution_params=tuple(p[idx] for p in dp),
                )
                b.next_proprio = nxt[idx]            # 附加字段, 与洗牌对齐
                b.next_critic_vel = nxtv[idx]        # Appendix A
                yield b


class PPOCMoE:
    """CMoE PPO algorithm (official-aligned)."""

    actor: MLPModel
    critic: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # Contrastive learning
        contrastive_loss_coef: float = 1.0,
        # Distributed training
        multi_gpu_cfg: dict | None = None,
        # 兼容字段(官方 CMoE 不支持, 若被配置传入则忽略)
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        if rnd_cfg is not None:
            print("[PPOCMoE] rnd_cfg is set but RND is not used in the CMoE port; ignoring.")
        if symmetry_cfg is not None:
            print("[PPOCMoE] symmetry_cfg is set but symmetry augmentation is not used in the CMoE port; ignoring.")
        if kwargs:
            print("[PPOCMoE] __init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))

        self.device = device

        # Multi-GPU
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Models
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        # Handles to the uncompiled modules (alias self.actor/self.critic if compilation is disabled).
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # 单一优化器 = 所有 actor + critic 参数(含估计器, 但估计器在 PPO loss 路径上 no_grad -> no-op)。
        # 与官方 optim.Adam(actor_critic.parameters()) 完全一致。
        ppo_params = list(dict.fromkeys(chain(self.actor.parameters(), self.critic.parameters())))
        self.optimizer = resolve_optimizer(optimizer)([
            {"params": ppo_params, "lr": learning_rate, "name": "ppo"},
        ])  # type: ignore

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

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
        self.contrastive_loss_coef = contrastive_loss_coef

    # ------------------------------------------------------------------ rollout
    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        # 共享门控: critic 复用 actor 刚算出的 gate_weights(已含 Softmax 的概率), detach
        self.transition.values = self.critic(obs, gate_input=self._raw_actor.gate_weights.detach()).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step (and update normalizers; no-op when obs_normalization=False)."""
        if obs["actor"].dim() == 3:
            obs_current = obs.clone()
            obs_current["actor"] = obs_current["actor"][:, -1]
        else:
            obs_current = obs
        self.actor.update_normalization(obs_current)
        self.critic.update_normalization(obs_current)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition (+ 下一步目标 for the estimators)
        self.storage.add_transition(self.transition)
        self.storage.add_next_proprio(obs)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets (GAE)."""
        st = self.storage
        critic_hidden_state = self.critic.get_hidden_state()
        with torch.no_grad():
            self.actor(obs)  # 先触发, 填 _raw_actor.gate_weights
        last_values = self.critic(obs, gate_input=self._raw_actor.gate_weights.detach()).detach()
        self.critic.reset(hidden_state=critic_hidden_state)
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    # ------------------------------------------------------------------ optimization
    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_contrastive_loss = 0.0
        mean_estimation_loss = 0.0
        mean_vae_loss = 0.0
        mean_recons_loss = 0.0
        mean_kld_loss = 0.0
        mean_terrain_loss = 0.0
        mean_explained_variance = 0.0

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            # ---- recompute policy / value for this minibatch (full fp32, no autocast) ----
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
                gate_input=self._raw_actor.gate_weights.detach(),   # critic 复用 actor 门控(官方做法)
            ).float()

            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            distribution_params = tuple(p.float() for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy

            # ---- adaptive KL learning-rate ----
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
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # ---- estimators: independent Adam (official update_estimators) ----
            history_flat = batch.observations["actor"].flatten(1)        # (B, H*C)
            next_proprio = torch.cat(
                [batch.next_proprio[:, 0:6], batch.next_proprio[:, 9:], batch.next_critic_vel], dim=-1
            )  # [angv3, grav3, q12, dq12, a12, next_vel3] = 45, 对齐官方 next_critic_obs[:, 3:48]
            # Official state_estimator uses two temporal targets:
            # velocity target = current critic_obs base_lin_vel;
            # reconstruction target = next proprio, already built
            # above with batch.next_critic_vel.
            next_vel = batch.observations["critic"][:, :3]
            map_flat = batch.observations["actor_map"][:, 2].flatten(1)  # (B, 77)
            (est_loss, vae_loss, recons_loss, kld_loss,
             _e2, terrain_loss, _r2, _k2) = self._raw_actor.update_estimators(
                history_flat, next_proprio, next_vel, map_flat, lr=self.learning_rate
            )

            # ---- contrastive loss (official compute_contrastive_loss) ----
            contrastive_loss = self._raw_actor.compute_contrastive_loss(batch.observations)

            # ---- PPO surrogate ----
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- value loss ----
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.contrastive_loss_coef * contrastive_loss
            )

            # ---- single PPO step (estimators are no-op here) ----
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]], self.max_grad_norm
            )
            self.optimizer.step()

            # ---- logging ----
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_contrastive_loss += contrastive_loss.item()
            mean_estimation_loss += est_loss
            mean_vae_loss += vae_loss
            mean_recons_loss += recons_loss
            mean_kld_loss += kld_loss
            mean_terrain_loss += terrain_loss
            with torch.inference_mode():
                ev = 1.0 - torch.var(batch.returns - values) / (torch.var(batch.returns) + 1e-8)
            mean_explained_variance += ev.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy_metric": mean_entropy / num_updates,  # 策略熵(指标, 非损失); runner 会打成 "Mean entropy_metric loss", 文档注明即可
            "contrastive": mean_contrastive_loss / num_updates,
            "estimation": mean_estimation_loss / num_updates,
            "vae": mean_vae_loss / num_updates,
            "vae_recon": mean_recons_loss / num_updates,
            "vae_kl": mean_kld_loss / num_updates,
            "terrain_ae": mean_terrain_loss / num_updates,
            "ev_metric": mean_explained_variance / num_updates,  # 价值函数解释方差(指标, 非损失)
        }
        self.storage.clear()
        return loss_dict

    # ------------------------------------------------------------------ misc
    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """估计器 / 对比头都是 actor 的子模块, 随 actor.state_dict() 一起存取。"""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor", True):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic", True):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer", True):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "PPOCMoE":
        """Construct the CMoE PPO algorithm."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))    # type: ignore
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))      # type: ignore
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))    # type: ignore

        # Resolve observation groups
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        # CMoE 无可共享的 CNN 编码器;保留该键的 pop 以兼容配置(False -> 跳过)
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = CMoERolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: "PPOCMoE" = alg_class(
            actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        # Compile if requested (建议保持 None / 不编译, 见 README)
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    # ------------------------------------------------------------------ multi-gpu
    def broadcast_parameters(self) -> None:
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        all_params = list(chain(self.actor.parameters(), self.critic.parameters()))
        grads = [p.grad.view(-1) for p in all_params if p.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in all_params:
            if p.grad is not None:
                numel = p.numel()
                p.grad.data.copy_(all_grads[offset:offset + numel].view_as(p.grad.data))
                offset += numel
