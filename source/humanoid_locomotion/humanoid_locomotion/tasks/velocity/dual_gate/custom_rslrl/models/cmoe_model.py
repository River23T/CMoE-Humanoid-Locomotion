# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# CMoE actor/critic model for the Isaac Lab rsl_rl fork.
# 网络结构照搬官方 CMoE(cmoe_actor_critic.py / expert_actor_critic.py):
#   - 5 个专家 MLP(157->512->256->128->out 的官方结构, 默认 PyTorch 初始化);
#   - 门控 gating_network = Sequential(Linear, act, Linear, Softmax) 末层含 Softmax, 输出专家概率;
#   - 状态/地形估计器(β-VAE / AE)由 modules.state_estimator / terrain_estimator 提供;
#   - 对比学习 compute_contrastive_loss(gate_projector / terrain_projector / prototypes + Sinkhorn)。
# 仅在 I/O 必须适配时移植: 把官方"一整条 obs 切片"改成 Isaac Lab 的 TensorDict 分组。

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.modules import MLP
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.utils import unpad_trajectories

from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.modules.state_estimator import StateEstimator
from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.modules.terrain_estimator import TerrainEstimator


@torch.no_grad()
def _sinkhorn(out: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
    """SwAV Sinkhorn-Knopp normalization — 照搬官方 cmoe_actor_critic.sinkhorn。"""
    Q = torch.exp(out / eps).T          # (K, B)
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()
    for _ in range(iters):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T                    # (B, K)


def _get_act(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU,
            "silu": nn.SiLU, "lrelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}[name]()


class CMoEModel(MLPModel):
    """CMoE actor/critic model (official-aligned).

    Actor 'systematic observation' (157 for G1-12DOF):
        [ cur_proprio(45), velocity(3), z_H(16), map_height(77), z_E(16) ]
    Critic 'systematic observation' (128 for G1-12DOF, privileged, no estimators):
        [ critic_proprio(51), critic_map_height(77) ]
    门控在 actor 端算一次(已含 Softmax 的概率), critic 端 detach 复用。
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        num_experts: int = 5,
        expert_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        gate_hidden_dims: tuple[int, ...] | list[int] = (128,),
        vae_latent_dim: int = 16,
        # contrastive hyper-parameters (official defaults; NOT read from the cfg)
        num_prototypes: int = 32,
        contrastive_temperature: float = 0.2,
        contrastive_proj_dim: int = 16,
        sinkhorn_iters: int = 3,
        sinkhorn_eps: float = 0.05,
        # accepted-but-ignored (so a leftover CNN-style cfg can never break construction)
        cnn_cfg=None,
        cnns=None,
    ) -> None:
        self._is_actor = obs_set == "actor"
        self.vae_z_dim = vae_latent_dim  # consumed by _get_latent_dim (must be set before super().__init__)

        # actor obs is (B, H, C); critic obs is (B, D). Take the last frame for the parent MLP setup.
        if obs[obs_set].dim() == 3:
            self._hist_len = obs[obs_set].shape[1]
            self._frame_dim = obs[obs_set].shape[2]
            obs_current = obs.clone()
            obs_current[obs_set] = obs_current[obs_set][:, -1]
        else:
            self._hist_len = 1
            self._frame_dim = obs[obs_set].shape[-1]
            obs_current = obs

        # parent sets: self.obs_groups / self.obs_dim / self.obs_normalizer / dead self.mlp / self.distribution
        super().__init__(
            obs_current, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        system_dim = self._get_latent_dim()  # 157 (actor) / 128 (critic)
        self.num_experts = num_experts
        # 单一(状态无关)std 在 distribution 里 -> 专家只输出均值(actor) / 值(critic)
        expert_out_dim = self.distribution.input_dim if self.distribution is not None else output_dim

        # ----- experts (both actor & critic) -----
        # 照搬官方: 专家用默认 PyTorch 初始化(官方 ExpertActorCritic.init_weights 定义了但 "not used")
        self.experts = nn.ModuleList(
            [MLP(system_dim, expert_out_dim, expert_hidden_dims, activation) for _ in range(self.num_experts)]
        )

        if self._is_actor:
            # ----- estimators (actor only; each owns an independent Adam) -----
            self.state_estimator = StateEstimator(
                prop_input_dim=self._hist_len * self._frame_dim,  # 10*45 = 450
                num_one_step_obs=self._frame_dim,                 # 45
                latent_dim=vae_latent_dim,
                explicit_dim=3,
                activation=activation,
                use_estimation_loss=True,
                use_latent_loss=True,
            )
            self.terrain_estimator = TerrainEstimator(
                map_dim=self.map_flat_dim,                        # 77
                latent_dim=vae_latent_dim,
                activation=activation,
                use_estimation_loss=True,
                use_latent_loss=True,
            )
            # ----- gating network (actor only) -----
            # 照搬官方 gating_network: Sequential(Linear, act, Linear, Softmax) -> 末层 Softmax, 输出概率
            gate_layers = []
            gin = system_dim
            for h in gate_hidden_dims:
                gate_layers += [nn.Linear(gin, h), _get_act(activation)]
                gin = h
            gate_layers += [nn.Linear(gin, self.num_experts), nn.Softmax(dim=-1)]
            self.gating_network = nn.Sequential(*gate_layers)
            # ----- contrastive head (actor only) -----
            self.gate_projector = MLP(self.num_experts, contrastive_proj_dim, (128, 64), activation)
            self.terrain_projector = MLP(
                self.map_flat_dim + vae_latent_dim, contrastive_proj_dim, (128, 64), activation
            )
            self.prototypes = nn.Embedding(num_prototypes, contrastive_proj_dim)
            self.contrastive_temperature = contrastive_temperature
            self.sinkhorn_iters = sinkhorn_iters
            self.sinkhorn_eps = sinkhorn_eps

        # caches
        self.gate_weights = None  # actor: set in forward (Softmax probs); critic reuses it via gate_input
        self._cur_map_flat = None
        self._cur_z_E = None

    # ------------------------------------------------------------------ system obs
    def get_latent(self, obs, masks=None, hidden_state=None):
        map_group = self.obs_groups_2d[0]
        if self._is_actor:
            history = obs["actor"]
            if history.dim() == 3:
                history_flat = history.flatten(1)   # (B, H*C)
                cur = history[:, -1]                # (B, C)
            else:
                history_flat, cur = history, history
            cur = self.obs_normalizer(cur)          # Identity when obs_normalization=False -> raw env-scaled
            map_flat = obs[map_group][:, 2].flatten(1)  # height channel -> (B, H*W)
            with torch.no_grad():
                vel, z_H = self.state_estimator(history_flat)
                z_E = self.terrain_estimator(map_flat)
            self._cur_map_flat = map_flat
            self._cur_z_E = z_E
            return torch.cat([cur, vel, z_H, map_flat, z_E], dim=-1)   # (B, 157)
        # critic
        critic = self.obs_normalizer(obs[self.obs_groups[0]])         # (B, 51)
        map_flat = obs[map_group][:, 2].flatten(1)                    # (B, 77)
        return torch.cat([critic, map_flat], dim=-1)                  # (B, 128)

    # ------------------------------------------------------------------ MoE forward
    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False, gate_input=None):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        system_obs = self.get_latent(obs, masks, hidden_state)
        if gate_input is not None:
            weights = gate_input                       # critic: 复用 actor 的 gate_weights(Softmax 概率)
        else:
            weights = self.gating_network(system_obs)  # actor: gating_network 末层含 Softmax -> 概率
        self.gate_weights = weights
        out_each = torch.stack([e(system_obs) for e in self.experts], dim=1)   # (B, E, out)
        moe_output = (weights.unsqueeze(-1) * out_each).sum(dim=1)             # (B, out)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(moe_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(moe_output)
        return moe_output

    # ------------------------------------------------------------------ contrastive
    def compute_contrastive_loss(self, obs):
        """照搬官方 CMoE.compute_contrastive_loss(梯度只进 gating_network / 投影头 / 原型)。"""
        system_obs = self.get_latent(obs)                              # refreshes _cur_map_flat / _cur_z_E
        gate_probs = self.gating_network(system_obs.detach())          # 已含 Softmax
        height_input = torch.cat([self._cur_map_flat.detach(), self._cur_z_E.detach()], dim=-1)
        g_z = F.normalize(self.gate_projector(gate_probs), dim=-1, p=2)
        h_z = F.normalize(self.terrain_projector(height_input), dim=-1, p=2)
        with torch.no_grad():
            w = F.normalize(self.prototypes.weight.data.clone(), dim=-1, p=2)
            self.prototypes.weight.copy_(w)
        score_s = g_z @ self.prototypes.weight.T
        score_t = h_z @ self.prototypes.weight.T
        with torch.no_grad():
            q_s = _sinkhorn(score_s, self.sinkhorn_eps, self.sinkhorn_iters)
            q_t = _sinkhorn(score_t, self.sinkhorn_eps, self.sinkhorn_iters)
        log_p_s = F.log_softmax(score_s / self.contrastive_temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.contrastive_temperature, dim=-1)
        return -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()

    # ------------------------------------------------------------------ estimator update (mirrors official update_estimators)
    def update_estimators(self, obs_history, next_proprio, next_vel, map_flat, lr):
        """对齐官方 CMoEActorCritic.update_estimators: 调两个估计器各自的 update, 返回 8 个标量。
        next_vel  : 下一步 base_lin_vel (速度估计目标, Appendix A)
        next_proprio: 下一步 proprio 帧 (β-VAE 重建目标)
        map_flat  : 当前高程图展平 (地形 AE 自重建目标)
        """
        e, l, r, k = self.state_estimator.update(obs_history, next_vel, next_proprio, lr)
        e2, l2, r2, k2 = self.terrain_estimator.update(map_flat, lr)
        return e, l, r, k, e2, l2, r2, k2

    # ------------------------------------------------------------------ obs dims
    def _get_obs_dim(self, obs, obs_groups, obs_set):
        active = obs_groups[obs_set]
        obs_dim_1d = 0
        groups_1d, groups_2d, dims_2d = [], [], []
        for g in active:
            shape = obs[g].shape
            if len(shape) == 4:        # (B, C, H, W) elevation map
                groups_2d.append(g)
                dims_2d.append((shape[2], shape[3]))
            elif len(shape) == 2:      # (B, D) 1D
                groups_1d.append(g)
                obs_dim_1d += shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for group '{g}': {tuple(shape)}")
        if not groups_2d:
            raise ValueError("CMoEModel expects exactly one elevation-map (4D) observation group.")
        self.obs_groups_2d = groups_2d
        self.obs_dims_2d = dims_2d
        self.map_flat_dim = dims_2d[0][0] * dims_2d[0][1]   # 7*11 = 77
        return groups_1d, obs_dim_1d

    def _get_latent_dim(self):
        if self.obs_groups[0] == "actor":
            # cur_proprio + velocity(3) + z_H + map + z_E
            return self.obs_dim + 3 + self.vae_z_dim + self.map_flat_dim + self.vae_z_dim
        return self.obs_dim + self.map_flat_dim

    def as_jit(self):
        """Return a TorchScript-scriptable copy of the *actor* inference path.

        rsl_rl 的 export_policy_to_jit() 用 torch.jit.script(不是 trace)编译本模块的
        forward 源码, 所以 wrapper 只能用"纯 tensor + 可脚本化算子"(不能出现 TensorDict /
        dict 取键 / **kwargs / optimizer)。这里把 actor 在 eval 期的前向数学完全复刻一遍:
            cur = history[:, -1]  # 最后一帧 proprio (B, 45)
            vel, z_H = state_estimator(history.flatten)    # eval: z=mu, 确定性
            z_E      = terrain_estimator(map_flat)         # eval: z=mu
            sys      = cat([cur, vel, z_H, map_flat, z_E]) # (B, 157)
            w        = softmax_gate(sys)                   # gating_network 末层已含 Softmax
            out      = Σ w_i · expert_i(sys)
            action   = deterministic_output(out)           # 分布的确定性均值
        critic 端没有专家外的推理需求, 因此只导出 actor。
        """
        if not self._is_actor:
            raise RuntimeError("as_jit() should be called on the actor CMoEModel, not the critic.")
        return _TorchCMoEModel(self)

    def as_onnx(self, verbose: bool = False):
        """Return an ONNX-export wrapper around the *actor* inference path."""
        if not self._is_actor:
            raise RuntimeError("as_onnx() should be called on the actor CMoEModel, not the critic.")
        return _OnnxCMoEModel(self, verbose)


class _TorchCMoEModel(nn.Module):
    """TorchScript-friendly copy of the CMoE actor (deterministic inference).

    输入(部署时机器人侧提供):
        history : (B, H, C) proprio 历史帧, H=10, C=45 (与训练 obs['actor'] 完全一致的缩放/顺序)
        map : (B, 3, 7, 11)  高程图, 取 channel 2 为高度 (与训练 obs['actor_map'] 一致)
    输出:
        action  : (B, 12) 确定性动作均值
    """

    def __init__(self, model: "CMoEModel") -> None:
        super().__init__()
        import copy

        # obs 归一化(本项目为 Identity, 但仍照搬以防将来打开)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)

        # ---- state estimator: 只取 eval 需要的三段(丢掉 optimizer / decoder / fc_var)----
        self.se_encoder = copy.deepcopy(model.state_estimator.encoder)     # Sequential
        self.se_fc_mu = copy.deepcopy(model.state_estimator.fc_mu)         # Linear -> z_H (16)
        self.se_fc_explicit = copy.deepcopy(model.state_estimator.fc_explicit)  # Linear -> vel (3)

        # ---- terrain estimator: 只取 encoder + fc_mu ----
        self.te_encoder = copy.deepcopy(model.terrain_estimator.encoder)   # Sequential
        self.te_fc_mu = copy.deepcopy(model.terrain_estimator.fc_mu)       # Linear -> z_E (16)

        # ---- gating + experts(照搬, 均可脚本化)----
        self.gating_network = copy.deepcopy(model.gating_network)          # 末层含 Softmax
        self.experts = nn.ModuleList([copy.deepcopy(e) for e in model.experts])

        # ---- 分布的确定性输出模块(与 sibling 模型一致的做法)----
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # 高度图 channel 索引(训练里用的是 obs[map][:, 2]); 存成常量供 script 使用
        self.map_height_channel: int = 2

    def forward(self, history: torch.Tensor, elevation_map: torch.Tensor) -> torch.Tensor:
        # 最后一帧 proprio(与训练 get_latent 的 cur = history[:, -1] 对齐)
        cur = history[:, -1]
        cur = self.obs_normalizer(cur)

        # proprio 历史展平(与训练 history.flatten(1) 对齐)
        history_flat = history.flatten(1)
        # state estimator eval 前向: z = mu, explicit = vel(确定性, 无重参数化)
        se_h = self.se_encoder(history_flat)
        vel = self.se_fc_explicit(se_h)
        z_H = self.se_fc_mu(se_h)

        # 高度通道展平(与训练 obs[map][:, 2].flatten(1) 对齐)-> (B, 77)
        map_flat = elevation_map[:, self.map_height_channel].flatten(1)
        te_h = self.te_encoder(map_flat)
        z_E = self.te_fc_mu(te_h)

        # 系统观测(顺序必须与训练 get_latent 完全一致)
        system_obs = torch.cat([cur, vel, z_H, map_flat, z_E], dim=-1)

        # 门控(已含 Softmax)+ 专家加权和
        # 注意: TorchScript 不支持用循环变量索引 nn.ModuleList —— 原写法
        # `self.experts[i]` 在 torch.jit.script 阶段必然报错:
        #   "Expected integer literal for index. ModuleList/Sequential indexing is only
        #    supported with integer literals. ... Enumeration is supported"
        # 这正是 play.py 里 runner.export_policy_to_jit() 的崩溃点。改为
        # "遍历 -> stack -> 加权求和", 与 eager 训练前向 (CMoEModel.forward 中
        # torch.stack + weighted sum) 的算子顺序完全一致, 导出结果与在线策略逐位相同。
        weights = self.gating_network(system_obs)
        expert_outs: List[torch.Tensor] = []
        for expert in self.experts:
            expert_outs.append(expert(system_obs))
        out_each = torch.stack(expert_outs, dim=1)            # (B, E, out)
        out = (weights.unsqueeze(-1) * out_each).sum(dim=1)   # (B, out)

        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """No recurrent state to reset (kept for API parity with sibling exports)."""
        pass


class _OnnxCMoEModel(nn.Module):
    """ONNX-export wrapper around the CMoE actor (deterministic inference)."""

    def __init__(self, model: "CMoEModel", verbose: bool) -> None:
        super().__init__()
        import copy

        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.se_encoder = copy.deepcopy(model.state_estimator.encoder)
        self.se_fc_mu = copy.deepcopy(model.state_estimator.fc_mu)
        self.se_fc_explicit = copy.deepcopy(model.state_estimator.fc_explicit)
        self.te_encoder = copy.deepcopy(model.terrain_estimator.encoder)
        self.te_fc_mu = copy.deepcopy(model.terrain_estimator.fc_mu)
        self.gating_network = copy.deepcopy(model.gating_network)
        self.experts = nn.ModuleList([copy.deepcopy(e) for e in model.experts])
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.map_height_channel = 2

        # 记录 dummy 形状(用于 torch.onnx.export 的 tracing)
        self._hist_len = model._hist_len          # 10
        self._frame_dim = model._frame_dim        # 45
        self._map_h = model.obs_dims_2d[0][0]     # 7
        self._map_w = model.obs_dims_2d[0][1]     # 11

    def forward(self, history: torch.Tensor, elevation_map: torch.Tensor) -> torch.Tensor:
        cur = self.obs_normalizer(history[:, -1])
        history_flat = history.flatten(1)
        se_h = self.se_encoder(history_flat)
        vel = self.se_fc_explicit(se_h)
        z_H = self.se_fc_mu(se_h)
        map_flat = elevation_map[:, self.map_height_channel].flatten(1)
        z_E = self.te_fc_mu(self.te_encoder(map_flat))
        system_obs = torch.cat([cur, vel, z_H, map_flat, z_E], dim=-1)
        weights = self.gating_network(system_obs)
        expert_out = torch.stack([e(system_obs) for e in self.experts], dim=1)
        out = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_hist = torch.zeros(1, self._hist_len, self._frame_dim)
        dummy_map = torch.zeros(1, 3, self._map_h, self._map_w)
        return (dummy_hist, dummy_map)

    @property
    def input_names(self) -> list[str]:
        return ["proprio_history", "elevation_map"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
