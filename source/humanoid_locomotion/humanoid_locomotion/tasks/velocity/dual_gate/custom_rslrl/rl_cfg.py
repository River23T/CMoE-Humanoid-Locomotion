# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl.rl_cfg import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg
from typing import Literal

#########################
# Model configurations #
#########################


@configclass
class RslRlCNNVelocityModelCfg(RslRlCNNModelCfg):
    """Configuration for CNN model."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.cnn_velocity_model:CNNVelocityModel"
    """The model class name. Defaults to VelocityCNNModel."""



############################
# Algorithm configurations #
############################


@configclass
class RslRlPpoVelocityAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_velocity:PPOVelocity"
    """The name of the Velocity Estimator PPO algorithm. Defaults to 'VelocityEstimatorPPOAlgorithm'"""

@configclass
class RslRlPpoAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_ae:PPOAE"
    """The name of the Velocity Estimator PPO algorithm. Defaults to 'VelocityEstimatorPPOAlgorithm'"""

@configclass
class RslRlPpoVAEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_vae:PPOVAE"

@configclass
class RslRlPpoSwAVAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Velocity Estimator PPO algorithm."""

    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_swav:PPOSwAV"

#########################
# Runner configurations #
#########################


@configclass
class RslRlOnPolicyRunnerCfgNew(RslRlOnPolicyRunnerCfg):
    """Configuration of the runner for on-policy algorithms."""

    class_name: str = "OnPolicyRunner"
    """The runner class name. Defaults to OnPolicyRunner."""

    torch_compile_mode : Literal["default", "max-autotune-no-cudagraphs"] | None = None

@configclass
class RslRlCMoEModelCfg(RslRlMLPModelCfg):
    # MLP estimators + MoE + gate + contrastive
    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.cmoe_model:CMoEModel"
    num_experts: int = 5                        # 论文 N=5
    expert_hidden_dims: list[int] = [512, 256, 128]  # 官方单专家子网
    gate_hidden_dims: list[int] = [128]         # 门控网络
    vae_latent_dim: int = 16                    # z_t^H 维度


@configclass
class RslRlPpoCMoEAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """CMoE 的 PPO 算法配置(PPO + 对比学习; VAE/AE/速度估计在各自估计器内部独立训练)。"""
    class_name: str = "humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_cmoe:PPOCMoE"
    share_cnn_encoders: bool = False            # construct_algorithm 用 .pop 读取,必须是声明字段
    contrastive_loss_coef: float = 1.0          # 官方等价 1.0