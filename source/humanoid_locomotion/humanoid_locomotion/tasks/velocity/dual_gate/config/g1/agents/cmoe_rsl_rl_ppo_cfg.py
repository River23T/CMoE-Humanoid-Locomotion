# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# from humanoid_locomotion.tasks.velocity import humanoid_locomotion
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
)
from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.rl_cfg import (
    RslRlOnPolicyRunnerCfgNew,
    RslRlCMoEModelCfg,
    RslRlPpoCMoEAlgorithmCfg
)

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfgNew):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "CMoE-G1"
    obs_groups = {
        "actor": ["actor", "actor_map"],
        "critic": ["critic", "critic_map"],
    }
    # wandb_project = "velocity"
    # logger = "wandb"
    # torch_compile_mode = "max-autotune-no-cudagraphs"   # "default", "max-autotune-no-cudagraphs"
    # TODO: actor，critic，policy网络与算法修改
    actor = RslRlCMoEModelCfg(
        class_name="humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.cmoe_model:CMoEModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False, # 修改：关掉运行期归一化，靠 env 缩放（与官方一致）
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"), # log修改
        num_experts=5, # 5个专家
        expert_hidden_dims=[512,256,128], # CMoE开源代码
        gate_hidden_dims=[128],  # 门控网络
        vae_latent_dim=16,
    )
    critic = RslRlCMoEModelCfg(
        class_name="humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.models.cmoe_model:CMoEModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False, # 修改
        num_experts=5,  # 5个专家
        expert_hidden_dims=[512,256,128],  # CMoE开源代码
        gate_hidden_dims=[128],  # 门控网络
        vae_latent_dim=16,
    )
    algorithm = RslRlPpoCMoEAlgorithmCfg(
        class_name="humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.algorithms.ppo_cmoe:PPOCMoE",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01, # CMoE开源代码
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=False,
        # 对比学习
        contrastive_loss_coef=1.0,  # CMoE开源代码
    )
