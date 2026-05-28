# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# DualGate-Flat-G1-v0
gym.register(
    id="DualGate-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfgWithSymmetryCfg",
    },
)
gym.register(
    id="DualGate-Flat-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfgWithSymmetryCfg",
    },
)
gym.register(
    id="DualGate-Flat-G1-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.mlp_rsl_rl_ppo_cfg:G1FlatPPORunnerCfgWithSymmetryCfg",
    },
)

# DualGate-Rough-G1-v0
gym.register(
    id="DualGate-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
    },
)
gym.register(
    id="DualGate-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
    },
)
gym.register(
    id="DualGate-Rough-G1-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1RoughEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
    },
)

# DualGate-Velocity-Rough-G1-v0
gym.register(
    id="DualGate-Velocity-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1VelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Velocity-Rough-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1VelocityRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Velocity-Rough-G1-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:G1VelocityRoughEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# DualGate-Attention-G1-v0
gym.register(
    id="DualGate-Attention-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Attention-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-MHA-G1-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfg_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# DualGate-Attention-G1-v1
gym.register(
    id="DualGate-Attention-G1-v1",  # change terrain curriculum, add height on curriculum
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Attention-G1-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv1_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Attention-G1-Eval-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv1_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# DualGate-Attention-G1-v2
gym.register(
    id="DualGate-Attention-G1-v2",  # Optimization terrain curriculum
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# DualGate-Attention-G1-v3
gym.register(
    id="DualGate-Attention-G1-v3",  # Optimization terrain curriculum
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "gated_rsl_rl_cfg_entry_point": f"{agents.__name__}.gated_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Attention-G1-Play-v3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv3_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "gated_rsl_rl_cfg_entry_point": f"{agents.__name__}.gated_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)
gym.register(
    id="DualGate-Attention-G1-Eval-v3",  # Optimization terrain curriculum
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv3_EVAL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.velocity_cnn_rsl_rl_ppo_cfg:G1RoughPPORunnerCfgWithSymmetryCfg",
        "mha_rsl_rl_cfg_entry_point": f"{agents.__name__}.mha_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "gated_rsl_rl_cfg_entry_point": f"{agents.__name__}.gated_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# DualGate-Dual-G1-v0
gym.register(
    id="DualGate-Dual-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dual_env_cfg:G1DualEnvCfg",
        "ae_rsl_rl_cfg_entry_point": f"{agents.__name__}.ae_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "vae_rsl_rl_cfg_entry_point": f"{agents.__name__}.vae_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "vae_d_rsl_rl_cfg_entry_point": f"{agents.__name__}.vae_d_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "swav_rsl_rl_cfg_entry_point": f"{agents.__name__}.swav_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "swav_s_rsl_rl_cfg_entry_point": f"{agents.__name__}.swav_s_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
        "concurrent_rsl_rl_cfg_entry_point": f"{agents.__name__}.concurrent_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)

# CMoE
gym.register(
    id="DualGate-Attention-G1-v1",  # change terrain curriculum, add height on curriculum
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg:G1AttentionEnvCfgv1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.cmoe_rsl_rl_ppo_cfg:G1RoughPPORunnerCfg",
    },
)