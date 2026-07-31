# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# Copyright (c) 2026, River23T.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Official fixed-upper-body, 12-DOF Unitree G1 configuration for CMoE."""

from pathlib import Path

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from humanoid_locomotion.assets.robots.unitree import (
    UnitreeArticulationCfg,
    UnitreeUrdfFileCfg,
)


_ASSET_DIR = Path(__file__).resolve().parent / "g1_cmoe"
_CMOE_G1_URDF = (
    _ASSET_DIR
    / "29dof_urdf"
    / "g1_29dof_with_hand_fixed_modify_collision.urdf"
)


# Policy/SDK joint order used by the official CMoE implementation.
# Isaac Lab internally interleaves left/right joints, so task-side
# action and observation selectors must use preserve_order=True.
CMOE_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


UNITREE_G1_CMOE_12DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(_CMOE_G1_URDF),
        merge_fixed_joints=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch_yaw": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_yaw_joint",
            ],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.0,
            min_delay=0,
            max_delay=0,
        ),
        "hip_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.0,
            min_delay=0,
            max_delay=0,
        ),
        "knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=150.0,
            damping=4.0,
            armature=0.0,
            min_delay=0,
            max_delay=0,
        ),
        "ankle": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.0,
            min_delay=0,
            max_delay=0,
        ),
    },
    joint_sdk_names=CMOE_G1_JOINT_NAMES,
)
