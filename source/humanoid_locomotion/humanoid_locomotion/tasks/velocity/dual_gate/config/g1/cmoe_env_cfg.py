# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

import math

import torch

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from humanoid_locomotion.tasks.velocity.dual_gate import mdp
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.cmoe import (
    CMOE_TERRAINS_CFG,
    CMOE_EASY_TERRAIN_NAMES,
)

from .rough_env_cfg import (
    G1VelocityRoughEnvCfg,
    RobotSceneCfg,
    RecorderManagerCfg,
)


# ---------------------------------------------------------------------------- #
#  2026-07-17 结构性对齐: 官方 CMoE 是 12 自由度纯腿策略。
#  - 官方 g1_cmoe_config.py 第15行: num_actions = 12;
#    第11行: num_one_step_observations = 45 = 角速度3 + 重力3 + 指令3 + q12 + qd12 + a12
#  - 官方 URDF g1_29dof_with_hand_fixed_modify_collision.urdf 实测:
#    12 个 revolute(双腿) + 50 个 fixed —— 腰 3 关节与双臂 14 关节全部焊死
#  - 官方 control.stiffness 字典只有 hip/knee/ankle, 根本没有上身增益
#  之前的移植把 29 个关节全部塞进了动作与观测空间(actor 单帧 96 维), 让策略在
#  没有任何上身姿态奖励的情况下额外控制 17 个自由关节 —— 这是本轮 1 秒即倒、
#  std 塌缩到 0.31、terrain level 钉死 0.06 的根因。以下常量把动作/观测/奖励/
#  域随机化全部限定到官方同款 12 个腿关节; 其余 17 关节由资产里已有的执行器
#  以默认角位置伺服(waist_yaw 200/5, 肩肘腕 40/1), 近似官方的刚性焊死。
# ---------------------------------------------------------------------------- #
LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]


def reset_joints_by_scale_selected(
    env,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """官方 humanoid.py::_reset_dofs 的关节子集版。

    官方语义: 它的仿真里只有 12 个腿关节, 复位时 q0 x U(0.5, 1.5)、速度 0。
    Isaac Lab 自带的 mdp.reset_joints_by_scale 会把缩放施加到**全部 29 关节**
    (它不读 asset_cfg.joint_ids), 导致被锁定的上身每回合从 0.5~1.5 倍默认角
    出发再被伺服拉回, 引入官方不存在的开局扰动。本函数只缩放 joint_ids 指定的
    腿关节, 其余关节精确落在默认角、零速度 —— 等价于官方"上身焊死"的复位。
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    ids = asset_cfg.joint_ids
    sel = joint_pos[:, ids]
    scale = torch.empty_like(sel).uniform_(*position_range)
    joint_pos[:, ids] = sel * scale
    if velocity_range != (0.0, 0.0):
        joint_vel[:, ids] = torch.empty_like(sel).uniform_(*velocity_range)

    # 与官方一致: 复位角夹在软限位内, 防止 x1.5 越过 URDF 限位
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


# ---------------------------------------------------------------------------- #
#  Scene: 追加 4 个脚部 RayCaster + 官方语义的 CMoE 高度扫描器
#  (feet_ground_parallel 用官方 5 采样点 pattern; feet_edge 用 3x3@0.05m 小栅格;
#   cmoe_height_scanner: 世界系对齐 + 每点 3 射线倒角, 见 mdp/cmoe_observations.py)
# ---------------------------------------------------------------------------- #
@configclass
class CMoESceneCfg(RobotSceneCfg):
    left_foot_sample = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=mdp.FootSamplePatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
    )
    right_foot_sample = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=mdp.FootSamplePatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
    )
    left_foot_edge = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),  # 3x3, 覆盖官方边缘膨胀 0.05m
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
    )
    right_foot_edge = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
    )
    # 官方 _get_heights 的 Isaac Lab 版:
    #  - ray_alignment="world": 官方采样网格只随机体平移, 不随 yaw 旋转
    #    (noisy_base_quat 仅是每回合 U(±0.2rad) 的固定噪声 yaw, 不含机体朝向)
    #  - ChamferGridPatternCfg: 每个网格点 3 条射线 (x±0.05), 复刻官方 3 点倒角平均
    #  - drift_range: 近似官方每回合 xy 平移噪声 N(0, 0.05)
    cmoe_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
        ray_alignment="world",
        pattern_cfg=mdp.ChamferGridPatternCfg(resolution=0.1, size=(1.0, 0.6)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        history_length=0,
        drift_range=(-0.05, 0.05),
    )


# ---------------------------------------------------------------------------- #
#  Commands: 官方 G1CMoECfg.commands (easy/hard 双档, heading 模式, 10s 重采样)
# ---------------------------------------------------------------------------- #
@configclass
class CMoECommandsCfg:
    base_velocity = mdp.CMoEVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),          # 官方 resampling_time = 10
        rel_standing_envs=0.0,                       # 官方无 standing 采样 (小指令置零由 min_command_norm 实现)
        rel_heading_envs=1.0,
        heading_command=True,                        # 官方 heading_command=True
        heading_control_stiffness=0.5,               # 官方 0.5*wrap_to_pi(...)
        # ranges.ang_vel_z 同时是 heading->yaw 率的 clip 区间: 官方 clip(-1, 1)
        ranges=mdp.CMoEVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0), heading=(-1.6, 1.6)
        ),
        easy_ranges=mdp.CMoEVelocityCommandCfg.Ranges(   # 官方 easy_terrain_ranges
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0), heading=(-1.6, 1.6)
        ),
        hard_ranges=mdp.CMoEVelocityCommandCfg.Ranges(   # 官方 hard_terrain_ranges
            lin_vel_x=(0.3, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
        easy_terrain_names=CMOE_EASY_TERRAIN_NAMES,
        min_command_norm=0.2,
        debug_vis=True,
    )


# ---------------------------------------------------------------------------- #
#  Rewards: 官方 G1CMoECfg.rewards.scales 全表 (论文 Table II), 权重逐项一致。
#  两个框架都对奖励乘 dt(0.02s), 因此权重可 1:1 照抄。
# ---------------------------------------------------------------------------- #
@configclass
class CMoERewardsCfg:
    # -- 任务项
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,               # 官方 base 系速度跟踪 exp(-err/0.25)
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    tracking_yaw = RewTerm(                          # 官方跟踪 heading 角, 非 yaw 角速度
        func=mdp.track_heading_exp, weight=2.0, params={"command_name": "base_velocity"}
    )
    # -- 机身
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    base_height = RewTerm(
        func=mdp.base_height_l2_cmoe,
        weight=-15.0,
        params={
            "target_height": 0.75,                   # 官方 base_height_target=0.75
            # 显式传 SceneEntityCfg: manager 只解析 params 里的, 写在函数默认值里的不会被解析
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    # -- 足部
    feet_stumble = RewTerm(
        func=mdp.feet_stumble_cmoe,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-15.0,                                # 官方 collision = -15 (hip/knee 触地即罚)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip_.*_link", ".*_knee_link"]),
            "threshold": 0.1,
        },
    )
    feet_lateral_distance = RewTerm(
        func=mdp.feet_lateral_distance, weight=0.8, params={"min_dist": 0.18, "max_dist": 0.24}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_cmoe,
        weight=1.0,
        params={
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    feet_ground_parallel = RewTerm(func=mdp.feet_ground_parallel, weight=-0.02)
    # -- 关节
    hip_dof_error = RewTerm(
        func=mdp.joint_deviation_sq,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # 2026-07-17: 以下 6 项全部限定 12 腿关节 —— 官方仿真中只有这 12 个自由度,
    # 它的 sum over all joints 天然只覆盖双腿; 被位置伺服锁定的上身关节会持续
    # 产生握持力矩/微小速度, 不限定会往这些惩罚里注入官方不存在的常量偏置。
    dof_acc = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    torques = RewTerm(
        func=mdp.joint_torques_l2, weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3)  # 动作管理器已是 12 维, 无需限定
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_soft, weight=-2.0,
        params={"soft_ratio": 0.9, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-1.0,
        params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    torque_limits = RewTerm(
        func=mdp.joint_torque_limits, weight=-1.0,
        params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    # -- 地形
    feet_edge = RewTerm(
        func=mdp.feet_edge_cmoe,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )


# ---------------------------------------------------------------------------- #
#  Terminations: 官方 check_termination (pelvis 触地 / |roll|>1 / |pitch|>1 / gap 掉落)
# ---------------------------------------------------------------------------- #
@configclass
class CMoETerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0},
    )
    bad_roll_pitch = DoneTerm(func=mdp.bad_roll_pitch, params={"limit_angle": 1.0})
    gap_fall = DoneTerm(
        func=mdp.root_height_below_on_terrain_types,
        params={"minimum_height": 0.5, "terrain_names": ["parkour_gap"]},
    )


# ---------------------------------------------------------------------------- #
#  Curriculum: 官方地形课程 (0.8 / 0.4 * cmd_x * episode_length)。
#  官方另有指令课程 update_command_curriculum, 但 G1CMoECfg.commands.curriculum=False,
#  即发布版训练**未启用** —— 如需启用, 取消下面 command_levels 的注释即可
#  (机制: easy 与 hard 组的 tracking_lin_vel 每步均值都 > 0.7*权重时,
#   两档 lin_vel_x 上限各 +0.1, 上限 easy 3.0 / hard 1.0)。
# ---------------------------------------------------------------------------- #
@configclass
class CMoECurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.cmoe_terrain_levels)
    # command_levels = CurrTerm(
    #     func=mdp.cmoe_command_levels,
    #     params={"command_name": "base_velocity", "reward_term": "tracking_lin_vel",
    #             "success_ratio": 0.7, "increment": 0.1, "max_easy": 3.0, "max_hard": 1.0},
    # )


# ---------------------------------------------------------------------------- #
#  Events: 官方 domain_rand 全套 (legged_robot_config.LeggedRobotCfg.domain_rand)
#  独立于你原 EventCfg, 数值逐项对齐官方; 官方没有的项(armature 随机)不出现。
# ---------------------------------------------------------------------------- #
@configclass
class CMoEEventCfg:
    # kp/kd x U(0.9, 1.1)  (官方 randomize_kp / randomize_kd)
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            # 2026-07-17: ".*" -> 腿关节。官方 randomize_kp/kd 作用于它仅有的 12 个
            # 腿自由度; 上身握持增益保持确定值, 让"焊死"近似更稳定。
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "operation": "scale",
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "distribution": "uniform",
        },
    )
    # 基座质量 + U(-1, 2) kg  (官方 randomize_payload_mass, 作用在 root=pelvis)
    payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    # 基座质心偏移 U(±0.05)^3  (官方 randomize_com_displacement)
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    # 摩擦 U(0, 1), 恢复系数不随机 (官方 randomize_friction / restitution=False)
    # 2026-07-18 修复: dynamic_friction_range (0,1) -> (1,1)。
    # 官方 IsaacGym 的 RigidShapeProperties.friction 是**单值**, 静/动摩擦同一个数;
    # 而 isaaclab 的 randomize_rigid_body_material 对静/动各自独立采样, 再由
    # make_consistent=True 取 dynamic = min(static, dynamic) —— 两次独立 U(0,1) 取 min
    # 的均值只有 0.33, 36% 回合动摩擦 < 0.2(官方从不低于 0.4), 脚一打滑就雪上加霜。
    # 把 dynamic 采样固定为 1.0 后, min(static, 1.0) = static, 即动==静单值, 与官方
    # 语义逐位一致; 配合 __post_init__ 里地形 0.8+average, 有效动摩擦同样落在 U(0.4,0.9)。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    # 官方 humanoid.py::_reset_root_states: xy ± 0.3, 无 yaw 随机, 根 6 维速度 U(±0.5)。
    # 注意: legged_gym 基类是 ±1.0, 但 CMoE 在 humanoid.py 里覆写成 ±0.3(其代码注释
    # 仍写 "within 1m", 以实现为准)。几何上必须用 ±0.3: parkour_gap 出生平台只有
    # x∈[0,1.0m), mix 的台阶从 x=1.5m 起、走廊 ±1.0m, 用 ±1.0 会有相当比例出生点
    # 直接踩沟沿/悬空。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    # 官方 _reset_dofs: q0 x U(0.5, 1.5), 速度 0 —— 只作用于它的 12 个腿关节。
    # 2026-07-17: 换成本文件顶部的 reset_joints_by_scale_selected: 腿缩放,
    # 上身 17 关节每回合精确回默认角(Isaac Lab 自带版会缩放全部 29 关节)。
    reset_robot_joints = EventTerm(
        func=reset_joints_by_scale_selected,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    # 官方 _push_robots: 每 16s 把基座 xy 速度替换为 U(-1, 1) m/s
    push_robot = EventTerm(
        func=mdp.push_by_replacing_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),               # 官方 push_interval_s = 16
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},  # 官方 max_push_vel_xy = 1
    )


# ---------------------------------------------------------------------------- #
#  环境
# ---------------------------------------------------------------------------- #
@configclass
class G1CMoEEnvCfg(G1VelocityRoughEnvCfg):
    """G1 + CMoE 官方地形/奖励/终止/指令/课程/观测/DR。

    2026-07-17 起对齐官方 12 自由度结构:
      actor 单帧 45 = 角速度3+重力3+指令3+q12+qd12+a12 (官方 num_one_step_observations=45),
      历史 H=10 (官方 num_observations = 45*10), critic 单帧 48 (45+基座线速度3),
      动作 12 (官方 num_actions=12)。map 3x7x11 (官方 77 点)。
    网络维度由 runner 从观测形状运行期推断, 无需改模型代码。
    """

    scene: CMoESceneCfg = CMoESceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CMoECommandsCfg = CMoECommandsCfg()
    rewards: CMoERewardsCfg = CMoERewardsCfg()
    terminations: CMoETerminationsCfg = CMoETerminationsCfg()
    curriculum: CMoECurriculumCfg = CMoECurriculumCfg()
    events: CMoEEventCfg = CMoEEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # CMoE 地形网格更重, physx 缓冲从父类迁入本任务(2026-07-18 分离)
        self.sim.physx.gpu_max_rigid_patch_count = 2 ** 20
        self.sim.physx.gpu_collision_stack_size = 2 ** 28
        # ---- 地形: 官方 8 类 (10m x 10m, hs=0.05, vs=0.005) ----
        # 用 .replace() 取独立副本: CMOE_TERRAINS_CFG 是模块级单例, _PLAY/_EVAL 会改写
        # num_rows/difficulty_range, 直接赋值会让同进程内多个环境实例互相污染。
        self.scene.terrain.terrain_generator = CMOE_TERRAINS_CFG.replace()
        # ---- 初始难度: 官方 legged_gym 默认 max_init_terrain_level = 5 (机器人初始
        # 随机分布在 0..5 级)。Isaac Lab 的 None 表示"取最高级"(即 0..9), 会把大量
        # 未训练的机器人直接扔上 0.7~0.9 难度地形, 放大早期失败率。对齐官方。----
        self.scene.terrain.max_init_terrain_level = 5
        # ---- 摩擦: 对齐官方"有效接触摩擦"分布 U(0.4, 0.9) —— 2026-07-18 修复 ----
        # 官方 IsaacGym: 地形材质摩擦 0.8 (legged_robot_config.py terrain.static_friction=0.8),
        #   机器人全 shape 摩擦每回合重采样 U(0,1) (legged_robot.py _process_rigid_shape_props /
        #   refresh_actor_rigid_shape_props), PhysX 默认按 **average** 合成两侧材质:
        #     μ_eff = (0.8 + U(0,1)) / 2 = U(0.4, 0.9), 均值 0.65, 下限 0.4。
        # 本移植继承的 rough_env_cfg 地形材质是 1.0 且 friction_combine_mode="multiply"
        #   (PhysX 规则: 配对取两侧 combine 枚举值较大者, multiply > average, multiply 生效):
        #     μ_eff = 1.0 x U(0,1) = U(0, 1) —— 40% 回合静摩擦低于官方下限 0.4, 10% 低于
        #     0.1(近似冰面); 动摩擦(见下方 events 勘误)更糟, 均值仅 0.33。
        # 前几轮审计只对了随机化区间的数字((0,1)==(0,1)), 漏了合成模式语义 —— 这里把
        # 地形材质改回官方的 0.8 + average, 机器人区间保持 (0,1), 有效分布即与官方逐位一致。
        # 注: rough_env_cfg.__post_init__ 里 self.sim.physics_material 与
        #     self.scene.terrain.physics_material 是同一对象引用, 原地改字段两处同时生效。
        self.scene.terrain.physics_material.static_friction = 0.8
        self.scene.terrain.physics_material.dynamic_friction = 0.8
        self.scene.terrain.physics_material.friction_combine_mode = "average"
        self.scene.terrain.physics_material.restitution_combine_mode = "average"
        self.sim.physics_material.static_friction = 0.8
        self.sim.physics_material.dynamic_friction = 0.8
        self.sim.physics_material.friction_combine_mode = "average"
        self.sim.physics_material.restitution_combine_mode = "average"
        # ---- 动作: 官方 action_scale = 0.25 (父类为 0.5) ----
        self.actions.joint_pos.scale = 0.25
        # ---- 动作空间: 29 -> 12。官方 num_actions=12, 上身在官方 URDF 里是焊死的;
        # 这里把动作限定到 12 个腿关节, 未被动作项覆盖的 17 个上身关节的 PD 目标
        # 恒为默认角, 由既有执行器位置伺服(等效"焊死"近似)。----
        self.actions.joint_pos.joint_names = list(LEG_JOINT_NAMES)

        # ---- 本体感受观测: 29 关节 -> 12 腿关节 (官方 45 维单帧) ----
        # 2026-07-17 修复: 四个观测项必须各自持有**独立的 SceneEntityCfg 实例**。
        # manager 解析时会原地改写该对象(把 joint_ids 从 slice 填成具体下标列表),
        # 共享同一实例会让第二个观测项解析时看到 "joint_names(正则) + joint_ids(已填)"
        # 并存, 触发 "Both 'joint_names' and 'joint_ids' are specified" 崩溃 ——
        # 与本文件高度图注释里 "actor_map/critic_map 必须各自独立 ObsTerm" 是同一类坑。
        self.observations.actor.joint_pos_rel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_NAMES)
        )
        self.observations.actor.joint_vel_rel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_NAMES)
        )
        self.observations.critic.joint_pos_rel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_NAMES)
        )
        self.observations.critic.joint_vel_rel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(LEG_JOINT_NAMES)
        )
        # last_action 项自动跟随动作管理器变为 12 维, 无需处理。

        # ---- 观测历史: 50 -> 10。官方 humanoid_config.py 第38行:
        # num_observations = num_one_step_observations * 10。父类的 50 步历史把
        # VAE 输入撑到 96x50=4800 维(官方 45x10=450), 白白放大状态估计难度。----
        self.observations.actor.history_length = 10
        # ---- 传感器与策略同频更新 ----
        for name in (
            "left_foot_sample", "right_foot_sample",
            "left_foot_edge", "right_foot_edge",
            "cmoe_height_scanner",
        ):
            getattr(self.scene, name).update_period = self.decimation * self.sim.dt
        # ---- 父类的两台 13×18 扫描器已被 cmoe_height_scanner 整体替代, 无消费者, 关掉省算力 ----
        self.scene.actor_height_scanner = None
        self.scene.critic_height_scanner = None

        # ---- 观测: 官方 scale / 噪声 ----
        # actor 本体感受 (顺序与官方不同但集合等价; 噪声幅值原本就与官方一致):
        #   ang_vel 缩放 0.2 -> 0.25 (官方 obs_scales.ang_vel)
        #   commands 乘官方 commands_scale = [lin_vel, lin_vel, ang_vel] = [2.0, 2.0, 0.25]
        self.observations.actor.base_ang_vel.scale = 0.25
        self.observations.actor.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25)
        # critic 的 base_lin_vel 乘官方 obs_scales.lin_vel = 2.0
        # (速度估计器的 MSE 目标随之处于 x2 空间, 与官方 privileged obs 一致)
        self.observations.critic.base_lin_vel.scale = 2.0
        # 官方 critic 拿到的本体感受与高度图和 actor 一样是带噪的 (humanoid.py 中
        # privileged_obs 由同一份 current_obs 切片而来), 因此打开 critic 噪声:
        self.observations.critic.enable_corruption = True
        self.scene.robot.actuators["N5020-16"].effort_limit_sim = {
        ".*_ankle_.*": 50.0,
        "waist_roll_joint": 50.0,
        "waist_pitch_joint": 50.0,
        ".*_shoulder_.*": 25.0,
        ".*_elbow_.*": 25.0,
        ".*_wrist_roll.*": 25.0,
        }
        # ---- 2026-07-19 上身对齐官方"焊死"(URDF: 腰3+臂14 全 fixed 于 q=0) ----
        # (a) 位姿: 官方焊在零位(手臂下垂)。原弯臂位双臂 8.09kg(23.5%)质心前移~2cm。
        self.scene.robot.init_state.joint_pos[".*_shoulder_pitch_joint"] = 0.0
        self.scene.robot.init_state.joint_pos["left_shoulder_roll_joint"] = 0.0
        self.scene.robot.init_state.joint_pos["right_shoulder_roll_joint"] = 0.0
        self.scene.robot.init_state.joint_pos[".*_elbow_joint"] = 0.0
        self.scene.robot.init_state.joint_pos["left_wrist_roll_joint"] = 0.0
        self.scene.robot.init_state.joint_pos["right_wrist_roll_joint"] = 0.0
        # (b) 握持刚度: 腰上 16.21kg、质心~0.25m -> 重力失稳刚度~40 N·m/rad, 与原 kp=40
        #     相抵(临界稳定)。腰提到 200/5(与 waist_yaw 同档), 臂 100/3; 踝保持官方 40/2。
        self.scene.robot.actuators["N5020-16"].stiffness = {
            ".*_ankle_.*": 40.0,
            "waist_roll_joint": 200.0,
            "waist_pitch_joint": 200.0,
            ".*_shoulder_.*": 100.0,
            ".*_elbow_.*": 100.0,
            ".*_wrist_roll.*": 100.0,
        }
        self.scene.robot.actuators["N5020-16"].damping = {
            ".*_ankle_.*": 2.0,
            "waist_roll_joint": 5.0,
            "waist_pitch_joint": 5.0,
            ".*_shoulder_.*": 3.0,
            ".*_elbow_.*": 3.0,
            ".*_wrist_roll.*": 3.0,
        }
        # ---- 高度图: 官方处理链 (绝对高 -> 0.2 陈旧 -> clip100 -> x5 -> +U(0.15) -> 椒盐 4+4) ----
        # 注意: actor_map 与 critic_map 必须各自持有**独立的 ObsTerm 实例**
        # (manager 会原地改写 term_cfg.params, 共享同一对象会让两个 group 互相干扰);
        # "两边看到同一张带噪图"是靠 cmoe_elevation_map 内部的 step 级缓存实现的, 与官方一致。
        def _make_cmoe_map_term() -> ObsTerm:
            return ObsTerm(
                func=mdp.cmoe_elevation_map,
                params={
                    "sensor_cfg": SceneEntityCfg("cmoe_height_scanner"),
                    "size": (7, 11),
                    "height_scale": 5.0,        # 官方 obs_scales.height_measurements
                    "height_noise": 0.03,       # 官方 noise_scales.height_measurements
                    "noise_level": 1.0,         # 官方 noise.noise_level
                    "staleness_prob": 0.2,      # 官方 0.2 概率沿用上一帧
                    "salt_pepper_points": 4,    # 官方 n=4/77, 现在逐位一致
                },
            )
        self.observations.actor_map.height_scanner = _make_cmoe_map_term()
        self.observations.critic_map.height_scanner = _make_cmoe_map_term()


@configclass
class G1CMoEEnvCfg_PLAY(G1CMoEEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
        self.commands.base_velocity.resampling_time_range = (30, 30)
        for rng in (self.commands.base_velocity.easy_ranges, self.commands.base_velocity.hard_ranges):
            rng.lin_vel_x = (0.8, 0.8)
            rng.lin_vel_y = (0.0, 0.0)
            rng.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }


@configclass
class G1CMoEEnvCfg_EVAL(G1CMoEEnvCfg):
    recorders: RecorderManagerCfg = RecorderManagerCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.num_rows = 10
        self.scene.terrain.terrain_generator.difficulty_range = (0.9, 1.0)
        self.commands.base_velocity.resampling_time_range = (30, 30)
        # 论文 benchmark: 固定 0.8 m/s 直行
        for rng in (self.commands.base_velocity.easy_ranges, self.commands.base_velocity.hard_ranges):
            rng.lin_vel_x = (0.8, 0.8)
            rng.lin_vel_y = (0.0, 0.0)
            rng.heading = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["y"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.terminations.success = DoneTerm(func=mdp.subterrain_out_of_bounds, params={"distance_buffer": 0.0})
