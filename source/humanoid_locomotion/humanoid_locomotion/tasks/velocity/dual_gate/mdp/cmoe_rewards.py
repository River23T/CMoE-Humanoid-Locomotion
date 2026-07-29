# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from dataclasses import MISSING
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, wrap_to_pi, euler_xyz_from_quat
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


# =============================================================================
#  0) 脚底 5 采样点: 官方 URDF foot_sample_point1..5 (ankle_roll 系, 鞋底 z=-0.035)
# =============================================================================
CMOE_FOOT_SAMPLE_OFFSETS = (
    (0.03, 0.0, -0.035),
    (0.12, 0.0, -0.035),
    (-0.05, 0.0, -0.035),
    (0.06, 0.03, -0.035),
    (0.06, -0.03, -0.035),
)
CMOE_FOOT_HEIGHT = 0.035  # 官方 cfg.rewards.foot_height


def foot_sample_pattern(cfg: "FootSamplePatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """RayCaster 自定义 pattern: 官方 5 个脚底采样点 (只取 xy, 射线竖直向下)。"""
    ray_starts = torch.tensor([[p[0], p[1], 0.0] for p in cfg.points], dtype=torch.float32, device=device)
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., 2] = -1.0
    return ray_starts, ray_directions


@configclass
class FootSamplePatternCfg(PatternBaseCfg):
    func = foot_sample_pattern
    points: tuple = CMOE_FOOT_SAMPLE_OFFSETS


# =============================================================================
#  1) 内部工具
# =============================================================================
def _resolve_ids(env, cfg: SceneEntityCfg) -> SceneEntityCfg:
    """确保 SceneEntityCfg 已解析出 body_ids/joint_ids。

    Isaac Lab 的 manager 只 resolve 写在 term_cfg.params 里的 SceneEntityCfg
    (manager_base._process_term_cfg_at_play), 写在函数签名里的**默认值**永远不会被
    解析, body_ids 会保持 slice(None) 从而切到全部刚体。这里做一次幂等兜底解析,
    使函数无论是否显式传 params 都正确 (解析结果缓存在 cfg 对象上, 只做一次)。
    """
    if isinstance(cfg.body_ids, slice) and cfg.body_names is not None:
        cfg.resolve(env.scene)
    elif isinstance(cfg.joint_ids, slice) and cfg.joint_names is not None:
        cfg.resolve(env.scene)
    return cfg


def _contact_filt(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float = 2.0) -> torch.Tensor:
    """官方 contact_filt = (|F_t|>2) | (|F_{t-1}|>2) 的等价实现。

    官方在策略步 (0.02s) 记录 last_contacts; 这里用 ContactSensor 的力历史
    (history_length=3 @ 物理步 0.005s, 覆盖 0.015s) 取最大, 语义相同(抗 PhysX 抖动)。
    """
    _resolve_ids(env, sensor_cfg)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, T, B, 3)
    return forces.norm(dim=-1).max(dim=1)[0] > threshold  # (N, B) bool


def _terrain(env) -> TerrainImporter:
    return env.scene.terrain


def _terrain_type_ids(env, names: Sequence[str]) -> torch.Tensor:
    """sub_terrains 名称 -> 类型号 (dict 顺序即类型号, 与 terrains/config/cmoe.py 对齐)。"""
    keys = list(_terrain(env).cfg.terrain_generator.sub_terrains.keys())
    ids = [keys.index(n) for n in names if n in keys]
    return torch.tensor(ids, dtype=torch.long, device=env.device)


def _env_terrain_type(env) -> torch.Tensor:
    """每个 env 的**类型号** (N,)。terrain_types 是列号(0..num_cols-1), 这里按
    TerrainGenerator._generate_curriculum_terrains 的同式把列号映射回类型号:
        sub_index = min(where(col/num_cols + 0.001 < cumsum(proportions)))
    2026-07-19 修复: 此前直接拿列号当类型号比较, 20 列 x 9 类下仅列 0/1 侥幸对齐。"""
    if not hasattr(env, "_cmoe_col2type"):
        import numpy as np
        gen = _terrain(env).cfg.terrain_generator
        props = np.array([s.proportion for s in gen.sub_terrains.values()], dtype=float)
        csum = np.cumsum(props / props.sum())
        col2type = [int(np.min(np.where(c / gen.num_cols + 0.001 < csum)[0])) for c in range(gen.num_cols)]
        env._cmoe_col2type = torch.tensor(col2type, dtype=torch.long, device=env.device)
    return env._cmoe_col2type[_terrain(env).terrain_types]


def _foot_body_id(env, body_name: str) -> int:
    asset: Articulation = env.scene["robot"]
    return asset.find_bodies(body_name)[0][0]


# =============================================================================
#  2) 奖励 —— 官方 _reward_* 逐条移植 (Table II)
# =============================================================================
def track_heading_exp(env: "ManagerBasedRLEnv", command_name: str = "base_velocity") -> torch.Tensor:
    """官方 _reward_tracking_yaw: exp(-|wrap_to_pi(heading_cmd - heading)|)。

    官方跟踪的是 heading 目标角(commands[:,3]), 不是 yaw 角速度; 这里从
    UniformVelocityCommand.heading_target 取目标 (需 heading_command=True)。
    """
    cmd_term = env.command_manager.get_term(command_name)
    heading = env.scene["robot"].data.heading_w
    return torch.exp(-torch.abs(wrap_to_pi(cmd_term.heading_target - heading)))


def base_height_l2_cmoe(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.75,
    foot_height: float = CMOE_FOOT_HEIGHT,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """官方 _reward_base_height + _get_base_heights: 以**触地脚**为参考面的机身高度。

    measured = 触地脚 z 的均值(无触地则两脚均值); base_h = root_z - (measured - 0.035);
    reward = (base_h - 0.75)^2。官方不用高度扫描, 纯由脚位置得到 —— 完全照搬。
    """
    _resolve_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]                 # (N, 2)
    filt = _contact_filt(env, sensor_cfg).float()                            # (N, 2)
    cnt = filt.sum(dim=1)
    measured = torch.where(
        cnt > 0,
        (feet_z * filt).sum(dim=1) / cnt.clamp(min=1.0),
        feet_z.mean(dim=1),
    )
    base_h = asset.data.root_pos_w[:, 2] - (measured - foot_height)
    return torch.square(base_h - target_height)


def feet_stumble_cmoe(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """官方 _reward_feet_stumble: any(|F_xy| > 3*|F_z|) —— 论文 Table II 同式 (系数 3)。"""
    _resolve_ids(env, sensor_cfg)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]                # (N, 2, 3)
    return torch.any(
        torch.norm(forces[:, :, :2], dim=2) > 3.0 * torch.abs(forces[:, :, 2]), dim=1
    ).float()


def feet_lateral_distance(
    env: "ManagerBasedRLEnv",
    min_dist: float = 0.18,
    max_dist: float = 0.24,
    left_body: str = "left_ankle_roll_link",
    right_body: str = "right_ankle_roll_link",
) -> torch.Tensor:
    """官方 _reward_feet_lateral_distance: 基座系下两脚 y 间距, clamp(d-0.18, max=0.06)。

    用显式左/右 body 名取脚 (不用正则 SceneEntityCfg): 该奖励要的是"左脚 y - 右脚 y",
    依赖左右次序; 正则匹配返回的次序由资产 body 索引决定, 显式命名更稳。
    """
    asset: Articulation = env.scene["robot"]
    if not hasattr(env, "_cmoe_lat_ids"):
        env._cmoe_lat_ids = (_foot_body_id(env, left_body), _foot_body_id(env, right_body))
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    y_base = []
    for bid in env._cmoe_lat_ids:
        rel = asset.data.body_pos_w[:, bid] - root_pos
        y_base.append(quat_apply_inverse(root_quat, rel)[:, 1])
    lateral_diff = torch.abs(y_base[0] - y_base[1])
    return torch.clamp(lateral_diff - min_dist, max=max_dist - min_dist)


def feet_air_time_cmoe(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """官方 _reward_feet_air_time: sum((air_time - 0.5) * first_contact)。

    与 isaaclab 自带 feet_air_time 的区别: 官方**没有**指令幅值门控, 这里照搬不加门控。
    """
    _resolve_ids(env, sensor_cfg)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    return torch.sum((last_air_time - threshold) * first_contact, dim=1)


def feet_ground_parallel(
    env: "ManagerBasedRLEnv",
    left_sensor: str = "left_foot_sample",
    right_sensor: str = "right_foot_sample",
    left_body: str = "left_ankle_roll_link",
    right_body: str = "right_ankle_roll_link",
    offsets: tuple = CMOE_FOOT_SAMPLE_OFFSETS,
) -> torch.Tensor:
    """官方 _reward_feet_ground_parallel: var(左脚 5 采样点离地高) + var(右脚同)。

    采样点世界 z 由脚部完整位姿 + 官方 URDF 偏移解析计算(与官方刚体点一致);
    点正下方地形高度由脚部 RayCaster(5 点 pattern, yaw 对齐)给出。
    """
    asset: Articulation = env.scene["robot"]
    if not hasattr(env, "_cmoe_foot_ids"):
        env._cmoe_foot_ids = (_foot_body_id(env, left_body), _foot_body_id(env, right_body))
        env._cmoe_foot_offsets = torch.tensor(offsets, dtype=torch.float32, device=env.device)  # (5,3)
    off = env._cmoe_foot_offsets
    total = 0.0
    for bid, sname in ((env._cmoe_foot_ids[0], left_sensor), (env._cmoe_foot_ids[1], right_sensor)):
        pos = asset.data.body_pos_w[:, bid]                                   # (N, 3)
        quat = asset.data.body_quat_w[:, bid]                                 # (N, 4)
        n = pos.shape[0]
        pts_w = pos.unsqueeze(1) + quat_apply(
            quat.unsqueeze(1).expand(-1, off.shape[0], -1).reshape(-1, 4),
            off.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3),
        ).reshape(n, -1, 3)                                                   # (N, 5, 3)
        sensor: RayCaster = env.scene.sensors[sname]
        hits_z = sensor.data.ray_hits_w[..., 2]                               # (N, 5)
        h = pts_w[..., 2] - hits_z
        h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))            # 射线未命中时置 0 兜底
        total = total + torch.var(h, dim=-1)
    return total


def joint_deviation_sq(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """官方 _reward_hip_dof_error 的通用式: sum((q - q_default)^2) (无阈值)。

    官方对 hip_roll + hip_yaw 两组求和; 在 cfg 里用 joint_names 正则同时选中两组即可。
    """
    _resolve_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(dev), dim=1)


def joint_pos_limits_soft(
    env: "ManagerBasedRLEnv", soft_ratio: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """官方 _reward_dof_pos_limits (soft_dof_pos_limit=0.9): 越过软限位的量。

    legged_gym 在建 env 时把限位按中点缩到 0.9 倍量程; 这里在奖励里等价地现算:
    lo' = m - 0.5*r*ratio, hi' = m + 0.5*r*ratio。
    """
    _resolve_ids(env, asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]              # (N, J, 2)
    mid = 0.5 * (limits[..., 0] + limits[..., 1])
    rng = limits[..., 1] - limits[..., 0]
    lo = mid - 0.5 * rng * soft_ratio
    hi = mid + 0.5 * rng * soft_ratio
    out = (lo - q).clip(min=0.0) + (q - hi).clip(min=0.0)
    return torch.sum(out, dim=1)


def feet_edge_cmoe(
    env: "ManagerBasedRLEnv",
    left_edge_sensor: str = "left_foot_edge",
    right_edge_sensor: str = "right_foot_edge",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    exclude_terrains: Sequence[str] = ("stairs_up", "stairs_down", "discrete"),
    level_threshold: int = 3,
    edge_height: float = 1.5 * 0.05,
) -> torch.Tensor:
    """官方 _reward_feet_edge: 触地脚落在地形棱边上则罚。

    官方棱边 = 高度场 trimesh 化时被 slope_treshold(1.5) 竖直化的格点, 再按 0.05m 膨胀。
    等价判据: 相邻 0.05m 格点高差 > 1.5*0.05 = 0.075m。这里用脚下 3x3@0.05m 射线小栅格
    (覆盖 ±0.05m, 即官方 1 格膨胀范围) 的 max-min > 0.075m 判棱边 —— 运行期等价实现。
    官方门控照搬: 仅 terrain_level > 3 计罚; env_class 为 stairs_up/stairs_down/discrete 置 0。
    """
    filt = _contact_filt(env, sensor_cfg)                                     # (N, 2) 列序 = body_ids 序 = [left, right]
    edges = []
    for sname in (left_edge_sensor, right_edge_sensor):
        hits_z = env.scene.sensors[sname].data.ray_hits_w[..., 2]             # (N, 9)
        finite = torch.isfinite(hits_z)
        zmax = torch.where(finite, hits_z, torch.full_like(hits_z, -1e6)).max(dim=1)[0]
        zmin = torch.where(finite, hits_z, torch.full_like(hits_z, 1e6)).min(dim=1)[0]
        edges.append((zmax - zmin) > edge_height)
    at_edge = filt & torch.stack(edges, dim=1)                                # (N, 2)
    terrain = _terrain(env)
    rew = (terrain.terrain_levels > level_threshold).float() * at_edge.sum(dim=-1).float()
    if not hasattr(env, "_cmoe_edge_excl_ids"):
        env._cmoe_edge_excl_ids = _terrain_type_ids(env, exclude_terrains)
    if len(env._cmoe_edge_excl_ids) > 0:
        rew[torch.isin(_env_terrain_type(env), env._cmoe_edge_excl_ids)] = 0.0
    return rew


# =============================================================================
#  3) 终止 —— 官方 humanoid.check_termination 移植
# =============================================================================
def bad_roll_pitch(env: "ManagerBasedRLEnv", limit_angle: float = 1.5) -> torch.Tensor:
    """官方 check_termination: |roll| > 1.5 或 |pitch| > 1.5 即终止 (分轴判定, 非合成倾角)。
    2026-07-17 勘误: 此前注释误写为 1.0; 官方 legged_robot.py 第 272-273 行实际为 1.5。"""
    asset: Articulation = env.scene["robot"]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)
    return (torch.abs(roll) > limit_angle) | (torch.abs(pitch) > limit_angle)


def root_height_below_on_terrain_types(
    env: "ManagerBasedRLEnv",
    minimum_height: float = 0.5,
    terrain_names: Sequence[str] = ("parkour_gap",),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """官方: height_cutoff = (root_z < 0.5) 且 env_class == 5 (仅 gap 地形生效)。

    官方 gap 瓦片地表 z=0, root 的世界 z 直接可比; Isaac Lab 里 parkour 瓦片 origin z 也为 0,
    world z 判据保持一致。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    below = asset.data.root_pos_w[:, 2] < minimum_height
    if not hasattr(env, "_cmoe_gap_type_ids"):
        env._cmoe_gap_type_ids = _terrain_type_ids(env, terrain_names)
    on_gap = torch.isin(_env_terrain_type(env), env._cmoe_gap_type_ids)
    return below & on_gap


# =============================================================================
#  4) 指令 —— 官方 easy/hard 双档采样 (legged_robot._resample_commands)
# =============================================================================
class CMoEVelocityCommand(UniformVelocityCommand):
    """按地形难度分档采样速度指令 —— 照搬官方 _resample_commands。

    官方按 env_id < num_envs*easy_terrain 划分 easy/hard; Isaac Lab 的 env->列 映射
    (terrain_types = floor(env_id / (num_envs/num_cols))) 与 legged_gym 完全相同,
    因此这里直接用 terrain_types 是否属于 easy 列来划分 —— 结果与官方逐 env 一致。
    另照搬官方 "set small commands to zero": ||v_xy|| <= 0.2 置零。
    """

    cfg: "CMoEVelocityCommandCfg"

    def _easy_mask(self) -> torch.Tensor:
        if not hasattr(self, "_easy_mask_cache"):
            terrain: TerrainImporter = self._env.scene.terrain
            keys = list(terrain.cfg.terrain_generator.sub_terrains.keys())
            ids = [keys.index(n) for n in self.cfg.easy_terrain_names if n in keys]
            ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
            self._easy_mask_cache = torch.isin(_env_terrain_type(self._env), ids_t)
        return self._easy_mask_cache

    def _resample_command(self, env_ids: Sequence[int]):
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        easy = self._easy_mask()[env_ids]
        for mask, ranges in ((easy, self.cfg.easy_ranges), (~easy, self.cfg.hard_ranges)):
            ids = env_ids[mask]
            if len(ids) == 0:
                continue
            r = torch.empty(len(ids), device=self.device)
            self.vel_command_b[ids, 0] = r.uniform_(*ranges.lin_vel_x)
            self.vel_command_b[ids, 1] = r.uniform_(*ranges.lin_vel_y)
            self.vel_command_b[ids, 2] = r.uniform_(*ranges.ang_vel_z)
            if self.cfg.heading_command:
                self.heading_target[ids] = r.uniform_(*ranges.heading)
                self.is_heading_env[ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        r = torch.empty(len(env_ids), device=self.device)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        # 官方: commands[:, :2] *= (norm(commands[:, :2]) > 0.2)
        small = torch.norm(self.vel_command_b[env_ids, :2], dim=1) <= self.cfg.min_command_norm
        self.vel_command_b[env_ids[small], :2] = 0.0


@configclass
class CMoEVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CMoEVelocityCommand

    easy_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """easy 地形 (rough slope 列) 的指令范围 —— 官方 easy_terrain_ranges。"""
    hard_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """其余 (parkour 等) 地形的指令范围 —— 官方 hard_terrain_ranges。"""
    easy_terrain_names: list[str] = ["rough_slope_inv", "rough_slope"]
    min_command_norm: float = 0.2


# =============================================================================
#  5) 地形课程 —— 官方 _update_terrain_curriculum (0.8/0.4 * cmd_x * T)
# =============================================================================
def cmoe_terrain_levels(
    env: "ManagerBasedRLEnv", env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    """官方判据: 行走距离 > 0.8 * cmd_x * episode_length_s 升级; < 0.4 * 阈值 降级。

    与 isaaclab 自带 terrain_levels_vel 的区别: 官方阈值随**指令 x 分量**缩放
    (自带版用固定半瓦片长升级)。升满级随机重置一行 —— terrain.update_env_origins
    内部逻辑与官方逐句相同, 直接复用。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    threshold = command[env_ids, 0] * env.max_episode_length_s
    move_up = distance > 0.8 * threshold
    move_down = distance < 0.4 * threshold
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    # 分地形记录平均等级 (沿用你工程 attention_terrain_levels 的日志风格)
    sub_terrains = list(terrain.cfg.terrain_generator.sub_terrains.keys())
    levels: dict[str, torch.Tensor] = {}
    env_types = _env_terrain_type(env)
    for i, name in enumerate(sub_terrains):
        mask = env_types == i
        if torch.any(mask):
            levels[name] = torch.mean(terrain.terrain_levels[mask].float())
    levels["all"] = torch.mean(terrain.terrain_levels.float())
    return levels


# =============================================================================
#  6) (可选) 高程图椒盐噪声 —— 论文 Eq.(9) / 官方 disturb_heights_extreme
# =============================================================================
def salt_pepper_heights(z: torch.Tensor, p: float = 0.02) -> torch.Tensor:
    """论文 Eq.(9): 以概率 p 抬到 U(M, 2M-m), 概率 p 压到 U(2m-M, m)。z: (N, P)。"""
    M = z.max(dim=1, keepdim=True)[0]
    m = z.min(dim=1, keepdim=True)[0]
    span = (M - m).clamp(min=1e-6)
    r = torch.rand_like(z)
    up = M + torch.rand_like(z) * span          # [M, 2M-m]
    down = (2 * m - M) + torch.rand_like(z) * span  # [2m-M, m]
    z = torch.where(r < p, up, z)
    z = torch.where((r >= p) & (r < 2 * p), down, z)
    return z
