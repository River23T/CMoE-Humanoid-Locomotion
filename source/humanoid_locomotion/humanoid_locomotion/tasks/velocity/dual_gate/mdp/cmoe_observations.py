# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#


from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


# =============================================================================
#  1) 倒角采样 pattern: 每个网格点发 3 条射线 (x-0.05, x, x+0.05)
#     官方 _get_heights 取 mean(h1, h1, h2, h3) = 0.5*中点 + 0.25*(x+1格) + 0.25*(x-1格),
#     其中格距 = horizontal_scale = 0.05m —— 即论文的"高程图直角边倒角"。
# =============================================================================
def chamfer_grid_pattern(cfg: "ChamferGridPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    nx = round(cfg.size[0] / cfg.resolution) + 1          # x 方向点数 (1.36/0.08+1 = 18)
    ny = round(cfg.size[1] / cfg.resolution) + 1          # y 方向点数 (0.96/0.08+1 = 13)
    x = torch.linspace(-cfg.size[0] / 2, cfg.size[0] / 2, nx, device=device)
    y = torch.linspace(-cfg.size[1] / 2, cfg.size[1] / 2, ny, device=device)
    gy, gx = torch.meshgrid(y, x, indexing="ij")          # (ny, nx): 行=y(13), 列=x(18), 与 (13,18) reshape 对齐
    base = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(nx * ny, device=device)], dim=1)  # (P, 3)
    offs = torch.tensor(cfg.chamfer_offsets, dtype=torch.float32, device=device)                  # (3,)
    starts = base.unsqueeze(1).repeat(1, offs.numel(), 1)
    starts[:, :, 0] += offs.unsqueeze(0)
    starts = starts.reshape(-1, 3)                        # (P*3, 3), 三条一组连续排列
    dirs = torch.zeros_like(starts)
    dirs[:, 2] = -1.0
    return starts, dirs


@configclass
class ChamferGridPatternCfg(PatternBaseCfg):
    func = chamfer_grid_pattern
    resolution: float = 0.08
    size: tuple[float, float] = (1.36, 0.96)              # (x 范围, y 范围), 与你原 13x18 网格一致
    chamfer_offsets: tuple = (-0.05, 0.0, 0.05)           # 官方 h[px-1] / h[px] / h[px+1] (hs=0.05m)


# =============================================================================
#  2) 官方椒盐噪声 disturb_heights_extreme —— 逐字移植 (humanoid.py L223-262)
#     每行(每个 env)随机选 2n 个互不相同的点: n 个抬到 U(max, 2max-min),
#     n 个压到 U(2min-max, min)。官方 n=4 (77 点中的 4+4)。
# =============================================================================
def disturb_heights_extreme(heights: torch.Tensor, n: int = 4) -> torch.Tensor:
    B, N = heights.shape
    assert 2 * n <= N, f"disturbance points per row 2n={2 * n} cannot exceed number of columns N={N}"
    row_max = heights.max(dim=1, keepdim=True)[0]
    row_min = heights.min(dim=1, keepdim=True)[0]
    # upward: [max, 2max - min]
    high_upper = 2 * row_max - row_min
    rand_upper = torch.rand((B, n), device=heights.device)
    rand_upper = rand_upper * (high_upper - row_max) + row_max
    # downward: [2min - max, min]
    low_lower = 2 * row_min - row_max
    rand_lower = torch.rand((B, n), device=heights.device)
    rand_lower = rand_lower * (row_min - low_lower) + low_lower
    all_indices = torch.multinomial(torch.ones((B, N), device=heights.device), num_samples=2 * n, replacement=False)
    upper_indices = all_indices[:, :n]
    lower_indices = all_indices[:, n:]
    row_indices = torch.arange(B, device=heights.device).unsqueeze(1).expand(-1, n)
    heights[row_indices, upper_indices] = rand_upper
    heights[row_indices, lower_indices] = rand_lower
    return heights


# =============================================================================
#  3) 官方高度观测 —— 完整处理链
#     绝对地形高 (世界系网格, 倒角平均) -> 0.2 概率不更新 -> clip(±100) -> x5.0
#     -> +U(±1)*0.03*5 -> 椒盐 -> 拼进 (B, 3, 13, 18) 的 z 通道。
#     actor_map 与 critic_map 共用同一份带噪结果(官方 current_obs 只算一次 heights,
#     critic 拿到的同样是带噪高度), 通过 step 级缓存实现。
# =============================================================================
def cmoe_elevation_map(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("cmoe_height_scanner"),
    size: tuple[int, int] = (13, 18),
    height_scale: float = 5.0,          # 官方 obs_scales.height_measurements
    height_noise: float = 0.03,         # 官方 noise_scales.height_measurements
    noise_level: float = 1.0,           # 官方 noise.noise_level
    staleness_prob: float = 0.2,        # 官方: 每步 0.2 概率沿用上一帧 measured_heights
    salt_pepper_points: int = 4,        # 官方 n=4 (77 点); 与官方等比例到 234 点约为 12
    raw_clip: float = 100.0,            # 官方 clip(measured_heights, -100, 100)
) -> torch.Tensor:
    # step 级缓存: 同一步内 actor_map / critic_map 返回同一张带噪图 (官方行为)
    step = int(env.common_step_counter)
    if getattr(env, "_cmoe_map_step", -1) == step and hasattr(env, "_cmoe_map_out"):
        return env._cmoe_map_out

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]                                   # (B, P*3)
    B = hits_z.shape[0]
    P = size[0] * size[1]
    hz = torch.nan_to_num(hits_z.view(B, P, 3), nan=0.0, posinf=0.0, neginf=0.0)
    # 官方 3 点倒角平均: mean(h1, h1, h2, h3), pattern 顺序为 (x-0.05, x, x+0.05)
    raw = 0.25 * hz[..., 0] + 0.5 * hz[..., 1] + 0.25 * hz[..., 2]            # (B, P) 绝对地形高

    # 首次调用: 初始化缓存与网格常量通道
    if not hasattr(env, "_cmoe_map_raw"):
        env._cmoe_map_raw = raw.clone()
        pat = sensor.cfg.pattern_cfg
        nx = round(pat.size[0] / pat.resolution) + 1
        ny = round(pat.size[1] / pat.resolution) + 1
        gx = torch.linspace(-pat.size[0] / 2, pat.size[0] / 2, nx, device=env.device)
        gy = torch.linspace(-pat.size[1] / 2, pat.size[1] / 2, ny, device=env.device)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")                        # (13, 18)
        env._cmoe_grid_xy = torch.stack([xx, yy], dim=0)                      # (2, 13, 18) 常量

    # 陈旧帧: 每 env 以 staleness_prob 概率保留旧值; 刚重置的 env 强制刷新
    keep = torch.rand(B, 1, device=env.device) < staleness_prob
    keep &= ~(env.episode_length_buf.unsqueeze(1) <= 1)
    env._cmoe_map_raw = torch.where(keep, env._cmoe_map_raw, raw)

    # 官方顺序: clip -> 缩放 -> 均匀噪声 -> 椒盐
    h = torch.clip(env._cmoe_map_raw, -raw_clip, raw_clip) * height_scale
    h = h + (2.0 * torch.rand_like(h) - 1.0) * (height_noise * noise_level * height_scale)
    if salt_pepper_points > 0:
        h = disturb_heights_extreme(h, n=salt_pepper_points)

    grid = env._cmoe_grid_xy.unsqueeze(0).expand(B, -1, -1, -1)               # (B, 2, 13, 18)
    out = torch.cat([grid, h.view(B, 1, size[0], size[1])], dim=1).contiguous()  # (B, 3, 13, 18)
    env._cmoe_map_step = step
    env._cmoe_map_out = out
    return out


# =============================================================================
#  4) 官方 push: 把基座 xy 线速度**替换**为 U(-max, max) (legged_robot._push_robots)
#     区别于 isaaclab 自带 push_by_setting_velocity 的"叠加"语义。
# =============================================================================
def push_by_replacing_velocity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    elif not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    vel = asset.data.root_vel_w[env_ids].clone()
    for i, key in enumerate(("x", "y")):
        lo, hi = velocity_range.get(key, (0.0, 0.0))
        vel[:, i] = torch.empty(env_ids.numel(), device=asset.device).uniform_(lo, hi)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


# =============================================================================
#  5) 官方指令课程 (legged_robot.update_command_curriculum 逐句移植)
#     条件: easy 组与 hard 组的 tracking_lin_vel 每步均值都 > 0.7 * 权重
#     动作: 两档 lin_vel_x 上限各 +0.1, 分别 clip 到 max_easy(3.0)/max_hard(1.0)
#     注意: 官方 G1CMoECfg.commands.curriculum = False, 即发布训练**未启用**,
#     这里按官方基础设施实现, 由 cmoe_env_cfg 里的注释行决定是否挂载。
# =============================================================================
def cmoe_command_levels(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    reward_term: str = "tracking_lin_vel",
    success_ratio: float = 0.7,
    increment: float = 0.1,
    max_easy: float = 3.0,              # 官方 commands.max_easy_terrain_curriculum
    max_hard: float = 1.0,              # 官方 commands.max_hard_terrain_curriculum
) -> dict[str, torch.Tensor]:
    cmd = env.command_manager.get_term(command_name)
    if not torch.is_tensor(env_ids):
        env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=env.device)
    easy_all = cmd._easy_mask()                                              # CMoEVelocityCommand 的列级 easy 掩码
    easy_ids = env_ids[easy_all[env_ids]]
    hard_ids = env_ids[~easy_all[env_ids]]

    ok = len(easy_ids) > 0 and len(hard_ids) > 0
    if ok:
        sums = env.reward_manager._episode_sums[reward_term]
        weight = env.reward_manager.get_term_cfg(reward_term).weight
        # isaaclab 的 episode_sums 已含 dt, 除以 episode 时长得到每秒均值,
        # 阈值 0.7*weight 与官方 (sums/max_episode_length > 0.7*scale, scale 含 dt) 等价。
        thr = success_ratio * weight
        ok = (sums[easy_ids].mean() / env.max_episode_length_s > thr) and (
            sums[hard_ids].mean() / env.max_episode_length_s > thr
        )
    if ok:
        er, hr = cmd.cfg.easy_ranges, cmd.cfg.hard_ranges
        er.lin_vel_x = (er.lin_vel_x[0], min(er.lin_vel_x[1] + increment, max_easy))
        hr.lin_vel_x = (hr.lin_vel_x[0], min(hr.lin_vel_x[1] + increment, max_hard))
    return {
        "easy_max_vx": torch.tensor(cmd.cfg.easy_ranges.lin_vel_x[1], device=env.device),
        "hard_max_vx": torch.tensor(cmd.cfg.hard_ranges.lin_vel_x[1], device=env.device),
    }
