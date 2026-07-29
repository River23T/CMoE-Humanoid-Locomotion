# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
import trimesh

from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh, convert_height_field_to_mesh


# ---------------------------------------------------------------------------- #
#  官方 Terrain.add_roughness + isaacgym.terrain_utils.random_uniform_terrain 移植
# ---------------------------------------------------------------------------- #
# 官方: 每种被选中的地形(含 parkour)最后都调用 add_roughness(terrain, difficulty):
#   max_h = (0.03-0.01)*difficulty + 0.01;  h = U(0.01, max_h)
#   random_uniform_terrain(min=-h, max=+h, step=0.005, downsampled_scale=0.075)  (叠加 +=)
_ROUGH_HEIGHT = (0.01, 0.03)      # 官方 cfg.terrain.rough_height
_ROUGH_STEP = 0.005               # 官方 add_roughness 固定 step
_ROUGH_DOWNSAMPLE = 0.075         # 官方 cfg.terrain.downsampled_scale


def _add_roughness(hf_raw: np.ndarray, difficulty: float, horizontal_scale: float, vertical_scale: float) -> None:
    """In-place 叠加分形噪声 —— 照搬官方 add_roughness + isaacgym random_uniform_terrain(线性插值)。"""
    max_height = (_ROUGH_HEIGHT[1] - _ROUGH_HEIGHT[0]) * difficulty + _ROUGH_HEIGHT[0]
    height = np.random.uniform(_ROUGH_HEIGHT[0], max_height)
    # isaacgym.random_uniform_terrain (kind='linear'):
    min_h = int(-height / vertical_scale)
    max_h = int(height / vertical_scale)
    step_h = int(_ROUGH_STEP / vertical_scale)
    heights_range = np.arange(min_h, max_h + step_h, step_h)
    W, L = hf_raw.shape
    w_ds = max(int(W * horizontal_scale / _ROUGH_DOWNSAMPLE), 2)
    l_ds = max(int(L * horizontal_scale / _ROUGH_DOWNSAMPLE), 2)
    hf_ds = np.random.choice(heights_range, (w_ds, l_ds))
    x = np.linspace(0, W * horizontal_scale, w_ds)
    y = np.linspace(0, L * horizontal_scale, l_ds)
    # 官方用 scipy.interp2d(kind='linear'), 新版 scipy 已移除 -> 等价的 RegularGridInterpolator(线性)
    f = interpolate.RegularGridInterpolator((x, y), hf_ds, method="linear")
    x_up = np.linspace(0, W * horizontal_scale, W)
    y_up = np.linspace(0, L * horizontal_scale, L)
    xg, yg = np.meshgrid(x_up, y_up, indexing="ij")
    z_up = np.rint(f(np.stack([xg, yg], axis=-1)))
    hf_raw += z_up.astype(hf_raw.dtype)


def _pad_edges(hf_raw: np.ndarray, horizontal_scale: float, vertical_scale: float,
               pad_width: float = 0.1, pad_height: float = 0.0) -> None:
    """官方 parkour_* 函数结尾的 pad edges (官方所有调用均 pad_height=0)。"""
    pw = int(pad_width // horizontal_scale)
    ph = int(pad_height // vertical_scale)
    if pw > 0:
        hf_raw[:, :pw] = ph
        hf_raw[:, -pw:] = ph
        hf_raw[:pw, :] = ph
        hf_raw[-pw:, :] = ph


def _hf_to_mesh_with_parkour_origin(hf_raw: np.ndarray, cfg: HfTerrainBaseCfg):
    """高度场 -> (mesh, 官方 parkour 出生点 origin=(0.75, W/2, 0))。"""
    vertices, triangles = convert_height_field_to_mesh(
        hf_raw, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    # 官方 add_terrain_to_map: parkour -> env_origin = (i*env_length + 0.75, 中线, 0)
    origin = np.array([0.75, 0.5 * cfg.size[1], 0.0])
    return [mesh], origin


def _alloc_hf(cfg: HfTerrainBaseCfg, standalone: bool = False) -> np.ndarray:
    """分配高度场缓冲。

    - 被 @height_field_to_mesh 装饰的函数: 装饰器在外层自带 +1 像素缓冲并把 cfg.size
      缩掉边框, 函数内必须按 int(size/hs) 分配 (与 Isaac Lab 自带 hf 地形一致)。
    - 独立 mesh 函数 (parkour 4 类): 自己负责整块瓦片, 按 Isaac Lab 约定 int(size/hs)+1
      分配, 使相邻瓦片无缝。
    """
    extra = 1 if standalone else 0
    w = int(cfg.size[0] / cfg.horizontal_scale + 1e-6) + extra
    l = int(cfg.size[1] / cfg.horizontal_scale + 1e-6) + extra
    return np.zeros((w, l), dtype=np.int16)


# ---------------------------------------------------------------------------- #
#  1) rough slope —— 官方 idx1: pyramid_sloped(slope=±0.4*difficulty) + roughness
# ---------------------------------------------------------------------------- #
@height_field_to_mesh
def cmoe_rough_slope_terrain(difficulty: float, cfg: "CMoERoughSlopeTerrainCfg") -> np.ndarray:
    slope = difficulty * 0.4                       # 官方: slope = difficulty*0.4 (~20°)
    if cfg.inverted:
        slope = -slope                             # 官方: choice<(p0+p1)/2 时 slope*=-1
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg)
    W, L = hf.shape
    # ---- isaacgym.terrain_utils.pyramid_sloped_terrain 照搬 ----
    x = np.arange(0, W)
    y = np.arange(0, L)
    center_x = int(W / 2)
    center_y = int(L / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(W, 1)
    yy = yy.reshape(1, L)
    max_height = int(slope * (hs / vs) * (W / 2))
    hf += (max_height * xx * yy).astype(hf.dtype)
    # 中心平台 (platform_size=3.0): 以平台角点高度截断出平顶/平底
    platform_size = int(3.0 / hs / 2)
    x1 = W // 2 - platform_size
    x2 = W // 2 + platform_size
    y1 = L // 2 - platform_size
    y2 = L // 2 + platform_size
    min_h = min(int(hf[x1, y1]), 0)
    max_h = max(int(hf[x1, y1]), 0)
    hf = np.clip(hf, min_h, max_h)
    # ---- 官方 add_roughness(terrain, difficulty) ----
    _add_roughness(hf, difficulty, hs, vs)
    return hf


@configclass
class CMoERoughSlopeTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_rough_slope_terrain
    inverted: bool = False
    """False: 中心凸起(下坡出发); True: 中心凹陷(官方 choice<0.05 的 slope*=-1)。"""


# ---------------------------------------------------------------------------- #
#  2) stairs up / down —— 官方 idx2/3: 本地版 pyramid_stairs_terrain + roughness
# ---------------------------------------------------------------------------- #
@height_field_to_mesh
def cmoe_pyramid_stairs_terrain(difficulty: float, cfg: "CMoEPyramidStairsTerrainCfg") -> np.ndarray:
    step_height = 0.05 + 0.18 * difficulty         # 官方: 0.05 + 0.18*difficulty (论文 0.05-0.23m)
    if cfg.inverted:
        step_height = -step_height                 # 官方 idx2 "stairs up": step_height *= -1 (中心为坑)
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg)
    W, L = hf.shape
    # ---- 官方 humanoid_terrain.pyramid_stairs_terrain (带 border_size=0.5) 照搬 ----
    step_width = int(0.30 / hs)
    step_h = int(step_height / vs)
    platform_size = int(3.0 / hs)
    border_size = int(0.5 / hs)
    height = 0
    start_x, stop_x = border_size, W - border_size
    start_y, stop_y = border_size, L - border_size
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_h
        hf[start_x:stop_x, start_y:stop_y] = height
    _add_roughness(hf, difficulty, hs, vs)
    return hf


@configclass
class CMoEPyramidStairsTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_pyramid_stairs_terrain
    inverted: bool = False
    """True = 官方 "stairs up"(中心低, 出生在坑底向外爬升); False = "stairs down"(中心高)。"""


# ---------------------------------------------------------------------------- #
#  3) discrete —— 官方 idx4: isaacgym discrete_obstacles_terrain + roughness
# ---------------------------------------------------------------------------- #
@height_field_to_mesh
def cmoe_discrete_obstacles_terrain(difficulty: float, cfg: "CMoEDiscreteObstaclesTerrainCfg") -> np.ndarray:
    obstacle_height = 0.05 + difficulty * 0.1      # 官方: discrete_obstacles_height
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg)
    W, L = hf.shape
    # ---- isaacgym.terrain_utils.discrete_obstacles_terrain 照搬 ----
    max_h = int(obstacle_height / vs)
    min_size = int(1.0 / hs)                       # 官方: rectangle_min_size=1.
    max_size = int(2.0 / hs)                       # 官方: rectangle_max_size=2.
    platform_size = int(3.0 / hs)
    height_range = [-max_h, -max_h // 2, max_h // 2, max_h]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)
    for _ in range(20):                            # 官方: num_rectangles=20
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, W - width, 4))
        start_j = np.random.choice(range(0, L - length, 4))
        hf[start_i:start_i + width, start_j:start_j + length] = np.random.choice(height_range)
    x1 = (W - platform_size) // 2
    x2 = (W + platform_size) // 2
    y1 = (L - platform_size) // 2
    y2 = (L + platform_size) // 2
    hf[x1:x2, y1:y2] = 0
    _add_roughness(hf, difficulty, hs, vs)
    return hf


@configclass
class CMoEDiscreteObstaclesTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_discrete_obstacles_terrain


# ---------------------------------------------------------------------------- #
#  4) parkour gap —— 官方 idx5: parkour_gap_terrain(...) + roughness, 出生点 x=0.75
# ---------------------------------------------------------------------------- #
def cmoe_parkour_gap_terrain(difficulty: float, cfg: "CMoEParkourGapTerrainCfg"):
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg, standalone=True)
    W, L = hf.shape
    # 官方调用: parkour_gap_terrain(terrain, platform_len=1.0, num_gaps=4,
    #   gap_size=0.1+0.7*difficulty, gap_depth=[0.5,1.5], pad_height=0,
    #   x_range=[0.8,1.4], half_valid_width=[1-0.5d, 1.5-0.5d], use_half_valid_width=True)
    gap_size_m = 0.1 + 0.7 * difficulty                      # 论文 gap 0.1-0.8m
    hvw_range = (1.0 - 0.5 * difficulty, 1.5 - 0.5 * difficulty)
    mid_y = L // 2
    dis_y_min = round(-0.1 / hs)                             # y_range 默认 [-0.1, 0.1]
    dis_y_max = round(0.1 / hs)
    platform_len = round(1.0 / hs)
    gap_depth = -round(np.random.uniform(0.5, 1.5) / vs)
    half_valid_width = round(np.random.uniform(*hvw_range) / hs)
    hf[0:platform_len, :] = 0                                # platform_height=0
    # 2026-07-16 修复: 加 max(...,2)。hs=0.1 且 difficulty->0 时 round(0.1/0.1)=1,
    # 1//2=0 会让 hf[dis_x-0:dis_x+0] 成为空切片 -> 沟消失。保底 2px 保证沟存在
    # (difficulty=0 时沟宽 0.2m 而非官方 0.1m, 对课程学习无实质影响)。
    gap_size = max(round(gap_size_m / hs), 2)
    dis_x_min = round(0.8 / hs) + gap_size
    dis_x_max = round(1.4 / hs) + gap_size
    dis_x = platform_len
    last_dis_x = dis_x
    for _ in range(4):                                       # num_gaps=4
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        hf[dis_x - gap_size // 2: dis_x + gap_size // 2, :] = gap_depth
        # use_half_valid_width=True: 走廊两侧同样挖成 gap_depth
        hf[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
        hf[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth
        last_dis_x = dis_x
    _pad_edges(hf, hs, vs, pad_width=0.1, pad_height=0.0)
    _add_roughness(hf, difficulty, hs, vs)
    return _hf_to_mesh_with_parkour_origin(hf, cfg)


@configclass
class CMoEParkourGapTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_parkour_gap_terrain


# ---------------------------------------------------------------------------- #
#  5) parkour hurdle —— 官方 idx8: parkour_hurdle_terrain(...) + roughness
# ---------------------------------------------------------------------------- #
def cmoe_parkour_hurdle_terrain(difficulty: float, cfg: "CMoEParkourHurdleTerrainCfg"):
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg, standalone=True)
    W, L = hf.shape
    # 官方调用: parkour_hurdle_terrain(terrain, num_stones=4, stone_len=0.1+0.2*d,
    #   hurdle_height_range=[0.2d, 0.15+0.25d], x_range=[1.2,2], half_valid_width=[4,4.5], pad_height=0)
    # use_half_valid_width 默认 False -> 栏杆横贯整幅宽度。
    stone_len_m = 0.1 + 0.2 * difficulty
    hh_min_m, hh_max_m = 0.0 + 0.2 * difficulty, 0.15 + 0.25 * difficulty   # 论文 hurdle 0.2-0.4m
    dis_x_min = round(1.2 / hs)
    dis_x_max = round(2.0 / hs)
    hurdle_height_max = round(hh_max_m / vs)
    hurdle_height_min = round(hh_min_m / vs)
    platform_len = round(2.0 / hs)                           # platform_len 默认 2
    hf[0:platform_len, :] = 0
    # 2026-07-16 修复: 加 max(...,2), 理由同 parkour_gap —— hs=0.1 且低难度时
    # stone_len=1 会因 1//2=0 产生空切片, 栏杆消失。
    stone_len = max(round(stone_len_m / hs), 2)
    dis_x = platform_len
    for _ in range(4):                                       # num_stones=4
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        hf[dis_x - stone_len // 2: dis_x + stone_len // 2, :] = \
            np.random.randint(hurdle_height_min, hurdle_height_max)
    _pad_edges(hf, hs, vs, pad_width=0.1, pad_height=0.0)
    _add_roughness(hf, difficulty, hs, vs)
    return _hf_to_mesh_with_parkour_origin(hf, cfg)


@configclass
class CMoEParkourHurdleTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_parkour_hurdle_terrain


# ---------------------------------------------------------------------------- #
#  6) mix —— 官方 idx9: mix_obstacles_terrain(hurdle_height_range=[difficulty,100])
# ---------------------------------------------------------------------------- #
def cmoe_mix_obstacles_terrain(difficulty: float, cfg: "CMoEMixObstaclesTerrainCfg"):
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    # 官方该函数以 horizontal_scale=0.05 的**像素常量**硬编码水平布局。
    # 2026-07-16 修复: 不再 assert hs==0.05(那会禁止为修碰撞烘焙而降分辨率),
    # 改为把官方 0.05m/px 的像素常量按比例换算到当前 hs —— 布局的米制尺寸逐段等价。
    # 高度常量单位是 vertical_scale(不变), 无需换算。
    assert abs(vs - 0.005) < 1e-9, "cmoe_mix_obstacles_terrain 高度常量按 vertical_scale=0.005 硬编码"
    px = lambda p_official: round(p_official * 0.05 / hs)    # 官方像素 -> 当前网格像素
    hf = _alloc_hf(cfg, standalone=True)
    W, L = hf.shape
    mid_y = L // 2
    diff = difficulty * 1.1                                  # 官方: diff = hurdle_height_range[0]*1.1
    gap_depth = -np.random.randint(100, 300)                 # 0.5 - 1.5 m
    # ---- 官方硬编码布局照搬 (官方单位: 像素 x0.05m / 高度 x0.005m) ----
    hf[px(30):px(36), :] = round(30 * diff)                  # 台阶 1: 0.15*diff m
    hf[px(36):px(42), :] = round(60 * diff)
    hf[px(42):px(48), :] = round(90 * diff)
    hf[px(48):px(60), :] = round(120 * diff)                 # 平台: 0.6*diff m
    hf[px(60):px(72 - round(10 - diff * 10)), :] = gap_depth # 沟 1
    hf[px(72 - round(10 - diff * 10)):px(84), :] = round(120 * diff)
    hf[px(86):px(96), :] = round(96 * diff)
    hf[px(96):px(99), :] = round(170 * diff)                 # 栏杆: 0.85*diff m (相对台面 +0.37*diff)
    hf[px(99):px(111), :] = round(120 * diff)
    hf[px(111):px(123 - round(10 - diff * 10)), :] = gap_depth   # 沟 2
    hf[px(123 - round(10 - diff * 10)):px(140), :] = round(120 * diff)
    hf[px(140):px(160), :] = round(60 * diff)
    # 走廊: 中线 ±(官方 20px = 1m), 两侧挖空
    hf[:, mid_y + px(20):] = gap_depth
    hf[:, : mid_y - px(20)] = gap_depth
    _pad_edges(hf, hs, vs, pad_width=0.1, pad_height=0.0)
    _add_roughness(hf, difficulty, hs, vs)
    return _hf_to_mesh_with_parkour_origin(hf, cfg)


@configclass
class CMoEMixObstaclesTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_mix_obstacles_terrain


# ---------------------------------------------------------------------------- #
#  7) narrow stairs —— 官方 idx10: narrow_stairs_terrain(...) + roughness
# ---------------------------------------------------------------------------- #
def cmoe_narrow_stairs_terrain(difficulty: float, cfg: "CMoENarrowStairsTerrainCfg"):
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    hf = _alloc_hf(cfg, standalone=True)
    W, L = hf.shape
    # 官方调用: narrow_stairs_terrain(terrain, num_stones=24, step_height=0.25*d,
    #   x_range=[0.30,1.5], y_range=[-0.4,0.8], half_valid_width=[1-0.5d,1.5-0.5d], pad_height=0)
    num_stones = 24
    step_height_m = 0.0 + 0.25 * difficulty
    mid_y = L // 2
    dis_x_min = round(0.30 / hs)                             # 官方 rand_x = dis_x_min (固定步长 0.3m)
    step_height = round(step_height_m / vs)
    half_valid_width = round((1.0 - 0.5 * difficulty) / hs)  # 官方只取 half_valid_width[0]
    platform_len = round(2.5 / hs)                           # platform_len 默认 2.5
    hf[0:platform_len, :] = 0
    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    gap_depth = -np.random.randint(10, 300)                  # 两侧下陷 0.05 - 1.5 m
    for i in range(num_stones):
        rand_x = dis_x_min
        rand_y = 0
        if i < num_stones // 2 - 2:
            stair_height += step_height
        elif i > num_stones // 2 + 2:
            stair_height -= step_height
        hf[dis_x:dis_x + rand_x, :] = stair_height
        dis_x += rand_x
        hf[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
        hf[last_dis_x:dis_x, mid_y + rand_y + half_valid_width:] = gap_depth
        last_dis_x = dis_x
    _pad_edges(hf, hs, vs, pad_width=0.1, pad_height=0.0)
    _add_roughness(hf, difficulty, hs, vs)
    return _hf_to_mesh_with_parkour_origin(hf, cfg)


@configclass
class CMoENarrowStairsTerrainCfg(HfTerrainBaseCfg):
    function = cmoe_narrow_stairs_terrain


# ---------------------------------------------------------------------------- #
#  官方 G1CMoECfg.terrain 的 Isaac Lab 等价配置
# ---------------------------------------------------------------------------- #
# 官方: terrain_length/width=10, num_rows=10, num_cols=40, horizontal_scale=0.05,
#       vertical_scale=0.005, slope_treshold=1.5, border_size=25, curriculum=True。
# 列 -> 地形映射与官方 curiculum() 完全同式: sub_index = min(where(col/num_cols+0.001 < cumsum(prop)))。
# 差异(框架层): Isaac Lab 的行内难度 = (row + U(0,1))/num_rows (官方为 row/num_rows, 无抖动)。
CMOE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=25.0,             # 官方 border_size=25m (平地包边); 嫌大可减小, 不影响训练语义
    num_rows=10,
    # 官方 num_cols=40。Isaac Lab 的 trimesh 地形是**单个全局 prim**, 40 列 x 10 行 x
    # (201x201 顶点/块) = 3200 万三角, 导致 PhysX/UJITSO 烘焙失败(机器人穿透地面),
    # 见 IsaacLab issue #2323。2026-07-16 实测: 降到 20 列(hs=0.05 下 1600 万三角)
    # **仍然烘焙失败**(启动日志 "Mesh approximation with 0 triangles ... skipping
    # mesh /World/ground/terrain/mesh"), 真正的修复是把 horizontal_scale 提到 0.1
    # (见下)。20 列本身保留: 9 类地形全在、每类实例数减半、easy 组仍精确占 10%
    # (12 列会把 easy 组挤到 17%, 10 列会丢掉 rough_slope —— 都不能用)。
    num_cols=20,
    # 2026-07-16 修复: 0.05 -> 0.1。
    # 0.05 时全图 = 10行 x 20列 x (201x201 顶点/块) = 808 万顶点 / 1600 万三角形,
    # 单一静态 trimesh 超出 omni.physx/UJITSO 碰撞烘焙上限, 启动时报
    #   "[omni.physx.plugin] Mesh approximation with 0 triangles encountered,
    #    skipping mesh /World/ground/terrain/mesh!"
    # -> 地形只有视觉(甚至渲染也被显存预算跳过)、没有碰撞体, 机器人出生即穿地下沉。
    # 0.1 时 = 202 万顶点 / 400 万三角形, 与 Isaac Lab 自带 ROUGH_TERRAINS_CFG 同一
    # 数量级(该配置全网大量训练验证可正常烘焙)。vertical_scale 不变, 所有高度语义
    # (台阶高/沟深/栏杆高)完全不受影响, 只有水平边缘量化从 5cm 变为 10cm——低于
    # 策略高度图的采样分辨率(8cm 栅格), 对学习几乎无感。
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,          # 官方判据 Δh > 1.5*0.05 = 7.5cm 起竖直; hs=0.1 下等效阈值 = 0.75
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # !! 顺序 = terrain_types 序号, 2/3/4/5 需与官方 env_class 对齐, 勿改 !!
        "rough_slope_inv": CMoERoughSlopeTerrainCfg(proportion=0.05, inverted=True),   # 0 官方 idx1(负)
        "rough_slope": CMoERoughSlopeTerrainCfg(proportion=0.05, inverted=False),      # 1 官方 idx1(正)
        "stairs_up": CMoEPyramidStairsTerrainCfg(proportion=0.10, inverted=True),      # 2 官方 idx2
        "stairs_down": CMoEPyramidStairsTerrainCfg(proportion=0.10, inverted=False),   # 3 官方 idx3
        "discrete": CMoEDiscreteObstaclesTerrainCfg(proportion=0.10),                  # 4 官方 idx4
        "parkour_gap": CMoEParkourGapTerrainCfg(proportion=0.30),                      # 5 官方 idx5
        "parkour_hurdle": CMoEParkourHurdleTerrainCfg(proportion=0.10),                # 6 官方 idx8
        "mix": CMoEMixObstaclesTerrainCfg(proportion=0.10),                            # 7 官方 idx9
        "narrow_stairs": CMoENarrowStairsTerrainCfg(proportion=0.10),                  # 8 官方 idx10
    },
)
"""CMoE 官方 8 类训练地形 (10 行难度 x 40 列, 10m x 10m/块)。"""

# 官方 easy_terrain = plane(0) + rough slope(0.1) = 0.1 -> 前 4 列 (rough_slope_inv/rough_slope)。
# cmoe_reward.CMoEVelocityCommand 用下面这份名单来区分 easy/hard 指令范围。
CMOE_EASY_TERRAIN_NAMES = ["rough_slope_inv", "rough_slope"]
