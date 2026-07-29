# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Paper-spec 3 m x 18 m benchmark terrains for CMoE.

Important limitation
--------------------
The public CMoE repository does not contain the private terrain generator used
for Table III.  This file therefore implements the geometry stated in the
paper (runway size and obstacle ranges), but it is not byte-for-byte identical
to the authors' internal benchmark.

Coordinate convention: x is the forward direction (18 m), y is the runway
width (3 m), and the robot starts near x=1 m on the runway centreline.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

_RUNWAY_LENGTH = 18.0
_RUNWAY_WIDTH = 3.0
_BASE_DEPTH = 0.50
_START_X = 1.0


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = (x, y, z)
    return transform


def _solid_segment(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    top_z: float = 0.0,
    base_z: float = -_BASE_DEPTH,
) -> trimesh.Trimesh:
    """Create a solid rectangular runway segment whose upper surface is top_z."""
    if x1 <= x0 or y1 <= y0 or top_z <= base_z:
        raise ValueError(
            f"Invalid segment: x=({x0}, {x1}), y=({y0}, {y1}), z=({base_z}, {top_z})"
        )
    extents = (x1 - x0, y1 - y0, top_z - base_z)
    center = (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (top_z + base_z))
    return trimesh.creation.box(extents=extents, transform=_translation(*center))


def _obstacle_box(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    height: float,
    floor_z: float = 0.0,
) -> trimesh.Trimesh:
    """Create a box obstacle sitting on floor_z."""
    return trimesh.creation.box(
        extents=(length, width, height),
        transform=_translation(center_x, center_y, floor_z + 0.5 * height),
    )


def _ramp_segment(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    base_z: float = -_BASE_DEPTH,
) -> trimesh.Trimesh:
    """Create a closed solid with a planar top ramping from z0 to z1."""
    vertices = np.asarray(
        [
            [x0, y0, base_z],
            [x1, y0, base_z],
            [x1, y1, base_z],
            [x0, y1, base_z],
            [x0, y0, z0],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],       # bottom
            [4, 5, 6], [4, 6, 7],       # top
            [0, 1, 5], [0, 5, 4],       # y = y0
            [3, 7, 6], [3, 6, 2],       # y = y1
            [0, 4, 7], [0, 7, 3],       # x = x0
            [1, 2, 6], [1, 6, 5],       # x = x1
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _origin(width: float = _RUNWAY_WIDTH, height: float = 0.0) -> np.ndarray:
    return np.asarray([_START_X, 0.5 * width, height], dtype=np.float64)


def _check_size(cfg: SubTerrainBaseCfg) -> tuple[float, float]:
    length, width = float(cfg.size[0]), float(cfg.size[1])
    if not math.isclose(length, _RUNWAY_LENGTH, abs_tol=1e-6):
        raise ValueError(f"CMoE benchmark length must be 18 m, got {length} m")
    if not math.isclose(width, _RUNWAY_WIDTH, abs_tol=1e-6):
        raise ValueError(f"CMoE benchmark width must be 3 m, got {width} m")
    return length, width


def cmoe_benchmark_slope(
    difficulty: float, cfg: "CMoEBenchmarkSlopeCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    angle_deg = cfg.angle_range_deg[0] + difficulty * (
        cfg.angle_range_deg[1] - cfg.angle_range_deg[0]
    )
    ramp_length = 5.0
    plateau_length = 4.0
    height = math.tan(math.radians(angle_deg)) * ramp_length
    x1 = cfg.start_platform_length
    x2 = x1 + ramp_length
    x3 = x2 + plateau_length
    x4 = x3 + ramp_length
    meshes = [
        _solid_segment(0.0, x1, 0.0, width, 0.0),
        _ramp_segment(x1, x2, 0.0, width, 0.0, height),
        _solid_segment(x2, x3, 0.0, width, height),
        _ramp_segment(x3, x4, 0.0, width, height, 0.0),
        _solid_segment(x4, length, 0.0, width, 0.0),
    ]
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkSlopeCfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_slope
    angle_range_deg: tuple[float, float] = (0.0, 20.0)
    start_platform_length: float = 2.0


def cmoe_benchmark_stairs(
    difficulty: float, cfg: "CMoEBenchmarkStairsCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    total_height = cfg.num_steps * step_height
    meshes: list[trimesh.Trimesh] = []
    x = cfg.start_platform_length

    if cfg.descending:
        meshes.append(_solid_segment(0.0, x, 0.0, width, total_height))
        for index in range(cfg.num_steps):
            top_z = total_height - (index + 1) * step_height
            meshes.append(_solid_segment(x, x + cfg.step_depth, 0.0, width, top_z))
            x += cfg.step_depth
        meshes.append(_solid_segment(x, length, 0.0, width, 0.0))
        return meshes, _origin(width, total_height)

    meshes.append(_solid_segment(0.0, x, 0.0, width, 0.0))
    for index in range(cfg.num_steps):
        top_z = (index + 1) * step_height
        meshes.append(_solid_segment(x, x + cfg.step_depth, 0.0, width, top_z))
        x += cfg.step_depth
    meshes.append(_solid_segment(x, length, 0.0, width, total_height))
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkStairsCfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_stairs
    descending: bool = False
    step_height_range: tuple[float, float] = (0.05, 0.23)
    num_steps: int = 8
    step_depth: float = 0.50
    start_platform_length: float = 2.0


def cmoe_benchmark_discrete(
    difficulty: float, cfg: "CMoEBenchmarkDiscreteCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    max_height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    meshes: list[trimesh.Trimesh] = [_solid_segment(0.0, length, 0.0, width, 0.0)]
    xs = (3.0, 4.2, 5.5, 6.8, 8.2, 9.5, 10.8, 12.0, 13.4, 14.7, 16.0)
    ys = (0.75, 1.50, 2.25, 1.05, 1.95, 0.70, 1.55, 2.30, 1.10, 1.90, 1.45)
    for index, (x, y) in enumerate(zip(xs, ys, strict=True)):
        height_scale = 0.55 + 0.45 * ((index % 4) / 3.0)
        meshes.append(
            _obstacle_box(
                center_x=x,
                center_y=y,
                length=cfg.obstacle_length,
                width=cfg.obstacle_width,
                height=max_height * height_scale,
            )
        )
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkDiscreteCfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_discrete
    height_range: tuple[float, float] = (0.10, 0.20)
    obstacle_length: float = 0.65
    obstacle_width: float = 0.65


def _segments_with_full_width_gaps(
    length: float, width: float, gaps: list[tuple[float, float]]
) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    cursor = 0.0
    for gap_start, gap_end in gaps:
        if gap_start > cursor:
            meshes.append(_solid_segment(cursor, gap_start, 0.0, width, 0.0))
        cursor = gap_end
    if cursor < length:
        meshes.append(_solid_segment(cursor, length, 0.0, width, 0.0))
    return meshes


def cmoe_benchmark_gap(
    difficulty: float, cfg: "CMoEBenchmarkGapCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    gap_width = cfg.gap_width_range[0] + difficulty * (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    )
    gap_centres = (3.4, 6.8, 10.2, 13.6)
    gaps = [(centre - 0.5 * gap_width, centre + 0.5 * gap_width) for centre in gap_centres]
    return _segments_with_full_width_gaps(length, width, gaps), _origin(width)


@configclass
class CMoEBenchmarkGapCfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_gap
    gap_width_range: tuple[float, float] = (0.10, 0.80)


def cmoe_benchmark_hurdle(
    difficulty: float, cfg: "CMoEBenchmarkHurdleCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    hurdle_height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    hurdle_thickness = cfg.width_range[0] + difficulty * (cfg.width_range[1] - cfg.width_range[0])
    meshes: list[trimesh.Trimesh] = [_solid_segment(0.0, length, 0.0, width, 0.0)]
    for x in (3.5, 7.0, 10.5, 14.0):
        meshes.append(
            _obstacle_box(
                center_x=x,
                center_y=0.5 * width,
                length=hurdle_thickness,
                width=width,
                height=hurdle_height,
            )
        )
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkHurdleCfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_hurdle
    height_range: tuple[float, float] = (0.20, 0.40)
    width_range: tuple[float, float] = (0.10, 0.30)


def cmoe_benchmark_mix1(
    difficulty: float, cfg: "CMoEBenchmarkMix1Cfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    gap_width = cfg.gap_width_range[0] + difficulty * (
        cfg.gap_width_range[1] - cfg.gap_width_range[0]
    )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    meshes: list[trimesh.Trimesh] = []

    first_gap = (3.0, 3.0 + gap_width)
    second_gap = (11.0, 11.0 + gap_width)
    meshes.append(_solid_segment(0.0, first_gap[0], 0.0, width, 0.0))
    meshes.append(_solid_segment(first_gap[1], 5.0, 0.0, width, 0.0))

    x = 5.0
    for index in range(3):
        meshes.append(_solid_segment(x, x + 0.5, 0.0, width, (index + 1) * step_height))
        x += 0.5
    meshes.append(_solid_segment(x, 8.0, 0.0, width, 3.0 * step_height))

    x = 8.0
    for top_z in (2.0 * step_height, step_height, 0.0):
        meshes.append(_solid_segment(x, x + 0.5, 0.0, width, top_z))
        x += 0.5
    meshes.append(_solid_segment(x, second_gap[0], 0.0, width, 0.0))
    meshes.append(_solid_segment(second_gap[1], length, 0.0, width, 0.0))
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkMix1Cfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_mix1
    gap_width_range: tuple[float, float] = (0.10, 0.80)
    step_height_range: tuple[float, float] = (0.10, 0.15)


def cmoe_benchmark_mix2(
    difficulty: float, cfg: "CMoEBenchmarkMix2Cfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    length, width = _check_size(cfg)
    bridge_width = cfg.bridge_width_range[1] - difficulty * (
        cfg.bridge_width_range[1] - cfg.bridge_width_range[0]
    )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    y0 = 0.5 * (width - bridge_width)
    y1 = y0 + bridge_width
    meshes: list[trimesh.Trimesh] = [
        _solid_segment(0.0, 3.0, 0.0, width, 0.0),
        _solid_segment(3.0, 5.0, y0, y1, 0.0),
        _solid_segment(5.0, 7.0, y0, y1, step_height),
        _solid_segment(7.0, 9.0, y0, y1, 2.0 * step_height),
        _solid_segment(9.0, 11.0, y0, y1, step_height),
        _solid_segment(11.0, 15.0, y0, y1, 0.0),
        _solid_segment(15.0, length, 0.0, width, 0.0),
    ]
    return meshes, _origin(width)


@configclass
class CMoEBenchmarkMix2Cfg(SubTerrainBaseCfg):
    function = cmoe_benchmark_mix2
    bridge_width_range: tuple[float, float] = (0.50, 1.00)
    step_height_range: tuple[float, float] = (0.10, 0.25)


CMOE_BENCHMARK_TERRAIN_NAMES = (
    "slope",
    "stairs_up",
    "stairs_down",
    "discrete",
    "gap",
    "hurdle",
    "mix1",
    "mix2",
)


CMOE_BENCHMARK_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(_RUNWAY_LENGTH, _RUNWAY_WIDTH),
    border_width=0.0,
    num_rows=1,
    num_cols=len(CMOE_BENCHMARK_TERRAIN_NAMES),
    difficulty_range=(1.0, 1.0),
    use_cache=False,
    sub_terrains={
        "slope": CMoEBenchmarkSlopeCfg(proportion=1.0),
        "stairs_up": CMoEBenchmarkStairsCfg(proportion=1.0, descending=False),
        "stairs_down": CMoEBenchmarkStairsCfg(proportion=1.0, descending=True),
        "discrete": CMoEBenchmarkDiscreteCfg(proportion=1.0),
        "gap": CMoEBenchmarkGapCfg(proportion=1.0),
        "hurdle": CMoEBenchmarkHurdleCfg(proportion=1.0),
        "mix1": CMoEBenchmarkMix1Cfg(proportion=1.0),
        "mix2": CMoEBenchmarkMix2Cfg(proportion=1.0),
    },
)
"""Eight 3 m x 18 m paper-spec runways used only for evaluation."""
