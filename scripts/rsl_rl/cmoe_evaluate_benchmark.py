# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a CMoE checkpoint on one paper-spec benchmark terrain.

Run this script once per terrain.  It writes per-episode CSV, a JSON summary,
and appends one row to a global summary CSV.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate CMoE success rate and travel distance.")
parser.add_argument("--task", type=str, default="DualGate-CMoE-G1-Benchmark")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--terrain", type=str, required=True)
parser.add_argument("--difficulty", type=float, default=1.0)
parser.add_argument("--velocity", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="logs/cmoe_benchmark")
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Disable remaining gain/COM/friction/joint/reset randomization.",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs import multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import humanoid_locomotion.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.cmoe_benchmark import (
    CMOE_BENCHMARK_TERRAIN_NAMES,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


installed_version = metadata.version("rsl-rl-lib")


def _set_fixed_command(env_cfg, velocity: float) -> None:
    for ranges in (
        env_cfg.commands.base_velocity.ranges,
        env_cfg.commands.base_velocity.easy_ranges,
        env_cfg.commands.base_velocity.hard_ranges,
    ):
        ranges.lin_vel_x = (velocity, velocity)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        ranges.heading = (0.0, 0.0)


def _select_terrain(env_cfg, terrain_name: str, difficulty: float, num_envs: int) -> None:
    if terrain_name not in CMOE_BENCHMARK_TERRAIN_NAMES:
        choices = ", ".join(CMOE_BENCHMARK_TERRAIN_NAMES)
        raise ValueError(f"Unknown terrain '{terrain_name}'. Valid values: {choices}")
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError(f"difficulty must be in [0, 1], got {difficulty}")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    generator = env_cfg.scene.terrain.terrain_generator
    selected_cfg = copy.deepcopy(generator.sub_terrains[terrain_name])
    selected_cfg.proportion = 1.0
    generator.sub_terrains = {terrain_name: selected_cfg}
    generator.curriculum = False
    generator.difficulty_range = (difficulty, difficulty)

    # 18 m x 3 m tiles: use more columns than rows to keep the global mesh compact.
    rows = max(1, int(round(math.sqrt(num_envs / 6.0))))
    cols = int(math.ceil(num_envs / rows))
    generator.num_rows = rows
    generator.num_cols = cols
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.terrain.max_init_terrain_level = None


def _set_deterministic_profile(env_cfg) -> None:
    """Disable remaining public-play randomizations while keeping mean friction."""
    env_cfg.events.pd_gains = None
    env_cfg.events.base_com = None

    # Terrain friction is 0.8 and combine mode is average.  Robot friction 0.5
    # therefore gives the mean effective coefficient (0.8 + 0.5) / 2 = 0.65.
    if env_cfg.events.physics_material is not None:
        params = env_cfg.events.physics_material.params
        params["static_friction_range"] = (0.5, 0.5)
        params["dynamic_friction_range"] = (0.5, 0.5)
        params["num_buckets"] = 1

    if env_cfg.events.reset_robot_joints is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    reset = env_cfg.events.reset_base.params
    reset["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    reset["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }


def _write_results(
    output_root: Path,
    checkpoint: str,
    terrain: str,
    seed: int,
    difficulty: float,
    velocity: float,
    profile: str,
    recorder,
) -> dict:
    checkpoint_name = Path(checkpoint).stem
    difficulty_name = f"difficulty_{difficulty:g}"

    run_dir = (
        output_root
        / checkpoint_name
        / terrain
        / difficulty_name
        / profile
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    success = recorder.success.detach().cpu()
    distance = recorder.final_distance.detach().cpu()
    duration = recorder.duration_s.detach().cpu()
    failure_code = recorder.failure_code.detach().cpu()

    code_to_name = {
        0: "none",
        1: "base_contact",
        2: "bad_roll_pitch",
        3: "gap_fall",
        99: "other",
    }
    episode_csv = run_dir / "episodes.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "env_id",
                "success",
                "forward_distance_m",
                "duration_s",
                "failure_code",
                "failure_reason",
            ]
        )
        for env_id in range(success.numel()):
            code = int(failure_code[env_id].item())
            writer.writerow(
                [
                    env_id,
                    int(success[env_id].item()),
                    f"{float(distance[env_id]):.6f}",
                    f"{float(duration[env_id]):.6f}",
                    code,
                    code_to_name.get(code, "unknown"),
                ]
            )

    failure_counts = {
        name: int((failure_code == code).sum().item())
        for code, name in code_to_name.items()
        if code != 0
    }
    summary = {
        "checkpoint": os.path.abspath(checkpoint),
        "checkpoint_name": checkpoint_name,
        "terrain": terrain,
        "seed": seed,
        "difficulty": difficulty,
        "velocity_mps": velocity,
        "profile": profile,
        "num_episodes": int(success.numel()),
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "average_travel_distance_m": float(distance.mean().item()),
        "successful_only_distance_m": (
            float(distance[success].mean().item()) if bool(success.any()) else 0.0
        ),
        "average_duration_s": float(duration.mean().item()),
        "failure_counts": failure_counts,
        "episodes_csv": str(episode_csv),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    global_csv = output_root / "summary.csv"
    new_file = not global_csv.exists()
    with global_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(
                [
                    "checkpoint_name",
                    "terrain",
                    "seed",
                    "difficulty",
                    "velocity_mps",
                    "profile",
                    "num_episodes",
                    "success_rate",
                    "average_travel_distance_m",
                    "successful_only_distance_m",
                    "average_duration_s",
                    "base_contact",
                    "bad_roll_pitch",
                    "gap_fall",
                    "other",
                ]
            )
        writer.writerow(
            [
                checkpoint_name,
                terrain,
                seed,
                difficulty,
                velocity,
                profile,
                summary["num_episodes"],
                f"{summary['success_rate']:.6f}",
                f"{summary['average_travel_distance_m']:.6f}",
                f"{summary['successful_only_distance_m']:.6f}",
                f"{summary['average_duration_s']:.6f}",
                failure_counts["base_contact"],
                failure_counts["bad_roll_pitch"],
                failure_counts["gap_fall"],
                failure_counts["other"],
            ]
        )
    return summary


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.seed = args_cli.seed
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    _select_terrain(env_cfg, args_cli.terrain, args_cli.difficulty, args_cli.num_envs)
    _set_fixed_command(env_cfg, args_cli.velocity)
    if args_cli.deterministic:
        _set_deterministic_profile(env_cfg)

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.removesuffix("-Benchmark")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise FileNotFoundError("No published checkpoint is available for this task.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_checkpoint = Path(resume_path).stem
        video_difficulty = f"difficulty_{args_cli.difficulty:g}"
        video_profile = "deterministic" if args_cli.deterministic else "public_play"

        video_kwargs = {
            "video_folder": str(
                Path(args_cli.output_dir)
                / "videos"
                / video_checkpoint
                / args_cli.terrain
                / video_difficulty
                / video_profile
                / f"seed_{args_cli.seed}"
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    recorder = env.unwrapped.recorder_manager._terms["benchmark"]
    obs = env.get_observations()
    max_steps = int(env.unwrapped.max_episode_length) + 10
    for _ in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
        if bool(torch.all(recorder.finished)):
            break

    if not bool(torch.all(recorder.finished)):
        unfinished = torch.nonzero(~recorder.finished, as_tuple=False).flatten().tolist()
        env.close()
        raise RuntimeError(f"Benchmark did not finish for env ids: {unfinished[:20]}")

    summary = _write_results(
        output_root=Path(args_cli.output_dir),
        checkpoint=resume_path,
        terrain=args_cli.terrain,
        seed=args_cli.seed,
        difficulty=args_cli.difficulty,
        velocity=args_cli.velocity,
        profile="deterministic" if args_cli.deterministic else "public_play",
        recorder=recorder,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
