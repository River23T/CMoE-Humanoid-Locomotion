# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CMoE paper-spec benchmark environment and in-memory episode recorder."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils import configclass

from humanoid_locomotion.tasks.velocity.dual_gate.terrains.config.cmoe_benchmark import (
    CMOE_BENCHMARK_TERRAINS_CFG,
)

from .cmoe_env_cfg import G1CMoEEnvCfg


class CMoEBenchmarkEpisodeRecorder(RecorderTerm):
    """Capture the first completed episode of every vectorized environment.

    The callback runs before Isaac Lab resets a terminated environment, so the
    final root position and termination flags are still available.  This avoids
    the common error of reading the already-reset pose after ``env.step``.
    """

    FAILURE_CODES = {
        "none": 0,
        "base_contact": 1,
        "bad_roll_pitch": 2,
        "gap_fall": 3,
        "other": 99,
    }

    def __init__(self, cfg: RecorderTermCfg, env):
        super().__init__(cfg, env)
        num_envs = self._env.num_envs
        device = self._env.device
        self.started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.start_x = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.final_distance = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.duration_s = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure_code = torch.full(
            (num_envs,), self.FAILURE_CODES["other"], dtype=torch.long, device=device
        )

    def _ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)
        if torch.is_tensor(env_ids):
            return env_ids.to(device=self._env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)

    def record_post_reset(self, env_ids: Sequence[int] | torch.Tensor | None):
        ids = self._ids(env_ids)
        first_start = ids[~self.started[ids]]
        if first_start.numel() > 0:
            robot = self._env.scene["robot"]
            self.start_x[first_start] = robot.data.root_pos_w[first_start, 0]
            self.started[first_start] = True
        return None, None

    def record_pre_reset(self, env_ids: Sequence[int] | torch.Tensor | None):
        ids = self._ids(env_ids)
        valid = ids[self.started[ids] & ~self.finished[ids]]
        if valid.numel() == 0:
            return None, None

        robot = self._env.scene["robot"]
        distance = robot.data.root_pos_w[valid, 0] - self.start_x[valid]
        self.final_distance[valid] = torch.clamp(distance, min=0.0)
        self.duration_s[valid] = (
            self._env.episode_length_buf[valid].to(torch.float32)
            * float(self._env.cfg.sim.dt)
            * int(self._env.cfg.decimation)
        )

        active_terms = tuple(self._env.termination_manager.active_terms)
        time_out = (
            self._env.termination_manager.get_term("time_out")[valid] > 0.5
            if "time_out" in active_terms
            else torch.zeros(valid.numel(), dtype=torch.bool, device=self._env.device)
        )

        failure = torch.zeros(valid.numel(), dtype=torch.bool, device=self._env.device)
        reason = torch.full(
            (valid.numel(),), self.FAILURE_CODES["other"], dtype=torch.long, device=self._env.device
        )
        for term_name in ("base_contact", "bad_roll_pitch", "gap_fall"):
            if term_name not in active_terms:
                continue
            triggered = self._env.termination_manager.get_term(term_name)[valid] > 0.5
            first_reason = triggered & ~failure
            reason[first_reason] = self.FAILURE_CODES[term_name]
            failure |= triggered

        # Any future non-timeout termination is also a failure, even if it is not
        # one of the three known CMoE terms above.
        for term_name in active_terms:
            if term_name in ("time_out", "base_contact", "bad_roll_pitch", "gap_fall"):
                continue
            failure |= self._env.termination_manager.get_term(term_name)[valid] > 0.5

        succeeded = time_out & ~failure
        reason[succeeded] = self.FAILURE_CODES["none"]
        self.success[valid] = succeeded
        self.failure_code[valid] = reason
        self.finished[valid] = True

        record = torch.stack(
            (
                self.final_distance[valid],
                self.duration_s[valid],
                self.success[valid].to(torch.float32),
                self.failure_code[valid].to(torch.float32),
            ),
            dim=-1,
        )
        return "cmoe_benchmark_episode", record


@configclass
class CMoEBenchmarkEpisodeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CMoEBenchmarkEpisodeRecorder


@configclass
class CMoEBenchmarkRecorderManagerCfg(RecorderManagerBaseCfg):
    benchmark = CMoEBenchmarkEpisodeRecorderCfg()


@configclass
class G1CMoEBenchmarkEnvCfg(G1CMoEEnvCfg):
    """Evaluation-only environment for paper-style success/distance metrics.

    Public-code evaluation profile:
    * 20 s episode and fixed 0.8 m/s forward command;
    * observation noise disabled;
    * payload randomization and external pushes disabled;
    * friction, gain, COM, and initial-joint randomization remain enabled, as in
      the public ``legged_gym/scripts/play.py``.  The benchmark script also
      offers ``--deterministic`` to disable these remaining randomizations.
    """

    recorders: CMoEBenchmarkRecorderManagerCfg = CMoEBenchmarkRecorderManagerCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 20.0
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator = CMOE_BENCHMARK_TERRAINS_CFG.replace()
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.curriculum = False
        # Benchmark 中沟槽地形的 key 是 "gap"，覆盖训练配置中的 "parkour_gap"。
        self.terminations.gap_fall.params["terrain_names"] = ["gap"]

        # No terrain curriculum during benchmark evaluation.
        self.curriculum.terrain_levels = None

        # Paper command: 0.8 m/s straight ahead for the whole 20 s episode.
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        for ranges in (
            self.commands.base_velocity.ranges,
            self.commands.base_velocity.easy_ranges,
            self.commands.base_velocity.hard_ranges,
        ):
            ranges.lin_vel_x = (0.8, 0.8)
            ranges.lin_vel_y = (0.0, 0.0)
            ranges.ang_vel_z = (0.0, 0.0)
            ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False

        # Match the public play profile: no observation noise, no payload, no push.
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False
        for group_name in ("actor_map", "critic_map"):
            term = getattr(self.observations, group_name).height_scanner
            term.params["height_noise"] = 0.0
            term.params["noise_level"] = 0.0
            term.params["staleness_prob"] = 0.0
            term.params["salt_pepper_points"] = 0

        self.events.payload = None
        self.events.push_robot = None
