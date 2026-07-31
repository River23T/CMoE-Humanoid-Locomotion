from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class CMoESubstepDelayedJointPositionAction(JointPositionAction):
    """CMoE joint-position action with the official Isaac Gym delay semantics.

    Official behavior for decimation=4:

    delay=0: current, current, current, current
    delay=1: previous, current, current, current
    delay=2: previous, previous, current, current
    delay=3: previous, previous, previous, current

    One delay value is sampled for each environment at every policy step.
    All controlled joints in the same environment share that delay.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._decimation = int(env.cfg.decimation)

        if self._decimation <= 0:
            raise ValueError(
                f"Invalid decimation: {self._decimation}"
            )

        default_targets = self._get_default_targets()

        # Before the first policy action, both previous and current
        # targets must represent zero raw action:
        # default_joint_pos + scale * 0.
        self._processed_actions.copy_(default_targets)

        self._previous_processed_actions = (
            default_targets.clone()
        )

        self._applied_processed_actions = (
            default_targets.clone()
        )

        # Shape (num_envs, 1) intentionally broadcasts over
        # all 12 joints, matching the official shared delay.
        self._delay_steps = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.long,
            device=self.device,
        )

        self._substep_index = 0

    def _get_default_targets(self) -> torch.Tensor:
        """Return targets corresponding to zero raw action."""

        if isinstance(self._offset, torch.Tensor):
            return self._offset

        return torch.full_like(
            self._processed_actions,
            float(self._offset),
        )

    def process_actions(
        self,
        actions: torch.Tensor,
    ) -> None:
        """Process one new policy action and sample its delay."""

        # The processed action from the preceding policy step
        # becomes the official previous action target.
        self._previous_processed_actions.copy_(
            self._processed_actions
        )

        # Preserve Isaac Lab's normal processing:
        # raw * scale + default offset, followed by clipping.
        super().process_actions(actions)

        # Official torch.randint(0, decimation) gives
        # 0, 1, 2, or 3 when decimation is 4.
        sampled_delays = torch.randint(
            low=0,
            high=self._decimation,
            size=(self.num_envs, 1),
            dtype=torch.long,
            device=self.device,
        )

        self._delay_steps.copy_(sampled_delays)
        self._substep_index = 0

    def apply_actions(self) -> None:
        """Apply previous/current target for this physics substep."""

        use_current = (
            self._substep_index >= self._delay_steps
        )

        selected_targets = torch.where(
            use_current,
            self._processed_actions,
            self._previous_processed_actions,
        )

        self._applied_processed_actions.copy_(
            selected_targets
        )

        self._asset.set_joint_position_target(
            selected_targets,
            joint_ids=self._joint_ids,
        )

        self._substep_index += 1

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Reset delayed-action state for selected environments."""

        if env_ids is None:
            env_ids = slice(None)

        super().reset(env_ids)

        default_targets = self._get_default_targets()

        self._processed_actions[env_ids] = (
            default_targets[env_ids]
        )

        self._previous_processed_actions[env_ids] = (
            default_targets[env_ids]
        )

        self._applied_processed_actions[env_ids] = (
            default_targets[env_ids]
        )

        self._delay_steps[env_ids] = 0
        self._substep_index = 0

    @property
    def delay_steps(self) -> torch.Tensor:
        """Per-environment delay selected for the current policy step."""

        return self._delay_steps

    @property
    def previous_processed_actions(self) -> torch.Tensor:
        """Processed targets from the preceding policy step."""

        return self._previous_processed_actions

    @property
    def applied_processed_actions(self) -> torch.Tensor:
        """Targets selected for the latest physics substep."""

        return self._applied_processed_actions

    @property
    def substep_index(self) -> int:
        """Index of the next physics substep."""

        return self._substep_index
