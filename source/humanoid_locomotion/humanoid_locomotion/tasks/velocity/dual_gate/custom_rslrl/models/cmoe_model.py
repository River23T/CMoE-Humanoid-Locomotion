# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import MLP
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState
from humanoid_locomotion.tasks.velocity.dual_gate.custom_rslrl.modules import CNN1D


class CMoEModel(MLPModel):
    """CNN-based neural model.

    This model uses one or more convolutional neural network (CNN) encoders to process one or more 2D observation groups
    before passing the resulting latent to an MLP. Any 1D observation groups are directly concatenated with the CNN
    latent and passed to the MLP. 1D observations can be normalized before being passed to the MLP. The output of the
    model can be either deterministic or stochastic, in which case a distribution module is used to sample the outputs.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
        cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        """Initialize the CNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the CNN and MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            cnn_cfg: Configuration of the CNN encoder(s).
            cnns: CNN modules to use, e.g., for sharing CNNs between actor and critic. If None, new CNNs are created.
        """
        # Resolve observation groups and dimensions
        if obs[obs_set].dim() == 3:
            obs_current = obs.clone()
            obs_current[obs_set] = obs_current[obs_set][:, -1]
        else:
            obs_current = obs
        obs_groups_1d, obs_dim_1d = self._get_obs_dim(obs_current, obs_groups, obs_set)

        # Create or validate CNN encoders
        if cnns is not None:
            # Check compatibility if CNNs are provided
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("The 2D observations must be identical for all models sharing CNN encoders.")
            print("Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored.")
        else:
            if cnn_cfg is None:
                raise ValueError("CNN configurations must be provided if CNNs are not shared.")
            # Create a cnn config for each 2D observation group in case only one is provided
            if not all(isinstance(v, dict) for v in cnn_cfg.values()):
                cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
            # Check that the number of configs matches the number of observation groups
            if len(cnn_cfg) != len(self.obs_groups_2d):
                raise ValueError("The number of CNN configurations must match the number of 2D observation groups.")
            # Create CNNs for each 2D observation
            cnns = {}
            for idx, obs_group in enumerate(self.obs_groups_2d):
                cnns[obs_group] = CNN(
                    input_dim=self.obs_dims_2d[idx],
                    input_channels=self.obs_channels_2d[idx] - 2,
                    **cnn_cfg[obs_group],
                )

        # Initialize the parent MLP model
        super().__init__(
            obs_current,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        # Initialize the weights of the MLP
        self.mlp.init_weights(
            [2**0.5, None, 2**0.5, None, 2**0.5, None, 0.01]
        )

        # Register CNN encoders
        if isinstance(cnns, nn.ModuleDict):
            self.cnns = cnns
        else:
            self.cnns = nn.ModuleDict(cnns)
        # Initialize the weights of the CNN
        for cnn in self.cnns.values():
            cnn.init_weights()

        # Initialize velocity estimator
        if obs_set == "actor":
            self.cnn1d = nn.Sequential(
                nn.Conv1d(
                    in_channels=obs["actor"].shape[-1], out_channels=obs["actor"].shape[-1],
                    kernel_size=3, stride=1, padding=1, groups=obs["actor"].shape[-1]
                ),
                nn.Conv1d(in_channels=obs["actor"].shape[-1], out_channels=128, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.ELU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(num_groups=8, num_channels=128),
                nn.ELU(),
                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ELU(),
                nn.Flatten(1),
                nn.Linear(in_features=800, out_features=128),
                nn.ELU(),
            )
            # Initialize the weights of the 1D-CNN
            for idx, module in enumerate(self.cnn1d):
                if isinstance(module, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(module.weight)
                    torch.nn.init.zeros_(module.bias)  # type: ignore

            self.velocity_estimator = MLP(128, 3, (64, 32), activation)
            # Initialize the weights of the MLP
            self.velocity_estimator.init_weights(
                [2 ** 0.5, None, 2 ** 0.5, None, 1.0]
            )
            self.velocity = None

        if obs_set == "actor":
            self.linear = nn.Linear(obs_dim_1d + 3, 64)
        else:
            self.linear = nn.Linear(obs_dim_1d, 64)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

        self.position = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 16),
        )
        nn.init.orthogonal_(self.position[0].weight, gain=1.0)
        nn.init.orthogonal_(self.position[3].weight, gain=2 ** 0.5)
        nn.init.zeros_(self.position[0].bias)
        nn.init.zeros_(self.position[3].bias)

        self.q_norm = nn.LayerNorm(64)
        self.k_norm = nn.LayerNorm(64)
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=16, batch_first=True)
        self.o_norm = nn.LayerNorm(64)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by combining normalized 1D and CNN-encoded 2D observation groups."""
        # get history observation
        if obs["actor"].dim() == 3:
            obs_current = obs.copy()
            obs_current["actor"] = obs_current["actor"][:, -1]
        else:
            obs_current = obs
        # Concatenate 1D observation groups and normalize
        latent_1d = super().get_latent(obs_current)
        # get velocity
        if self.obs_groups[0] == "actor":
            velocity = self.get_velocity(obs).detach()
            latent_1d = torch.cat((velocity, latent_1d), dim=-1)
        latent_1d_encoder = self.linear(latent_1d).unsqueeze(1) # (N, 1, 64)
        # Process 2D observation groups with CNNs
        latent_cnn_list = [self.cnns[obs_group](obs[obs_group][:, 2:3, ...]) for obs_group in self.obs_groups_2d]
        latent_cnn = torch.cat(latent_cnn_list, dim=-1) # (N, 48, 13, 18)
        latent_cnn = latent_cnn.flatten(2).permute(0, 2, 1) # (N, 234, 48)
        latent_position = self.position(obs["actor_map"].flatten(2).permute(0, 2, 1))   # (N, 234, 16)
        latent_mapping = torch.cat([latent_cnn, latent_position], dim=-1)   # (N, 234, 64)
        query = self.q_norm(latent_1d_encoder)
        key = self.k_norm(latent_mapping)
        latent_mha, _ = self.mha(query, key, latent_mapping, need_weights=False)    # (N, 1, 64)
        latent_mha = self.o_norm(latent_mha).flatten(1) # (N, 64)
        # Concatenate 1D and CNN latents
        return torch.cat([latent_1d, latent_mha], dim=-1)   # (N, 128)

    def get_velocity(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ):
        """Build the base velocity by concatenating and normalizing selected observation groups."""
        # Normalize hitstory observation groups and normalize
        obs_history = (obs["actor"] - self.obs_normalizer._mean) / (self.obs_normalizer._std + self.obs_normalizer.eps)
        latent_history = self.cnn1d(obs_history.permute(0, 2, 1).contiguous())
        self.velocity = self.velocity_estimator(latent_history).float()
        return self.velocity

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCNNModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim_1d = 0
        obs_groups_1d = []
        obs_dims_2d = []
        obs_channels_2d = []
        obs_groups_2d = []

        # Iterate through active observation groups and separate 1D and 2D observations
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) == 4:  # B, C, H, W
                obs_groups_2d.append(obs_group)
                obs_dims_2d.append(obs[obs_group].shape[2:4])
                obs_channels_2d.append(obs[obs_group].shape[1])
            elif len(obs[obs_group].shape) == 2:  # B, C
                obs_groups_1d.append(obs_group)
                obs_dim_1d += obs[obs_group].shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}")

        if not obs_groups_2d:
            raise ValueError("No 2D observations are provided. If this is intentional, use the MLP model instead.")

        # Store active 2D observation groups and dimensions directly as attributes
        self.obs_dims_2d = obs_dims_2d
        self.obs_channels_2d = obs_channels_2d
        self.obs_groups_2d = obs_groups_2d
        # Return active 1D observation groups and dimension for parent class
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        if self.obs_groups[0] == "actor":
            return self.obs_dim + 64 + 3
        else:
            return self.obs_dim + 64


class _TorchCNNModel(nn.Module):
    """Exportable CNN model for JIT."""

    def __init__(self, model: CNNModel) -> None:
        """Create a TorchScript-friendly copy of a CNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxCNNModel(nn.Module):
    """Exportable CNN model for ONNX."""

    def __init__(self, model: CNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a CNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_1d = self.obs_normalizer(obs_1d)

        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))

        latent_cnn = torch.cat(latent_cnn_list, dim=-1)
        latent = torch.cat([latent_1d, latent_cnn], dim=-1)

        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
