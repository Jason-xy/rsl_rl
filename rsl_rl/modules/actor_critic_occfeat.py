# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic
from rsl_rl.networks.feature_extractor import Conv2DEncoder, Conv3DEncoder, MLPEncoder


class ActorCriticOccFeat(ActorCritic):
    """Actor-critic model that processes observations containing occupancy grid data.
    
    This model extracts features from occupancy grid data in the observation vector,
    then reconstructs the observation vector by replacing the raw occupancy grid with
    the extracted features before passing it to the standard actor-critic networks.
    
    Supports multi-frame observations by specifying num_frames and frame_length.
    """
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        occ_input_shape=(32, 32, 8),  # Original occupancy grid shape (width, height, depth)
        occ_start_idx=0,  # Starting index of occupancy grid in flattened observation
        occ_feat_dim=128,  # Dimension of extracted occupancy grid features
        type="conv2d",  # Type of feature encoder: "conv2d", "conv3d", or "mlp"
        num_frames=1,  # Number of frames in each observation
        **kwargs
    ):
        assert num_actor_obs % num_frames == 0, "num_actor_obs must be divisible by num_frames"
        frame_length = num_actor_obs // num_frames

        # Calculate the size of flattened occupancy grid
        occ_size = occ_input_shape[0] * occ_input_shape[1] * occ_input_shape[2]

        # Calculate total reduction in size
        total_reduction = num_frames * (occ_size - occ_feat_dim)

        # Calculate the adjusted observation sizes after replacing occupancy grid with features
        adj_num_actor_obs = num_actor_obs - total_reduction
        adj_num_critic_obs = num_critic_obs - total_reduction

        super().__init__(
            num_actor_obs=adj_num_actor_obs,
            num_critic_obs=adj_num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs
        )
        
        # Store occupancy grid parameters
        self.occ_input_shape = occ_input_shape
        self.occ_start_idx = occ_start_idx
        self.occ_size = occ_size
        self.occ_feat_dim = occ_feat_dim
        self.num_frames = num_frames
        self.frame_length = frame_length
        
        # Create the feature extractor based on type
        if type == "conv3d":
            self.occ_feature_extractor = Conv3DEncoder(
                input_shape=occ_input_shape,
                output_dim=occ_feat_dim
            )
        elif type == "mlp":
            self.occ_feature_extractor = MLPEncoder(
                input_dim=occ_size,
                output_dim=occ_feat_dim,
                hidden_dims=[512, 256],
                activation=activation
            )
        else:  # Default to conv2d
            self.occ_feature_extractor = Conv2DEncoder(
                input_shape=occ_input_shape,
                output_dim=occ_feat_dim
            )
        
        print(f"Occupancy grid feature extractor: {self.occ_feature_extractor.__class__.__name__}")
        print(f"Number of frames: {self.num_frames}, Frame length: {self.frame_length}")
        print(f"Original obs dim: {num_actor_obs}, Adjusted obs dim: {adj_num_actor_obs}")
    
    def _process_observations(self, observations):
        """Extract occupancy grid features and reconstruct the observation vector.
        
        Processes multi-frame observations by iterating through each frame,
        extracting the occupancy grid, and replacing it with features.
        
        Args:
            observations (torch.Tensor): Input observations of shape (batch_size, obs_dim).
            
        Returns:
            torch.Tensor: Processed observations with occupancy grid replaced by features.
        """
        batch_size = observations.size(0)
        processed_frames = []
        
        # Process each frame
        for frame_idx in range(self.num_frames):
            # Calculate frame start and end indices
            frame_start = frame_idx * self.frame_length
            frame_end = frame_start + self.frame_length
            
            # Extract current frame
            frame = observations[:, frame_start:frame_end]
            
            # Calculate occupancy grid start and end indices within the frame
            occ_start = self.occ_start_idx
            occ_end = occ_start + self.occ_size
            
            # Extract occupancy grid from the frame
            occ_grid_flat = frame[:, occ_start:occ_end]
            
            # Extract features based on encoder type
            if isinstance(self.occ_feature_extractor, MLPEncoder):
                # MLPEncoder expects flattened input
                occ_features = self.occ_feature_extractor(occ_grid_flat)
            else:
                # Conv2D and Conv3D encoders expect 3D input
                occ_grid = occ_grid_flat.view(
                    batch_size,
                    self.occ_input_shape[0],
                    self.occ_input_shape[1],
                    self.occ_input_shape[2]
                )
                occ_features = self.occ_feature_extractor(occ_grid)

            # Reconstruct frame by replacing occupancy grid with features
            processed_frame = torch.cat([
                frame[:, :occ_start],
                occ_features,
                frame[:, occ_end:]
            ], dim=1)

            # Add processed frame to list
            processed_frames.append(processed_frame)

        # Concatenate processed frames
        processed_obs = torch.cat(processed_frames, dim=1)

        return processed_obs

    def act(self, observations, **kwargs):
        processed_obs = self._process_observations(observations)
        return super().act(processed_obs, **kwargs)

    def act_inference(self, observations):
        processed_obs = self._process_observations(observations)
        return super().act_inference(processed_obs)

    def evaluate(self, critic_observations, **kwargs):
        processed_obs = self._process_observations(critic_observations)
        return super().evaluate(processed_obs, **kwargs)
