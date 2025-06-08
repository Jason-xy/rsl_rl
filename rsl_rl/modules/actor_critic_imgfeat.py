# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.networks.feature_extractor import ImageEncoder, STImageEncoder, MLPEncoder

class ActorCriticImgFeat(ActorCritic):
    """Actor-critic model that processes observations containing image data.
    
    This model extracts features from image data in the observation vector.
    For multi-frame observations, images from all frames are stacked along the 
    channel dimension and processed together to produce a single fused feature vector.
    Non-image parts from all frames are concatenated separately.
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
        img_input_shape=(1, 64, 64),  # Single-channel image shape (channels, height, width)
        img_start_idx=0,  # Starting index of image data in flattened observation
        img_feat_dim=128,  # Dimension of extracted image features
        num_frames=1,  # Number of frames in each observation
        **kwargs
    ):
        assert num_actor_obs % num_frames == 0, "num_actor_obs must be divisible by num_frames"
        frame_length = num_actor_obs // num_frames

        # Calculate the size of flattened single-channel image
        single_img_size = img_input_shape[0] * img_input_shape[1] * img_input_shape[2]
        total_img_reduction = num_frames * single_img_size - img_feat_dim

        # Calculate the adjusted observation sizes after replacing stacked images with fused features
        adj_num_actor_obs = num_actor_obs - total_img_reduction
        adj_num_critic_obs = num_critic_obs - total_img_reduction

        # Create stacked image input shape for the encoder (k channels, H, W)
        stacked_img_shape = (num_frames * img_input_shape[0], img_input_shape[1], img_input_shape[2])

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
        
        # Store image parameters
        self.img_input_shape = img_input_shape
        self.stacked_img_shape = stacked_img_shape
        self.img_start_idx = img_start_idx
        self.single_img_size = single_img_size
        self.img_feat_dim = img_feat_dim
        self.num_frames = num_frames
        self.frame_length = frame_length
        
        # Create the image feature extractor with stacked input shape
        # self.img_feature_extractor = ImageEncoder(
        #     input_shape=stacked_img_shape,
        #     output_dim=img_feat_dim,
        # )
        # self.img_feature_extractor = STImageEncoder(
        #     input_shape=stacked_img_shape,
        #     output_dim=img_feat_dim,
        # )
        self.img_feature_extractor = MLPEncoder(
            input_shape=stacked_img_shape,
            output_dim=img_feat_dim,
            hidden_dims=[256, 256],
            activation=activation
        )

        print(f"Image feature extractor: {self.img_feature_extractor.__class__.__name__}")
        print(f"Number of frames: {self.num_frames}, Frame length: {self.frame_length}")
        print(f"Single image shape: {img_input_shape}, Stacked shape: {stacked_img_shape}")
        print(f"Original obs dim: {num_actor_obs}, Adjusted obs dim: {adj_num_actor_obs}")

    def _process_observations(self, observations):
        """Extract image features and reconstruct the observation vector.
        For multi-frame observations:
        1. Extract images from all frames and stack them along channel dimension
        2. Process stacked images with the encoder to get fused features
        3. Extract and concatenate non-image parts from all frames
        4. Concatenate fused image features with non-image features
        Args:
            observations (torch.Tensor): Input observations of shape (batch_size, obs_dim).
        Returns:
            torch.Tensor: Processed observations with stacked image data replaced by fused features.
        """
        batch_size = observations.size(0)

        # Lists to collect image and non-image parts
        all_images = []
        all_non_image_parts = []

        # Process each frame to extract images and non-image parts
        for frame_idx in range(self.num_frames):
            # Calculate frame start and end indices
            frame_start = frame_idx * self.frame_length
            frame_end = frame_start + self.frame_length

            # Extract current frame
            frame = observations[:, frame_start:frame_end]

            # Calculate image start and end indices within the frame
            img_start = self.img_start_idx
            img_end = img_start + self.single_img_size

            # Extract image data from the frame
            img_data_flat = frame[:, img_start:img_end]

            # Reshape image data to proper format (batch_size, channels, height, width)
            img_data = img_data_flat.view(
                batch_size,
                self.img_input_shape[0],
                self.img_input_shape[1],
                self.img_input_shape[2]
            )

            all_images.append(img_data)

            # Extract non-image parts (before and after image data)
            non_img_parts = [
                frame[:, :img_start],  # Part before image
                frame[:, img_end:]     # Part after image
            ]
            # Filter out empty tensors and concatenate
            non_img_parts = [part for part in non_img_parts if part.size(1) > 0]
            if non_img_parts:
                frame_non_img = torch.cat(non_img_parts, dim=1)
                all_non_image_parts.append(frame_non_img)

        # Stack all images along channel dimension (batch_size, k*channels, height, width)
        stacked_images = torch.cat(all_images, dim=1)
        # Extract fused image features from stacked images
        fused_img_features = self.img_feature_extractor(stacked_images)
        # Concatenate all non-image parts from all frames
        concatenated_non_img = torch.cat(all_non_image_parts, dim=1)
        # Concatenate fused image features with non-image features
        processed_obs = torch.cat([concatenated_non_img, fused_img_features], dim=1)

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
