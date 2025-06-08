# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Feature extractors for occupancy grid inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FeatureEncoder(nn.Module):
    """Base class for feature extractors that process 3D occupancy grid inputs."""
    
    def __init__(self):
        super(FeatureEncoder, self).__init__()
    
    def forward(self, x):
        """Forward pass through the feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor representing occupancy grid.
            
        Returns:
            torch.Tensor: Extracted features.
        """
        raise NotImplementedError


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for parameter efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class Conv2DEncoder(FeatureEncoder):
    """Lightweight feature extractor for 3D occupancy grids using 2D convolutions.
    
    Optimized for edge deployment with ~25k parameters while preserving spatial features.
    """
    
    def __init__(self, input_shape=(32, 32, 8), output_dim=128):
        """Initialize the lightweight 2D convolutional feature extractor.
        
        Args:
            input_shape (tuple): Shape of the input occupancy grid (width, height, depth).
            output_dim (int): Dimension of the output feature vector.
        """
        super(Conv2DEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # Treat depth as channels for 2D convolution
        in_channels = input_shape[2]
        
        # Lightweight convolutional layers with reduced channels
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = DepthwiseSeparableConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv2d(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv4 = DepthwiseSeparableConv2d(48, 64, kernel_size=3, stride=2, padding=1)
        
        # Global average pooling to preserve spatial information while reducing parameters
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Compact fully connected layers
        self.fc = nn.Linear(64, output_dim, bias=False)
        self.ln = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """Forward pass through the lightweight 2D convolutional feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, width, height, depth).
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, output_dim).
        """
        # Reshape input to (batch_size, depth, width, height) for 2D convolution
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        
        # Apply convolutions with efficient design
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Global average pooling to reduce spatial dimensions while preserving features
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Final projection
        x = self.fc(x)
        x = self.ln(x)
        
        return x


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution for parameter efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class Conv3DEncoder(FeatureEncoder):
    """Lightweight feature extractor for 3D occupancy grids using 3D convolutions.
    
    Optimized for edge deployment with ~25k parameters while preserving 3D spatial features.
    """
    
    def __init__(self, input_shape=(32, 32, 8), output_dim=128):
        """Initialize the lightweight 3D convolutional feature extractor.
        
        Args:
            input_shape (tuple): Shape of the input occupancy grid (width, height, depth).
            output_dim (int): Dimension of the output feature vector.
        """
        super(Conv3DEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # Lightweight 3D convolutional layers with reduced channels
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(8)
        
        self.conv2 = DepthwiseSeparableConv3d(8, 16, kernel_size=3, stride=(2, 2, 1), padding=1)
        self.conv3 = DepthwiseSeparableConv3d(16, 32, kernel_size=3, stride=(2, 2, 1), padding=1)
        self.conv4 = DepthwiseSeparableConv3d(32, 48, kernel_size=3, stride=(2, 2, 2), padding=1)
        
        # 3D global average pooling to preserve spatial information
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Compact fully connected layer
        self.fc = nn.Linear(48, output_dim, bias=False)
        self.ln = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """Forward pass through the lightweight 3D convolutional feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, width, height, depth).
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, output_dim).
        """
        # Add channel dimension: (batch_size, width, height, depth) -> (batch_size, 1, width, height, depth)
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        
        # Apply 3D convolutions with efficient design
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Global average pooling to reduce dimensions while preserving 3D spatial features
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Final projection
        x = self.fc(x)
        x = self.ln(x)
        
        return x

class MLPEncoder(nn.Module):
    """Multi-layer perceptron (MLP) encoder for feature extraction."""
    
    def __init__(self, input_shape, output_dim, hidden_dims=[256, 256], activation="elu"):
        """Initialize the MLP encoder.
        
        Args:
            input_dim (int or tuple): Dimension of the input features. If tuple, will be flattened.
            output_dim (int): Dimension of the output features.
            hidden_dims (list): List of hidden layer dimensions.
            activation (str): Activation function to use.
        """
        super(MLPEncoder, self).__init__()
        
        # Handle multi-dimensional input
        if isinstance(input_shape, (tuple, list)):
            self.input_shape = input_shape
            self.flattened_input_dim = int(torch.prod(torch.tensor(input_shape)))
        else:
            self.input_shape = (input_shape,)
            self.flattened_input_dim = input_shape
        
        # Get activation function
        if activation == "elu":
            act_fn = nn.ELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ELU  # Default to ELU
        
        layers = []
        in_dim = self.flattened_input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the MLP encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
                             Will be automatically flattened to (batch_size, flattened_input_dim).
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)
        
        # Flatten input while preserving batch dimension
        if len(x.shape) > 2:
            x = x.view(batch_size, -1)
        elif len(x.shape) == 1:
            # Handle case where batch dimension might be missing
            x = x.unsqueeze(0)
        
        # Ensure the flattened dimension matches expected input dimension
        if x.size(1) != self.flattened_input_dim:
            raise ValueError(f"Expected flattened input dimension {self.flattened_input_dim}, "
                           f"but got {x.size(1)}. Input shape: {x.shape}")
        
        return self.mlp(x)


class ImageEncoder(nn.Module):
    """Optimized CNN encoder for low-resolution multi-frame image feature extraction.
    
    Designed specifically for 8x8 multi-frame inputs with efficient temporal fusion.
    Uses minimal downsampling and lightweight operations to preserve spatial information.
    """
    
    def __init__(self, input_shape=(3, 8, 8), output_dim=128):
        """Initialize the optimized image encoder for low-resolution inputs.
        
        Args:
            input_shape (tuple): Shape of the input image (channels, height, width).
                                For multi-frame inputs, channels = num_frames * original_channels
            output_dim (int): Dimension of the output feature vector.
        """
        super(ImageEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        channels, height, width = input_shape
        
        # Detect if this is multi-frame input (more than 3 channels typically indicates stacked frames)
        self.is_multiframe = channels > 3
        self.num_frames = channels if channels <= 16 else channels // 3  # Estimate frame count
        
        if self.is_multiframe and height == 8 and width == 8:
            # Optimized architecture for 8x8 multi-frame inputs
            
            # First layer: frame-wise convolution with groups to process each frame separately
            # then fuse with 1x1 conv
            self.frame_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, 
                                      groups=min(channels, 8), bias=False)  # Grouped conv for frame-wise processing
            self.frame_fusion = nn.Conv2d(channels, 16, kernel_size=1, bias=False)  # Temporal fusion
            
            # Spatial feature extraction with minimal downsampling to preserve 8x8 spatial info
            self.spatial_conv1 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=False)
            self.spatial_conv2 = nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False)  # 8x8 -> 4x4
            
            # Final spatial processing
            self.final_conv = nn.Conv2d(32, 40, kernel_size=3, stride=1, padding=1, bias=False)  # Keep 4x4
            
            # Use average pooling instead of adaptive pooling for deterministic behavior
            self.pool = nn.AvgPool2d(4)  # 4x4 -> 1x1
            
            # Compact projection without LayerNorm for efficiency
            self.fc = nn.Linear(40, output_dim, bias=True)
            
        else:
            # Fallback architecture for other input sizes or single frame
            reduction_factor = max(1, (height * width) // 16)  # Adaptive reduction
            
            self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, 
                                 stride=2 if height > 16 else 1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, 
                                 stride=2 if height > 8 else 1, padding=1, bias=False)
            self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(48, output_dim, bias=True)

    def forward(self, x):
        """Forward pass through the optimized image encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, output_dim).
        """
        batch_size = x.size(0)
        _, height, width = x.shape[1:]
        
        if self.is_multiframe and height == 8 and width == 8:
            # Optimized path for 8x8 multi-frame inputs
            
            # Frame-wise processing and temporal fusion
            x = F.relu(self.frame_conv(x))      # Process each frame separately
            x = F.relu(self.frame_fusion(x))    # Fuse temporal information
            
            # Spatial feature extraction with minimal information loss
            x = F.relu(self.spatial_conv1(x))   # 8x8 -> 8x8, extract spatial patterns
            x = F.relu(self.spatial_conv2(x))   # 8x8 -> 4x4, controlled downsampling
            x = F.relu(self.final_conv(x))      # 4x4 -> 4x4, refine features
            
            # Final pooling and projection
            x = self.pool(x)                     # 4x4 -> 1x1
            x = x.view(batch_size, -1)
            x = self.fc(x)
            
        else:
            # Fallback path for other configurations
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            x = self.global_pool(x)
            x = x.view(batch_size, -1)
            x = self.fc(x)
        
        return x

class SpatioTemporalBlock(nn.Module):
    """Optimized spatio-temporal attention block with reduced reshape operations.
    
    Implements efficient factorized attention with memory optimizations
    and reduced tensor transformations for better performance.
    """
    
    def __init__(self, embed_dim=32, num_heads=2, F=5, H=8, W=8, dropout=0.1):
        """Initialize the optimized spatio-temporal attention block.
        
        Args:
            embed_dim (int): Embedding dimension for attention computation.
            num_heads (int): Number of attention heads.
            F (int): Number of frames (temporal dimension).
            H (int): Height of each frame.
            W (int): Width of each frame.
            dropout (float): Dropout rate for attention.
        """
        super(SpatioTemporalBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.F, self.H, self.W = F, H, W
        
        # Use smaller embedding for attention to reduce memory, but not overly constrained
        attn_dim = max(32, embed_dim // 2)  # Ensure minimum viable attention dimension
        
        # Fix attention head calculation - don't be overly constrained by attn_dim//8
        actual_heads = min(num_heads, max(1, attn_dim // 16))  # More reasonable constraint
        
        # Project to smaller dimension for attention computation
        self.input_proj = nn.Linear(embed_dim, attn_dim, bias=False)
        self.output_proj = nn.Linear(attn_dim, embed_dim, bias=False)
        
        # Optimized attention with better head calculation
        self.spatial_mha = nn.MultiheadAttention(
            attn_dim, actual_heads, dropout=dropout, batch_first=True
        )
        
        self.temporal_mha = nn.MultiheadAttention(
            attn_dim, actual_heads, dropout=dropout, batch_first=True
        )
        
        # Replace GroupNorm with LayerNorm for better performance
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)  # Add norm before FFN
        
        # Simplified feed-forward with memory optimization
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def _fused_spatial_attention(self, x):
        """Optimized spatial attention with reduced reshape operations."""
        B, C, F, H, W = x.shape
        
        # Single reshape operation for spatial attention
        x_spatial = x.permute(0, 2, 3, 4, 1).contiguous().view(B * F, H * W, C)
        x_spatial_proj = self.input_proj(x_spatial)
        
        # Apply spatial attention
        x_spatial_attn, _ = self.spatial_mha(x_spatial_proj, x_spatial_proj, x_spatial_proj)
        
        # Project back and add residual in one operation
        x_spatial_out = self.output_proj(x_spatial_attn)
        x_spatial = x_spatial + x_spatial_out
        
        # Reshape back to original format with proper contiguous handling
        return x_spatial.view(B, F, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    
    def _fused_temporal_attention(self, x):
        """Optimized temporal attention with reduced reshape operations."""
        B, C, F, H, W = x.shape
        
        # Single reshape operation for temporal attention
        x_temporal = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, F, C)
        x_temporal_proj = self.input_proj(x_temporal)
        
        # Apply temporal attention
        x_temporal_attn, _ = self.temporal_mha(x_temporal_proj, x_temporal_proj, x_temporal_proj)
        
        # Project back and add residual in one operation
        x_temporal_out = self.output_proj(x_temporal_attn)
        x_temporal = x_temporal + x_temporal_out
        
        # Reshape back to original format with proper contiguous handling
        return x_temporal.view(B, H, W, F, C).permute(0, 4, 3, 1, 2).contiguous()
    
    def forward(self, x):
        """Forward pass with direct computation for optimal performance.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, embed_dim, F, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, embed_dim, F, H, W).
        """
        # Direct computation without checkpointing for simplicity
        x_spatial = self._fused_spatial_attention(x)
        x_flat = x_spatial.reshape(-1, self.embed_dim)  # Use reshape instead of view
        x_normed = self.norm1(x_flat).view_as(x_spatial)
        
        x_temporal = self._fused_temporal_attention(x_normed)
        x_flat = x_temporal.reshape(-1, self.embed_dim)  # Use reshape instead of view
        x_normed = self.norm2(x_flat).view_as(x_temporal)
        
        # Feed-forward with pre-normalization
        identity = x_normed
        x_flat = x_normed.reshape(-1, self.embed_dim)  # Use reshape instead of view
        x_pre_ffn = self.norm3(x_flat)  # Pre-norm for stability
        x_ffn = self.ffn(x_pre_ffn)
        x = (x_ffn + x_flat).view_as(identity)
        
        return x


class STImageEncoder(nn.Module):
    """Optimized Spatio-Temporal Image Encoder for multi-frame feature extraction.
    
    Optimized for performance with reduced overhead in tensor operations
    while maintaining representation quality.
    """
    
    def __init__(self, input_shape=(5, 8, 8), output_dim=128, embed_dim=24, num_heads=2,
                 dropout=0.1, use_mixed_precision=True):
        """Initialize the optimized spatio-temporal image encoder.
        
        Args:
            input_shape (tuple): Shape of the input (frames, height, width).
            output_dim (int): Dimension of the output feature vector.
            embed_dim (int): Embedding dimension (reduced for memory efficiency).
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            use_mixed_precision (bool): Whether to use mixed precision training.
        """
        super(STImageEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.use_mixed_precision = use_mixed_precision
        
        F, H, W = input_shape
        
        # Lightweight initial feature extraction with BatchNorm
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(embed_dim // 2),  # Use BatchNorm for better performance
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        
        # Optimized spatio-temporal attention block
        self.st_block = SpatioTemporalBlock(
            embed_dim, num_heads, F, H, W, dropout
        )
        
        # Memory-efficient pooling and projection
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 2, 2))  # Reduced pooling
        self.final_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Lightweight final projection with improved normalization
        self.final_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),  # Add normalization before projection
            nn.Linear(embed_dim, output_dim, bias=False),
            nn.GELU()
        )
        
        # Optional lightweight reconstruction head (disabled by default for memory)
        self.use_reconstruction = False
        if self.use_reconstruction:
            self.reconstruction_head = nn.Sequential(
                nn.ConvTranspose3d(embed_dim, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )
    
    def configure_performance(self, use_mixed_precision=None):
        """Configure performance settings dynamically.
        
        Args:
            use_mixed_precision (bool, optional): Enable/disable mixed precision.
        """
        if use_mixed_precision is not None:
            self.use_mixed_precision = use_mixed_precision
    
    def forward(self, x, return_reconstruction=False):
        """Optimized forward pass with reduced memory overhead.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, F, H, W) or (B, F*C, H, W).
            return_reconstruction (bool): Whether to return reconstruction.
            
        Returns:
            torch.Tensor: Extracted features of shape (B, output_dim).
            Optional[torch.Tensor]: Reconstructed input if return_reconstruction=True.
        """
        # For tracing compatibility, avoid dynamic control flow
        if torch.jit.is_tracing():
            return self._forward_trace_friendly(x, return_reconstruction)
        
        batch_size = x.size(0)
        
        # Optimized input shape handling
        if len(x.shape) == 4:
            input_frames = x.shape[1]
            max_frames = self.input_shape[0]
            
            # Efficient frame selection without unnecessary operations
            if input_frames > max_frames:
                x = x[:, :max_frames, :, :].contiguous()
            
            F, H, W = x.shape[1:]
        else:
            raise ValueError(f"Expected 4D input tensor, got {x.shape}")
        
        # Use mixed precision if enabled and available
        if self.use_mixed_precision and self.training and torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                return self._forward_impl(x, batch_size, return_reconstruction)
        else:
            return self._forward_impl(x, batch_size, return_reconstruction)
    
    def _forward_trace_friendly(self, x, return_reconstruction=False):
        """Trace-friendly forward pass without dynamic control flow."""
        batch_size = x.size(0)
        
        # Assume input is already correct shape for tracing
        if len(x.shape) == 4:
            # Take only the expected number of frames using tensor slicing
            max_frames = self.input_shape[0]
            x = x[:, :max_frames, :, :].contiguous()
        else:
            raise ValueError(f"Expected 4D input tensor, got {x.shape}")
        
        return self._forward_impl(x, batch_size, return_reconstruction)
    
    def _forward_impl(self, x, batch_size, return_reconstruction):
        """Optimized implementation of forward pass."""
        # Convert to 3D convolution format with minimal memory overhead
        x_input = x.unsqueeze(1)  # B x 1 x F x H x W
        
        # Initial feature extraction without checkpointing
        latent_features = self.initial_conv(x_input)
        
        # Apply spatio-temporal attention
        attended_features = self.st_block(latent_features)
        
        # Memory-efficient pooling with single operation
        pooled_features = self.final_pool(
            self.adaptive_pool(attended_features)
        ).view(batch_size, self.embed_dim)
        
        # Final projection with normalization
        output_features = self.final_projection(pooled_features)
        
        # Avoid TracerWarning by using separate methods for different return types
        if return_reconstruction:
            return self._forward_with_reconstruction(attended_features, output_features)
        
        return output_features
    
    def _forward_with_reconstruction(self, attended_features, output_features):
        """Handle reconstruction separately to avoid tracing issues."""
        if self.use_reconstruction:
            reconstruction = self.reconstruction_head(attended_features)
            reconstruction = reconstruction.squeeze(1)
            return output_features, reconstruction
        return output_features

    def compute_auxiliary_loss(self, x, target=None, loss_weight=0.1):
        """Compute lightweight auxiliary loss for better representation learning.
        
        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target for reconstruction.
            loss_weight (float): Weight for auxiliary loss.
            
        Returns:
            torch.Tensor: Weighted auxiliary loss.
        """
        if not self.use_reconstruction:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
        output_features, reconstruction = self.forward(x, return_reconstruction=True)
        
        if target is None:
            target = x
        
        # Use lightweight loss computation
        loss = F.mse_loss(reconstruction, target, reduction='mean')
        return loss * loss_weight
    
    def enable_reconstruction(self, enable=True):
        """Enable or disable reconstruction head to save memory."""
        self.use_reconstruction = enable
        if not enable and hasattr(self, 'reconstruction_head'):
            # Delete reconstruction head to free memory
            delattr(self, 'reconstruction_head')
    
    def get_performance_info(self):
        """Get performance configuration and model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        total_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'memory_mb': total_memory_mb,
            'embed_dim': self.embed_dim,
            'use_reconstruction': self.use_reconstruction,
            'use_mixed_precision': self.use_mixed_precision,
            'attention_heads': self.st_block.spatial_mha.num_heads,
            'attention_dim': self.st_block.input_proj.out_features
        }


if __name__ == "__main__":
    # Example usage
    input_tensor = torch.randn(8, 5, 8, 8)  # Batch of 8, 5 frames, 8x8 resolution
    encoder = STImageEncoder(input_shape=(5, 8, 8), output_dim=128)

    features = encoder(input_tensor)
    print("Extracted features shape:", features.shape)

    # Check memory usage
    memory_info = encoder.get_memory_usage()
    print("Memory usage info:", memory_info)

    # Export model for visualization
    encoder.eval()  # Set to evaluation mode

    import onnx
    torch.onnx.export(
        encoder,
        input_tensor,
        "st_image_encoder.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={'input': {0: 'batch_size'}, 'features': {0: 'batch_size'}}
    )
    print("✓ ONNX model exported to: st_image_encoder.onnx")