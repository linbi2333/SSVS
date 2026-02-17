#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networks.py — 3D U-Net and attention modules definitions for vertebra segmentation and classification.

Modules:
  - DoubleConv3D: Basic two-layer 3D convolutional block with InstanceNorm and LeakyReLU.
  - ChannelAttention3D: Channel-wise attention mechanism for 3D feature maps.
  - SpatialAttention3D: Spatial attention mechanism for 3D feature maps.
  - CBAM3D: Convolutional Block Attention Module combining channel and spatial attention.
  - UNet3D: Simplified 3D U-Net with separate decoder branches for segmentation and vertebra classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet3D(nn.Module):
    """
    Simplified 3D U-Net with dual decoder heads for segmentation and classification.

    Encoder extracts multi-scale features; two decoders upsample to different resolutions.
    """
    def __init__(
        self,
        n_channels: int = 1,
        n_seg_classes: int = 2,
        n_vert_classes: int = 26,
        base_filters: int = 32
    ):
        """
        Initialize UNet3D architecture.

        Parameters
        ----------
        n_channels : int
            Number of input channels (e.g., CT has 1 channel).
        n_seg_classes : int
            Number of output segmentation classes.
        n_vert_classes : int
            Number of vertebra classification classes.
        base_filters : int
            Number of filters for first conv layer; doubles at each downsample.
        """
        super().__init__()
        f = base_filters
        self.num_classes = n_vert_classes

        # Encoder path
        self.inc = DoubleConv3D(n_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(f, f*2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(f*2, f*4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(f*4, f*8))

        # Classification decoder (1/8 → 1/4 resolution)
        self.cls_up = nn.Sequential(
            nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2),
            nn.Conv3d(f*4, f*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(f*4, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            CBAM3D(channels=f*4)
        )
        self.cls_conv = nn.Sequential(
            DoubleConv3D(f*8, f*4),
            DoubleConv3D(f*4, f*2),
            DoubleConv3D(f*2, f)
        )
        self.cls_cbam = CBAM3D(channels=f)
        self.cls_out = nn.Conv3d(f, n_vert_classes, kernel_size=1)

        # Segmentation decoder (1/8 → full resolution)
        self.seg_up1 = nn.Sequential(
            nn.ConvTranspose3d(f*8, f*4, kernel_size=2, stride=2),
            nn.Conv3d(f*4, f*4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(f*4, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.seg_conv1 = DoubleConv3D(f*8, f*4)
        self.seg_ds4_out = nn.Conv3d(f*4, n_seg_classes, kernel_size=1)

        self.seg_up2 = nn.Sequential(
            nn.ConvTranspose3d(f*4, f*2, kernel_size=2, stride=2),
            nn.Conv3d(f*2, f*2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(f*2, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.seg_conv2 = DoubleConv3D(f*4, f*2)
        self.seg_ds2_out = nn.Conv3d(f*2, n_seg_classes, kernel_size=1)

        self.seg_up3 = nn.Sequential(
            nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2),
            nn.Conv3d(f, f, kernel_size=3, padding=1),
            nn.InstanceNorm3d(f, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.seg_conv3 = DoubleConv3D(f*2, f)
        self.seg_out = nn.Conv3d(f, n_seg_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Forward propagation through encoder and decoders.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, n_channels, H, W, D].

        Returns
        -------
        cls_logits : torch.Tensor
            Classification logits at 1/4 resolution [B, n_vert_classes, H/4, W/4, D/4].
        seg_logits : torch.Tensor
            Segmentation logits at full resolution [B, n_seg_classes, H, W, D].
        seg_ds2 : torch.Tensor
            Deep supervision output at 1/2 resolution.
        seg_ds4 : torch.Tensor
            Deep supervision output at 1/4 resolution.
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Classification branch
        uc = self.cls_up(x4)
        uc = torch.cat([uc, x3], dim=1)
        uc = self.cls_conv(uc)
        uc = self.cls_cbam(uc)
        cls_logits = self.cls_out(uc)

        # Segmentation branch
        us = self.seg_up1(x4)
        us = torch.cat([us, x3], dim=1)
        us = self.seg_conv1(us)
        seg_ds4 = self.seg_ds4_out(us)

        us = self.seg_up2(us)
        us = torch.cat([us, x2], dim=1)
        us = self.seg_conv2(us)
        seg_ds2 = self.seg_ds2_out(us)

        us = self.seg_up3(us)
        us = torch.cat([us, x1], dim=1)
        us = self.seg_conv3(us)
        seg_logits = self.seg_out(us)

        return cls_logits, seg_logits, seg_ds2, seg_ds4


class DoubleConv3D(nn.Module):
    """
    Two consecutive 3D convolution blocks: Conv3d → InstanceNorm3d → LeakyReLU.

    This block preserves spatial dimensions via padding=1 and applies two such convolutions.
    """
    def __init__(self, in_ch: int, out_ch: int):
        """
        Initialize DoubleConv3D module.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        """
        super().__init__()
        # First convolutional layer
        self.block1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        # Second convolutional layer
        self.block2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through two convolutional blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, in_ch, H, W, D].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [B, out_ch, H, W, D].
        """
        x = self.block1(x)
        x = self.block2(x)
        return x


class ChannelAttention3D(nn.Module):
    """
    Channel-wise attention for 3D feature maps.

    Uses both global average and max pooling followed by a shared MLP.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initialize ChannelAttention3D.

        Parameters
        ----------
        in_channels : int
            Number of input feature channels.
        reduction : int, optional
            Reduction ratio for hidden MLP layer (default=16).
        """
        super().__init__()
        hidden = in_channels // reduction
        # Shared MLP implemented via 1x1x1 convolutions
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Output shape [B, C, 1, 1, 1]
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_channels, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute channel attention and scale input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W, D].

        Returns
        -------
        torch.Tensor
            Channel-refined feature map.
        """
        # Average pooling branch
        avg_pool = self.mlp(x)
        # Max pooling branch (reuse same MLP)
        max_pool = self.mlp(F.adaptive_max_pool3d(x, output_size=1))
        # Combine and apply sigmoid activation
        attn = self.sigmoid(avg_pool + max_pool)
        # Scale input features by attention weights
        return x * attn


class SpatialAttention3D(nn.Module):
    """
    Spatial attention for 3D feature maps.

    Learns spatial mask via convolution on concatenated channel-wise average and max projections.
    """
    def __init__(self, kernel_size: int = 7):
        """
        Initialize SpatialAttention3D.

        Parameters
        ----------
        kernel_size : int, optional
            Convolution kernel size for spatial attention (default=7).
        """
        super().__init__()
        # Ensure padding maintains shape
        padding = (kernel_size - 1) // 2
        # Convolution expects 2-channel input (avg + max)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention and scale input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W, D].

        Returns
        -------
        torch.Tensor
            Spatially refined feature map.
        """
        # Channel-wise average pooling: shape [B,1,H,W,D]
        avg = torch.mean(x, dim=1, keepdim=True)
        # Channel-wise max pooling: shape [B,1,H,W,D]
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel axis: shape [B,2,H,W,D]
        cat = torch.cat([avg, max_val], dim=1)
        # Generate spatial attention map
        attn = self.sigmoid(self.conv(cat))
        # Scale input features by spatial attention
        return x * attn


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D inputs.

    Sequentially applies channel and spatial attention.
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        """
        Initialize CBAM3D.

        Parameters
        ----------
        channels : int
            Number of feature channels.
        reduction : int, optional
            Reduction ratio for channel attention (default=16).
        spatial_kernel : int, optional
            Kernel size for spatial attention (default=7).
        """
        super().__init__()
        self.channel_att = ChannelAttention3D(channels, reduction)
        self.spatial_att = SpatialAttention3D(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM: channel attention followed by spatial attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W, D].

        Returns
        -------
        torch.Tensor
            Attention-refined feature map.
        """
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x



