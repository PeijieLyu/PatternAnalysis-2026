"""Building blocks for a 2D U-Net segmentation model."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Apply two 3x3 convolutions with ReLU, preserving spatial dimensions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample with 2x2 max pooling, then apply DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Resize to skip dimensions, concatenate channels, then apply DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Four-level baseline U-Net returning full-resolution class logits."""

    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        if min(in_channels, num_classes, base_channels) < 1:
            raise ValueError("Channel counts and num_classes must be positive")

        self.in_channels = in_channels
        self.num_classes = num_classes
        b = base_channels

        self.inc = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)

        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.out = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input shape [B, C, H, W]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        if min(x.shape[-2:]) < 16:
            raise ValueError("Input height and width must each be at least 16")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
