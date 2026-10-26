"""Building blocks for a 2D U-Net segmentation model."""

from torch import nn


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
