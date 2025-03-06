import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()

        # First downsampling block
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # Skip connection (downsampling by 4 using a single convolution and pooling)
        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (downsampling)
        x = self.conv(x)
        x += identity  # Add skip connection
        return x


# Encoder: CNN extracts a latent representation from the image
class ConvResEncoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128):
        super().__init__()

        self.dim_after_conv = (2 * img_size) // 128  # After 5 Conv Layers

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128

            ConvResidualBlock(256, 128),
            nn.AvgPool2d(kernel_size=4, stride=4), # 128x128 -> 32x32

            ConvResidualBlock(128, 64, dilation_rate=2),
            nn.AvgPool2d(kernel_size=4, stride=4), # 32x32 -> 8x8

            ConvResidualBlock(64, 32, dilation_rate=4),
            nn.AvgPool2d(kernel_size=4, stride=4), # 8x8 -> 2x2

            nn.Flatten(),

            nn.Linear(32 * self.dim_after_conv**2, img_size),
            nn.BatchNorm1d(img_size),
            nn.ReLU(),

            nn.Linear(img_size, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)



class ConvHead(nn.Module):

    def __init__(self, input_dim=64):
        super(ConvHead, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After Conv Layers

        self.conv_head = nn.Sequential(
            ConvResidualBlock(32),

            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            ConvResidualBlock(32),

            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8 -> 4x4

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * self.dim_after_conv * self.dim_after_conv, 64),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv_head(x)
        return out



class MultiHeadRegressor(nn.Module):

    def __init__(self, out_dim=2, input_dim=64, **kwargs):
        super(MultiHeadRegressor, self).__init__()

        self.input_dim = input_dim

        # Feature extractors
        self.head1 = ConvResEncoder(latent_dim=out_dim//2, img_size=input_dim)
        self.head2 = ConvResEncoder(latent_dim=out_dim//2, img_size=input_dim)

    def forward(self, x):
        out1 = self.head1(x)
        out2 = self.head2(x)

        out = torch.cat([out1, out2], dim=-1)
        return out
      
      
      