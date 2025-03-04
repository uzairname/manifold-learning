import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch


class ConvResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First upsampling block
        self.upsample1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.upsample2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x += identity  # Add skip connection
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128):
        super().__init__()

        self.dim_after_conv = (2 * img_size) // 128
        resnet_start_channels = 256

        # Decoder (Mirroring the Encoder)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, img_size),
            nn.BatchNorm1d(img_size),
            nn.ReLU(),

            nn.Linear(img_size, resnet_start_channels * self.dim_after_conv * self.dim_after_conv),
            nn.ReLU(),
            nn.Unflatten(1, (resnet_start_channels, self.dim_after_conv, self.dim_after_conv)), # 2x2
        )

        self.decoder_conv = nn.Sequential(
            # Upsampling steps (each block doubles the resolution)
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels),  # 2x2 -> 8x8
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels // 2),  # 8x8 -> 32x32
            ConvResidualDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 4),  # 32x32 -> 128x128
            nn.Conv2d(resnet_start_channels // 4, 1, kernel_size=3, stride=1, padding=1), # 128x128 -> 128x128
            nn.Sigmoid()  # Output in range [0,1]
        )


    def forward(self, x):
        x = self.decoder_fc(x)
        x = self.decoder_conv(x)
        return x


class ClockDecoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(ClockDecoder, self).__init__()

        self.decoder = ResNetDecoder(latent_dim=latent_dim, img_size=img_size)

    def forward(self, x):
        reconstructed = self.decoder(x)
        return reconstructed


class FiLM(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.gamma = nn.Linear(m, n)
        self.beta = nn.Linear(m, n)

    def forward(self, x, latent):
        gamma = self.gamma(latent).unsqueeze(1)  # (B, 1, n)
        beta = self.beta(latent).unsqueeze(1)    # (B, 1, n)
        out = gamma * x + beta # B, num_pixels, 1
        return out #



class INRDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim=128, out_channels=1, hidden_dim=256, num_layers=4, fourier_features=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fourier_features = fourier_features

        # Learnable frequencies for Fourier features
        self.freqs = nn.Parameter(torch.randn(fourier_features, 2) * 2 * np.pi)

        self.fc_in = nn.Linear(latent_dim + 2 * fourier_features, hidden_dim)
        self.hidden_layers = nn.Sequential(
            *[(nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords, z):
        """
        coords: (B, N, 2) - Spatial coordinates
        z: (B, latent_dim) - Latent representation
        """
        B, N, _ = coords.shape

        encoded_coords = torch.cat([
            torch.sin(coords @ self.freqs.T),
            torch.cos(coords @ self.freqs.T)
        ], dim=-1)
        torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1) if self.fourier_features > 0 else coords

        # Expand latent vector to match coordinate dimensions
        z = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        inputs = torch.cat([encoded_coords, z], dim=-1)  # Concatenate (B, N, latent_dim + 2 * fourier_features)

        x = self.fc_in(inputs) # (B, H*W, hidden_dim)
        x = self.hidden_layers(x) # (B, H*W, hidden_dim)
        x = self.fc_out(x) # (B, H*W, output_dim)

        # Reshape to (B, C, H, W)
        C = x.shape[-1]  # Number of output channels
        x = x.view(B, self.output_dim, self.output_dim, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x