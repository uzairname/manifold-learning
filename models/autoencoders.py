import torch
import torch.nn as nn
from models.decoders import ResNetDecoder, INRDecoder
from models.encoders import ConvEncoderBlock, ConvMLPEncoder, MLPEncoder
import torch.nn.functional as F


class ResnetAutoencoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(ResnetAutoencoder, self).__init__()

        self.encoder = ConvEncoderBlock(out_dim=latent_dim, img_size=img_size)
        self.decoder = ResNetDecoder(latent_dim=latent_dim, img_size=img_size)

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed


class ConvInrAutoencoder(nn.Module):
    def __init__(self, img_size=128, latent_dim=16, device='cuda'):
        super().__init__()
        self.input_dim = img_size
        self.encoder = ConvEncoderBlock(latent_dim, img_size=img_size)
        self.decoder = INRDecoder(latent_dim, output_dim=img_size, out_channels=1, fourier_features=8)

        x = torch.linspace(-1, 1, img_size).to(device)
        y = torch.linspace(-1, 1, img_size).to(device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)  # (H*W, 2)

    def forward(self, x):
        """
        x: (B, 1, H, W) - Input grayscale image
        """
        B = x.size(0)
        latent = self.encoder(x)  # (B, latent_dim)

        # Expand coordinates to match batch size
        coords = self.coords.unsqueeze(0).expand(B, -1, -1) # (B, H*W, 2)

        # Decode intensities per pixel (B, H*W, C)
        reconstructed = self.decoder(coords, latent) # (B, H*W, 1)

        return reconstructed



class MLPResnetAutoencoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(MLPResnetAutoencoder, self).__init__()

        self.encoder = ConvMLPEncoder(latent_dim=latent_dim, img_size=img_size)
        self.decoder = ResNetDecoder(latent_dim=latent_dim, img_size=img_size)

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed

