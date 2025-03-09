import torch
import torch.nn as nn
from models.decoders import ResNetDecoder
from models.encoders import ConvMLPEncoder



class MLPResnetAutoencoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(MLPResnetAutoencoder, self).__init__()

        self.encoder = ConvMLPEncoder(latent_dim=latent_dim, img_size=img_size)
        self.decoder = ResNetDecoder(latent_dim=latent_dim, img_size=img_size)

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed

