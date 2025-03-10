import torch
import torch.nn as nn
from models.decoders import ResNetDecoder, ResNetDecoder2, ResNetDecoder3
from models.encoders import ConvMLPEncoder



class MLPResnetAutoencoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, encoder_args:dict=None, decoder_args: dict=None):
        super(MLPResnetAutoencoder, self).__init__()

        self.encoder = ConvMLPEncoder(latent_dim=latent_dim, img_size=img_size, **(encoder_args or {}))
        self.decoder = ResNetDecoder3(latent_dim=latent_dim, img_size=img_size, **(decoder_args or {}))

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed

