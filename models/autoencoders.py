import torch
import torch.nn as nn
from models.decoders import ActivationType, ResNetDecoder3, ImplicitNeuralDecoder
from models.encoders import ConvMLPEncoder, MLPEncoder
from models.other import MLP


class MLPResnetAutoencoder(nn.Module):

    def __init__(self, latent_dim=2, img_size=128, encoder_args:dict=None, decoder_args: dict=None):
        super(MLPResnetAutoencoder, self).__init__()

        self.encoder = ConvMLPEncoder(latent_dim=latent_dim, img_size=img_size, **(encoder_args or {}))
        self.decoder = ResNetDecoder3(latent_dim=latent_dim, img_size=img_size, **(decoder_args or {}))

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed


class ConvINRAutoencoder(nn.Module):
  
    def __init__(self, latent_dim=2, img_size=128, encoder_args:dict=None, decoder_args: dict=None):
        super(ConvINRAutoencoder, self).__init__()

        self.encoder = ConvMLPEncoder(latent_dim=latent_dim, img_size=img_size, **(encoder_args or {}))
        self.decoder = ImplicitNeuralDecoder(latent_dim=latent_dim, img_size=img_size, **(decoder_args or {}))

    def forward(self, latent):
        latent = self.encoder(latent)
        reconstructed = self.decoder(latent)
        return reconstructed


class MLPAutoencoder(nn.Module):
  
  def __init__(self, encoder_dims=[1, 1], decoder_dims=[1, 1], decoder_activation: ActivationType='sigmoid'):
    super(MLPAutoencoder, self).__init__()

    self.encoder = MLP(dims=encoder_dims)
    self.decoder = MLP(dims=decoder_dims, activation=decoder_activation)
    
  def forward(self, x):
    latent = self.encoder(x)
    reconstructed = self.decoder(latent)
    return reconstructed
  