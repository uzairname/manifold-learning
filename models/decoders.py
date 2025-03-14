import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
from functools import partial
from enum import Enum


class ConvResidualDecoderBlock(nn.Module):
    def __init__(
      self, 
      in_channels, 
      out_channels, 
      convt_strides=[2,2], 
      dilation=1
    ):
        super().__init__()
                
        # First upsampling block
        self.upsample1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convt_strides[0]+2, stride=convt_strides[0], padding=1)), # x2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.upsample2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=convt_strides[1]+2, stride=convt_strides[1], padding=1)), # x2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=np.prod(convt_strides), mode="bilinear", align_corners=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = x + identity  # Add skip connection
        return x




class ConvOnlyResidualDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()

        # First upsampling block
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
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
      


# class ConvTransposeResidualDecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, convt_strides=[2,2], dilation=1):
#         super().__init__()
        
#         # First upsampling block
#         self.upsample1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convt_strides[0]+2, stride=convt_strides[0], padding=1), # x2
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2)
#         )

#         # Second upsampling block
#         self.upsample2 = nn.Sequential(
#             nn.ConvTranspose2d(out_channels, out_channels, kernel_size=convt_strides[1]+2, stride=convt_strides[1], padding=1), # x2
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2),

#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(0.2)
#         )

#         # Skip connection (upsampling by 4 using interpolation)
#         self.skip = nn.Sequential(
#             nn.Upsample(scale_factor=np.prod(convt_strides), mode="bilinear", align_corners=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_channels),
#         )
        
#     def forward(self, x):
#         identity = self.skip(x)  # Skip connection (upsampling)
#         x = self.upsample1(x)
#         x = self.upsample2(x)
#         x += identity  # Add skip connection
#         return x





class ConvResidualDecoderBlock3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, dilation=1):
        super(ConvResidualDecoderBlock3d, self).__init__()
        
        self.upsample1 = nn.Sequential(
          nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
          nn.BatchNorm3d(out_planes),
          nn.LeakyReLU(),
          nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=padding, dilation=dilation),
          nn.BatchNorm3d(out_planes),
          nn.LeakyReLU()
        )
        
        self.upsample2 = nn.Sequential(
          nn.ConvTranspose3d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
          nn.BatchNorm3d(out_planes),
          nn.LeakyReLU(),
          nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=padding, dilation=dilation),
          nn.BatchNorm3d(out_planes),
          nn.LeakyReLU()
        )

        self.skip = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=stride**2, mode='nearest'),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.upsample1(x)
        out = self.upsample2(out)
        out += identity
        return out



class ActivationType(str, Enum):
    leakyrelu = 'leakyrelu'
    relu = 'relu'
    sigmoid = 'sigmoid'
    sine = 'sine'
    tanh = 'tanh'


def str_to_activation(activation: str):
    if activation == ActivationType.leakyrelu:
        return partial(nn.LeakyReLU, negative_slope=0.2)
    elif activation == ActivationType.sigmoid:
        return partial(nn.Sigmoid)
    elif activation == ActivationType.relu:
        return partial(nn.ReLU)
    elif activation == ActivationType.sine:
        return partial(nn.SiLU)
    elif activation == ActivationType.tanh:
        return partial(nn.Tanh)
    else:
        raise ValueError(f"Unknown activation type: {activation}")



class ResNetDecoder3(nn.Module):
    def __init__(
      self, 
      latent_dim=2, 
      img_size=128, 
      fc_size=128,
      resnet_start_channels=256,
    ):
        super().__init__()

        self.dim_before_conv = (4 * img_size) // 128
        
        fc_size = resnet_start_channels*self.dim_before_conv**2 // 2

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, resnet_start_channels*self.dim_before_conv**2),
            nn.ReLU(),
            nn.Unflatten(1, (resnet_start_channels, self.dim_before_conv, self.dim_before_conv)), # 4x4
        )
        
        self.decoder_conv = nn.Sequential(
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels // 2, convt_strides=[1,2]), # -> 8x8
            ConvResidualDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 4, convt_strides=[2,2], dilation=2),  # -> 32x32
            ConvResidualDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 4, convt_strides=[2,2], dilation=2),  # -> 128x128
            nn.Conv2d(resnet_start_channels // 4, 1, kernel_size=3, stride=1, padding=1), # -> 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        if torch.isnan(x).any(): print("NaN in decoder fc output")
        x = self.decoder_conv(x)
        if torch.isnan(x).any(): print("NaN in decoder conv output")
        return x



class ResNetDecoder4(nn.Module):
    def __init__(
      self, 
      latent_dim=2, 
      img_size=128, 
      resnet_start_channels=128,
      fc_size=128,
    ):
      super().__init__()

      self.dim_before_conv = (2 * img_size) // 128
      
      fc_size = resnet_start_channels*self.dim_before_conv**2 // 2

      self.fc = nn.Sequential(
          nn.Linear(latent_dim, fc_size),
          nn.BatchNorm1d(fc_size),
          nn.ReLU(),
          nn.Linear(fc_size, resnet_start_channels*self.dim_before_conv**2),
          nn.BatchNorm1d(resnet_start_channels*self.dim_before_conv**2),
          nn.ReLU(),
          nn.Unflatten(1, (resnet_start_channels, self.dim_before_conv, self.dim_before_conv)), # 2
      )
      
      self.decoder_conv = nn.Sequential(
        ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels, convt_strides=[2,2]), # -> 4
        ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels // 8, convt_strides=[2,2], dilation=2),  # -> 16
        ConvResidualDecoderBlock(resnet_start_channels // 8, resnet_start_channels // 32, convt_strides=[1,2], dilation=2),  # -> 32
        ConvResidualDecoderBlock(resnet_start_channels // 32, resnet_start_channels // 64, convt_strides=[1,2], dilation=2),  # -> 64
        nn.Conv2d(resnet_start_channels // 64, 1, kernel_size=3, stride=1, padding=1), # -> 128
        nn.Sigmoid()
      )

    def forward(self, x):
      x = self.fc(x)
      if torch.isnan(x).any(): print("NaN in decoder fc output")
      x = self.decoder_conv(x)
      if torch.isnan(x).any(): print("NaN in decoder conv output")
      return x




class ImplicitNeuralDecoder(nn.Module):
  def __init__(
    self, 
    img_size:int=128, 
    latent_dim=2, 
    fc_dims=[128], 
    pe_dim=10, 
    activation=ActivationType.sine,
    use_film=True,
):
    super().__init__()
    self.img_size = img_size
    self.pe_dim = pe_dim
    self.use_film = use_film

    # (x, y) coordinates
    input_dim = 2  

    # Positional encoding expands input dimensions. *4 for sin/cos for x/y
    input_dim += 4 * pe_dim  
    
    # Define FiLM conditioning network
    
    if use_film:
      # If FiLM is used, no need to use use context in the input
      self.input_dim = input_dim
      self.film_generator = nn.Sequential(
        nn.Linear(latent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2 * sum(fc_dims))  # Generates (γ, β) for each layer
      )
    else:
      self.input_dim = input_dim + latent_dim


    self.fc = nn.ModuleList()
    for i in range(len(fc_dims)):
      self.fc.append(nn.Linear(self.input_dim if i == 0 else fc_dims[i-1], fc_dims[i]))
      self.fc.append(nn.BatchNorm1d(fc_dims[i]))
      self.fc.append(str_to_activation(activation)())

    self.output_fc = nn.Sequential(
      nn.Linear(fc_dims[-1], 1),  # Output layer
      nn.Sigmoid() # Sigmoid activation for pixel values
    )

    # Precompute coordinates and their positional encodings
    self.register_buffer("coords", self._generate_coords())
    self.register_buffer("coords_pe", self._compute_positional_encoding(self.coords))


  def _generate_coords(self):
    x = torch.linspace(0, 1, self.img_size)
    y = torch.linspace(0, 1, self.img_size)
    x, y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([x.flatten(), y.flatten()], dim=-1)  # (img_size^2, 2)
    return coords

  def _compute_positional_encoding(self, coords):
    pe = []
    if self.pe_dim == 0:
      return torch.zeros(coords.shape[0], 0, device=coords.device)
    for i in range(self.pe_dim):
      pe.append(torch.sin((2.0 ** i) * np.pi * coords))
      pe.append(torch.cos((2.0 ** i) * np.pi * coords))
    return torch.cat(pe, dim=-1)  # (n*n, 2*pe_dim)

  def forward(self, context):
    batch_size = context.shape[0] if context.dim() > 1 else 1
    coords = self.coords  # (img_size^2, 2)

    # Expand context for each coordinate
    context = context.unsqueeze(1).expand(-1, coords.shape[0], -1)  # (batch, img_size^2, context_dim)
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, img_size^2, 2)
    coords = torch.cat([coords, self.coords_pe.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)

    # Flatten for input to MLP
    coords = coords.reshape(-1, coords.shape[-1])
    context = context.reshape(-1, context.shape[-1])
    
    if self.use_film:
      # if FiLM is used, no need to use context in the input
      x = coords
    else:
      # Concatenate coordinates and context
      x = torch.cat([coords, context], dim=-1)
    
    if self.use_film:
      film_params = self.film_generator(context[:, :context.shape[-1]])  # (batch*n*n, 2 * sum(fc_dims))
      gamma, beta = torch.split(film_params, film_params.shape[-1] // 2, dim=-1)

    split_idx = 0
    for layer in self.fc:
      x = layer(x)
      
      # Apply FiLM parameters
      if isinstance(layer, nn.Linear) and self.use_film:
        gamma_layer = gamma[:, split_idx:split_idx + layer.out_features]
        beta_layer = beta[:, split_idx:split_idx + layer.out_features]
        x = gamma_layer * x + beta_layer
        split_idx += layer.out_features
          
    output = self.output_fc(x)  # (batch * n*n, out_channels)

    return output.view(batch_size, 1, self.img_size, self.img_size)  # (batch, 1, img_size, img_size)

