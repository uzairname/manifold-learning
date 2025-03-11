import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class ConvResidualEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, **kwargs):
        super().__init__()

        # First upsampling block
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

        # Second upsampling block
        self.downsample2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x += identity  # Add skip connection
        return x




class MLPEncoder(nn.Module):
  def __init__(self, latent_dim=2, img_size=128):
    super(MLPEncoder, self).__init__()
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(img_size**2, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, latent_dim),
      nn.Sigmoid(),
    )
    
  def forward(self, x):
    x = self.fc(x)
    return x

  
  


class ConvMLPEncoder(nn.Module):
  def __init__(
    self, 
    latent_dim=2, 
    img_size=128, 
    fc_dims=[128],
    sigmoid=False,
    n_conv_blocks=3,
    channels=[1, 16, 32, 64]
  ):
    super(ConvMLPEncoder, self).__init__()
    
    conv_blocks = []
    
    for i in range(n_conv_blocks):
      conv_blocks.append(
        ConvResidualEncoderBlock(channels[i], channels[i+1], 2)
      )
      
    self.conv = nn.Sequential(*conv_blocks, nn.Flatten())
    
    dummy_input = torch.randn(1, 1, img_size, img_size)
    dummy_input = self.conv(dummy_input)
    self.dim_after_flatten = dummy_input.shape[-1]
    
    fc_layers = []
    
    for i in range(len(fc_dims)):
      in_dim = self.dim_after_flatten if i == 0 else fc_dims[i-1]
      fc_layers.extend([
        nn.Linear(in_dim, fc_dims[i]),
        nn.BatchNorm1d(fc_dims[i]),
        nn.Tanh(),
      ])
      
    self.fc = nn.Sequential(
      *fc_layers,
      nn.Linear(fc_dims[-1], latent_dim),
      nn.BatchNorm1d(latent_dim),
      nn.Sigmoid() if sigmoid else nn.Tanh(),
    )
  
  
  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x



  