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
    fc_size=128,
    sigmoid=False,
  ):
    super(ConvMLPEncoder, self).__init__()
    
    self.conv = nn.Sequential(
      ConvResidualEncoderBlock(1, 16, 2), # 128x128 -> 32x32
      ConvResidualEncoderBlock(16, 32, 2), # 32x32 -> 8x8
      ConvResidualEncoderBlock(32, 64, 2), # 8x8 -> 2x2
      nn.Flatten(),
    )
    
    dummy_input = torch.randn(1, 1, img_size, img_size)
    dummy_input = self.conv(dummy_input)
    self.dim_after_flatten = dummy_input.shape[-1]
    
    self.fc = nn.Sequential(
      nn.Linear(self.dim_after_flatten, fc_size),
      nn.BatchNorm1d(fc_size),
      nn.Tanh(),
      nn.Linear(fc_size, latent_dim + 1),
      nn.BatchNorm1d(latent_dim + 1),
      nn.Tanh(),
      nn.Linear(latent_dim + 1, latent_dim),
      nn.Sigmoid() if sigmoid else nn.Tanh(),
    )
  
  def forward(self, x):
    x = self.conv(x)
    if torch.isnan(x).any(): print("NaN in encoder conv output")
    x = self.fc(x)
    if torch.isnan(x).any(): print("NaN in encoder fc output")
    return x



  