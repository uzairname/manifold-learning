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




class ConvResidualEncoderBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()

        # First downsampling block using strided convolution
        self.downsample1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

        # Second downsampling block using strided convolution
        self.downsample2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

        # Skip connection using 1x1 conv and strided conv to match spatial dimensions
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=4, padding=1)  # 128x128 -> 32x32
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection
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
    supervised=False,
    channels=[16, 32, 64]
  ):
    super(ConvMLPEncoder, self).__init__()
    
    self.supervised = supervised
      
    self.conv = nn.Sequential()
    for i in range(len(channels)):
      in_channels = 1 if i == 0 else channels[i-1]
      self.conv.append(
        ConvResidualEncoderBlock2(in_channels, channels[i], dilation_rate=i+1)
      )
    self.conv.append(nn.Flatten())
    
    dim_after_flatten = self.conv(torch.randn(1, 1, img_size, img_size)).shape[-1]
    
    self.fc = nn.Sequential()
    for i in range(len(fc_dims)):
      in_dim = dim_after_flatten if i == 0 else fc_dims[i-1]
      self.fc.extend([
        nn.Linear(in_dim, fc_dims[i]),
        nn.BatchNorm1d(fc_dims[i]),
        nn.Tanh(),
      ])
      
    self.last_layer = nn.Sequential(
      nn.Linear(fc_dims[-1], latent_dim),
      nn.Sigmoid() if supervised else nn.Tanh(),
    )


  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    x = self.last_layer(x)
    if self.supervised:
      # expand range to include 0 and 1 better
      x = x * 1.1 - 0.05
    return x


