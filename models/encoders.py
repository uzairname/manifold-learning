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
            nn.LeakyReLU(0.2),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.downsample2 = nn.Sequential(

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x += identity  # Add skip connection
        return x






class MLPEncoder(nn.Module):
  def __init__(self, latent_dim=2, img_size=128, **kwargs):
    super(MLPEncoder, self).__init__()
    
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(img_size**2, 512),
      nn.BatchNorm1d(512, momentum=0.5),
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
    sigmoid_scale=1
  ):
    super(ConvMLPEncoder, self).__init__()
    self.sigmoid_scale = sigmoid_scale
    
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
      nn.ReLU(),
      nn.Linear(fc_size, latent_dim),
    )
  
  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    x = self.sigmoid_scale*(F.sigmoid(x)-0.5)+0.5
    return x

  

  
class DeepMLPEncoder(nn.Module):
  def __init__(self, latent_dim=2, img_size=128, **kwargs):
    super(DeepMLPEncoder, self).__init__()
    
    self.conv = nn.Sequential(
      nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128
      nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
      nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, dilation=2),  # 64x64 -> 64x64
      nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
      nn.Flatten(),
    )
    
    dummy_input = torch.randn(1, 1, img_size, img_size)
    dummy_input = self.conv(dummy_input)
    self.dim_after_flatten = dummy_input.shape[-1]
    
    self.fc = nn.Sequential(
      nn.Linear(self.dim_after_flatten, 256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 256),
      nn.LayerNorm(256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 128),
      nn.LayerNorm(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 64),
      nn.LayerNorm(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(64, 32),
      nn.LayerNorm(32),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(32, 16),
      nn.LayerNorm(16),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(16, 8),
      nn.LayerNorm(8),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(8, 4),
      nn.LayerNorm(4),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(4, latent_dim),
      nn.Sigmoid(),
    )
    
  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x
  
  
  

  