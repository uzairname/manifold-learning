import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from models.decoders import ConvOnlyResidualDecoderBlock

class ConvResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()

        # Convolutional block
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # Skip connection
        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv(x)
        x += identity
        return x



class ConvResidualEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, **kwargs):
        super().__init__()

        # First upsampling block
        self.downsample1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3) if kwargs.get('dropout_position', False)==0 else nn.Identity(),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.downsample2 = nn.Sequential(

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32

            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32
        )

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x += identity  # Add skip connection
        return x





# Encoder: CNN extracts a latent representation from the image
class ConvEncoderBlock(nn.Module):
    def __init__(self, out_dim=2, img_size=128, **kwargs):
        super().__init__()       

        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128

            ConvResidualBlock(256, 128),
            nn.AvgPool2d(kernel_size=4, stride=4), # 128x128 -> 32x32

            ConvResidualBlock(128, 64, dilation_rate=2),
            nn.AvgPool2d(kernel_size=4, stride=4), # 32x32 -> 8x8

            ConvResidualBlock(64, 32, dilation_rate=4),
            nn.AvgPool2d(kernel_size=4, stride=4), # 8x8 -> 2x2

            nn.Flatten(),
        )

        dummy_input = torch.randn(1, 1, img_size, img_size)
        dummy_output = self.conv(dummy_input)
        dim_after_flatten = dummy_output.shape[-1]
        
        self.fc = nn.Sequential(
            nn.Linear(dim_after_flatten, img_size),
            nn.BatchNorm1d(img_size),
            nn.ReLU(),

            nn.Linear(img_size, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x



class DeepConvEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super().__init__()       

        self.conv = nn.Sequential(
            ConvResidualBlock(in_channels, out_channels, dilation_rate=dilation_rate),

            ConvResidualBlock(out_channels, out_channels, dilation_rate=dilation_rate),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
            
            ConvResidualBlock(out_channels, out_channels, dilation_rate=dilation_rate),

            ConvResidualBlock(out_channels, out_channels, dilation_rate=dilation_rate),
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
        )
        
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv(x)
        x += identity
        return x



class ConvResEncoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super().__init__()
        self.encoder = ConvEncoderBlock(out_dim=latent_dim, img_size=img_size)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        return x



class DeepConvResEncoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super().__init__()
       
        self.conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128

            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32

            DeepConvEncoderBlock(256, 64, dilation_rate=2), # 32x32 -> 8x8
            
            DeepConvEncoderBlock(64, 32, dilation_rate=4), # 8x8 -> 2x2
                                  
            nn.Flatten()
        )
        
        dummy_input = torch.randn(1, 1, img_size, img_size)
        dummy_input = self.conv(dummy_input)
        self.dim_after_flatten = dummy_input.shape[-1]
        
        self.fc = nn.Sequential(
            nn.Linear(self.dim_after_flatten, img_size),
            nn.BatchNorm1d(img_size),
            nn.ReLU(),

            nn.Linear(img_size, latent_dim),
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
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
  def __init__(self, latent_dim=2, img_size=128, **kwargs):
    super(ConvMLPEncoder, self).__init__()
    
    self.conv = nn.Sequential(
      ConvResidualEncoderBlock(1, 128, 2, dropout_position=kwargs.get('dropout_position', -1)),
      nn.Dropout(0.3) if kwargs.get('dropout_position', -1)==1 else nn.Identity(),
      ConvResidualEncoderBlock(128, 32, 2),
      nn.Dropout(0.3) if kwargs.get('dropout_position', -1)==2 else nn.Identity(),
      nn.Flatten(),
    )
    
    dummy_input = torch.randn(1, 1, img_size, img_size)
    dummy_input = self.conv(dummy_input)
    self.dim_after_flatten = dummy_input.shape[-1]
    
    self.fc = nn.Sequential(
      nn.Linear(self.dim_after_flatten, 128),
      nn.LayerNorm(128),
      nn.ReLU(),
      nn.Dropout(0.3) if kwargs.get('dropout_position', -1)==3 else nn.Identity(),
      nn.Linear(128, latent_dim),
      nn.Sigmoid(),
    )
    
  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
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
  
  
  

  
    

# class WideConvResEncoder(nn.Module):
#     def __init__(self, latent_dim=2, img_size=128, **kwargs):
#         super().__init__()
       
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 512, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128
            
#             nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64

#             ConvResidualBlock(512, 64),
#             nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64

#             ConvResidualBlock(64, 32, dilation_rate=2),
#             nn.AvgPool2d(kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32

#             nn.Flatten()
#         )
        
#         dummy_input = torch.randn(1, 1, img_size, img_size)
#         dummy_input = self.conv(dummy_input)
#         self.dim_after_flatten = dummy_input.shape[-1]
        
#         self.fc = nn.Sequential(
#             nn.Linear(self.dim_after_flatten, img_size*2),
#             nn.BatchNorm1d(img_size*2),
#             nn.ReLU(),

#             nn.Linear(img_size*2, latent_dim),
#         )
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(self.fc(x))
#         return x



class WideConvResEncoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(WideConvResEncoder, self).__init__()
       
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128
            
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 128x128 -> 32x32
            
            ConvResidualBlock(16, 256, dilation_rate=2),

            nn.Conv2d(256, 8, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            
            nn.AvgPool2d(kernel_size=4, stride=4, padding=1), # 32x32 -> 8x8
            
            nn.Flatten()
        )
        
        dummy_input = torch.randn(1, 1, img_size, img_size)
        dummy_input = self.conv(dummy_input)
        self.dim_after_flatten = dummy_input.shape[-1]
        
        self.fc = nn.Sequential(
            nn.Linear(self.dim_after_flatten, img_size),
            nn.BatchNorm1d(img_size),
            nn.ReLU(),

            nn.Linear(img_size, latent_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return x



class ConvSelfAttentionEncoder(nn.Module):
  def __init__(self, latent_dim=2, img_size=128, **kwargs):
    super(ConvSelfAttentionEncoder, self).__init__()
    
    self.conv = ConvEncoderBlock(out_dim=32, img_size=img_size)
        
    # self.attention = nn.MultiheadAttention(32, num_heads=8)
    
    self.fc = nn.Sequential(
      nn.Linear(32, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),
      nn.Linear(512, latent_dim),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    x = self.conv(x)
    # x, _ = self.attention(x, x, x)
    x = self.fc(x)
    return x
  


class TwoHeadConvEncoder(nn.Module):

    def __init__(self, out_dim=2, img_size=128, **kwargs):
        super(TwoHeadConvEncoder, self).__init__()

        # Feature extractors
        self.conv = ConvEncoderBlock(out_dim=1, img_size=img_size)
        self.head2 = ConvEncoderBlock(out_dim=1, img_size=img_size)
        
        self.fc = nn.Sequential(
          nn.Linear(2, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Linear(512, out_dim),
          nn.BatchNorm1d(out_dim),
          nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.head1(x)
        out2 = self.head2(x)
        out = torch.cat([out1, out2], dim=-1)
        out = self.fc(out)
        return out
      



