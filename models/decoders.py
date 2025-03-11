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
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convt_strides[0]+2, stride=convt_strides[0], padding=1)), # x2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Second upsampling block
        self.upsample2 = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=convt_strides[1]+2, stride=convt_strides[1], padding=1)), # x2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        # Skip connection (upsampling by 4 using interpolation)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=np.prod(convt_strides), mode="bilinear", align_corners=True),
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = x + identity  # Add skip connection
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



class ResNetDecoder(nn.Module):
    def __init__(self, latent_dim=2, img_size=128):
        super().__init__()
                
        self.dim_before_conv = (2 * img_size) // 128
        resnet_start_channels = 256
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, resnet_start_channels * self.dim_before_conv**2),
            nn.ReLU(),
            nn.Unflatten(1, (resnet_start_channels, self.dim_before_conv, self.dim_before_conv)), # 2x2
        )

        self.decoder_conv = nn.Sequential(
            # Upsampling steps (each block doubles the resolution)
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels),  # 2x2 -> 8x8
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels // 2),  # 8x8 -> 32x32
            ConvResidualDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 4),  # 32x32 -> 128x128
            nn.Conv2d(resnet_start_channels // 4, 1, kernel_size=3, stride=1, padding=1), # 128x128 -> 128x128
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder_conv(x)
        return x




class ResNetDecoder2(nn.Module):
    def __init__(self, latent_dim=2, img_size=128, resnet_start_channels=256, dropout_rate=0.2):
        super().__init__()

        self.dim_before_conv = (8 * img_size) // 128
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, resnet_start_channels * self.dim_before_conv**2),
            nn.ReLU(),
            nn.Unflatten(1, (resnet_start_channels, self.dim_before_conv, self.dim_before_conv)), # 2x2
        )

        self.decoder_conv = nn.Sequential(
            ConvResidualDecoderBlock(resnet_start_channels, resnet_start_channels // 2, convt_strides=[2,2]), #
            nn.Dropout(0.2),
            ConvResidualDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 4, convt_strides=[1,1], dilation=1),  #
            nn.Dropout(0.1),
            ConvResidualDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 4, convt_strides=[1,1], dilation=1),  # 
            nn.Dropout(0.04),
            ConvResidualDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 4, convt_strides=[1,1], dilation=1),  # 
            nn.Dropout(0.02),
            ConvResidualDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 4, convt_strides=[1,1], dilation=1),  # 
            nn.Dropout(0.01),
            ConvResidualDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 16, convt_strides=[1,2], dilation=1),  # 

            ConvResidualDecoderBlock(resnet_start_channels // 16, resnet_start_channels // 16, convt_strides=[1,2], dilation=1),  # 128x128 -> 128x128
            nn.Conv2d(resnet_start_channels // 16, 1, kernel_size=3, stride=1, padding=1), # 128x128 -> 128x128
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder_conv(x)
        return x




class ActivationType(str, Enum):
    leakyrelu = 'leakyrelu'
    relu = 'relu'
    sigmoid = 'sigmoid'



def str_to_activation(activation: str):
    if activation == ActivationType.leakyrelu:
        return partial(nn.LeakyReLU, negative_slope=0.2)
    elif activation == ActivationType.sigmoid:
        return partial(nn.Sigmoid)
    elif activation == ActivationType.relu:
        return partial(nn.ReLU)
    else:
        raise ValueError(f"Unknown activation type: {activation}")


class ResNetDecoder3(nn.Module):
    def __init__(
      self, 
      latent_dim=2, 
      img_size=128, 
      resnet_start_channels=256,
      fc_size=128,
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
            nn.Dropout(0.2),
            ConvResidualDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 8, convt_strides=[2,2], dilation=2),  # -> 32x32
            nn.Dropout(0.04),
            ConvResidualDecoderBlock(resnet_start_channels // 8, resnet_start_channels // 32, convt_strides=[2,2], dilation=2),  # -> 128x128
            nn.Conv2d(resnet_start_channels // 32, 1, kernel_size=3, stride=1, padding=1), # -> 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        if torch.isnan(x).any(): print("NaN in decoder fc output")
        x = self.decoder_conv(x)
        if torch.isnan(x).any(): print("NaN in decoder conv output")
        return x


