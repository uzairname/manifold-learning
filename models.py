import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils import spectral_norm
import numpy as np


class ConvResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.skip(x)  # Skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        out += residual  # Add skip connection
        return out
    
    
class ResNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Transposed Conv for upsampling by 2x
        self.conv1 = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))  # 2x upsample
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))  # Refinement

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (upsampling by 2 using interpolation)
        self.skip = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1))
        )
        
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = self.skip(x)  # Skip connection (upsampling)

        x = self.conv1(x)  # Upsample by 2x
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)  # Keeps size the same
        x = self.norm2(x)
      
        x += identity  # Add skip connection
        x = self.activation(x)
        
        return x


class MLPResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
      
        if out_dim is None:
            out_dim = in_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # skip connection
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = self.skip(x)
        out = self.mlp(x)
        out += residual
        return out



class DeepAutoencoder(nn.Module):
    input_dim: int
    dim_after_conv: int
        
    def __init__(self, latent_dim=2, img_size=128, **kwargs):
        super(DeepAutoencoder, self).__init__()

        self.input_dim = img_size
        self.dim_after_conv = 2 * img_size // 128  # After 5 Conv Layers
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.MaxPool2d(kernel_size=2, stride=4),  # 64x64 -> 16x16

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            ConvResidualBlock(256, 256), # Residual Connection

            nn.MaxPool2d(kernel_size=2, stride=4),  # 8x8 -> 2x2

            # Flatten and Fully Connected Layers
            nn.Flatten(),
            
            nn.Linear(256 * self.dim_after_conv * self.dim_after_conv, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            MLPResidualBlock(256, 128),

            nn.Linear(128, latent_dim),
        )
        
        resnet_start_channels = 512

        # Decoder (Mirroring the Encoder)
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128 * self.dim_after_conv * self.dim_after_conv),
            nn.ReLU(),
            nn.Unflatten(1, (128, self.dim_after_conv, self.dim_after_conv)), # 2x2
        )
        
        self.decoder_init = nn.Sequential( 
            spectral_norm(nn.ConvTranspose2d(128, resnet_start_channels, kernel_size=3, stride=1, padding=1)),  # 2x2 -> 2x2
            nn.BatchNorm2d(resnet_start_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder_conv = nn.Sequential(
            # Upsampling steps (each block doubles the resolution)
            ResNetDecoderBlock(resnet_start_channels, resnet_start_channels // 2),  # 2x2 -> 4x4
            ResNetDecoderBlock(resnet_start_channels // 2, resnet_start_channels // 4),  # 4x4 -> 8x8
            ResNetDecoderBlock(resnet_start_channels // 4, resnet_start_channels // 8),  # 8x8 -> 16x16
            ResNetDecoderBlock(resnet_start_channels // 8, resnet_start_channels // 16),  # 16x16 -> 32x32
            ResNetDecoderBlock(resnet_start_channels // 16, resnet_start_channels // 32),  # 32x32 -> 64x64
            ResNetDecoderBlock(resnet_start_channels // 32, resnet_start_channels // 64),  # 64x64 -> 128x128
            nn.Sigmoid()  # Output in range [0,1]
        )

        self.final_conv = nn.Conv2d(resnet_start_channels // 64, 1, kernel_size=3, stride=1, padding=1) # 128x128 -> 128x128
        # Final output layer

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder_fc(latent)
        reconstructed = self.decoder_init(reconstructed)
        reconstructed = self.decoder_conv(reconstructed)
        reconstructed = self.final_conv(reconstructed)
        return reconstructed, latent



# Total parameters ~ 7M for 128 input dim
class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int=2, img_size: int=128, **kwargs):
        super(Autoencoder, self).__init__()

        self.input_dim = img_size
        self.dim_after_conv = (4 * img_size) // 128  # After 5 Conv Layers

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.MaxPool2d(kernel_size=2, stride=4),  # 64x64 -> 16x16

            ConvResidualBlock(16, 4),  # 16x16 -> 16x16

            nn.MaxPool2d(kernel_size=2, stride=4),  # 16x16 -> 4x4

            nn.Flatten(),
            
            MLPResidualBlock(4 * self.dim_after_conv**2, self.input_dim // 2),

            nn.Linear(self.input_dim // 2, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.input_dim // 2),
            nn.BatchNorm1d(self.input_dim // 2),
            nn.ReLU(),
            
            MLPResidualBlock(self.input_dim // 2, 4 * self.dim_after_conv**2),
            
            nn.Unflatten(1, (4, self.dim_after_conv, self.dim_after_conv)),

            # Upsample and ConvTranspose Layers
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),  # 4x4 -> 16x16

            ConvResidualBlock(4, 16),  # 16x16 -> 16x16

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16 -> 32x32
            
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64x64 -> 128x128
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 128x128 -> 128x128
            
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent








# Encoder: CNN extracts a latent representation from the image
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=2, input_dim=128):
        super().__init__()
        
        self.dim_after_conv = (4 * input_dim) // 128  # After 5 Conv Layers

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.MaxPool2d(kernel_size=2, stride=4),  # 64x64 -> 16x16

            ConvResidualBlock(256, 32),  # 16x16 -> 16x16

            nn.MaxPool2d(kernel_size=2, stride=4),  # 16x16 -> 4x4

            nn.Flatten(),
            
            MLPResidualBlock(32 * self.dim_after_conv**2, input_dim // 2),

            nn.Linear(input_dim // 2, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class FiLM(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.gamma = nn.Linear(m, n)
        self.beta = nn.Linear(m, n)

    def forward(self, x, latent):
        gamma = self.gamma(latent).unsqueeze(1)  # (B, 1, n)
        beta = self.beta(latent).unsqueeze(1)    # (B, 1, n)
        out = gamma * x + beta # B, num_pixels, 1
        return out #


class INRDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim=128, out_channels=1, hidden_dim=256, num_layers=4, fourier_features=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fourier_features = fourier_features
        
        # Learnable frequencies for Fourier features
        self.freqs = nn.Parameter(torch.randn(fourier_features, 2) * 2 * np.pi)

        self.fc_in = nn.Linear(latent_dim + 2 * fourier_features, hidden_dim)
        self.hidden_layers = nn.Sequential(
            *[(nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords, z):
        """
        coords: (B, N, 2) - Spatial coordinates
        z: (B, latent_dim) - Latent representation
        """
        B, N, _ = coords.shape

        encoded_coords = torch.cat([
            torch.sin(coords @ self.freqs.T),
            torch.cos(coords @ self.freqs.T)
        ], dim=-1)
        torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1) if self.fourier_features > 0 else coords

        # Expand latent vector to match coordinate dimensions
        z = z.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)
        inputs = torch.cat([encoded_coords, z], dim=-1)  # Concatenate (B, N, latent_dim + 2 * fourier_features)
        
        x = self.fc_in(inputs) # (B, H*W, hidden_dim)
        x = self.hidden_layers(x) # (B, H*W, hidden_dim)
        x = self.fc_out(x) # (B, H*W, output_dim)
        
        # Reshape to (B, C, H, W)
        C = out.shape[-1]  # Number of output channels
        out = out.view(B, self.output_dim, self.output_dim, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class ConvInrAutoencoder(nn.Module):
    def __init__(self, img_size=128, latent_dim=16, device='cuda'):
        super().__init__()
        self.input_dim = img_size
        self.encoder = ConvEncoder(latent_dim, input_dim=img_size)
        self.decoder = INRDecoder(latent_dim, output_dim=img_size, out_channels=1, fourier_features=8)
        
        x = torch.linspace(-1, 1, img_size).to(device)
        y = torch.linspace(-1, 1, img_size).to(device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.coords = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)  # (H*W, 2)
    
    def forward(self, x):
        """
        x: (B, 1, H, W) - Input grayscale image
        """
        B = x.size(0)
        z = self.encoder(x)  # (B, latent_dim)

        # Expand coordinates to match batch size
        coords = self.coords.unsqueeze(0).expand(B, -1, -1) # (B, H*W, 2)

        # Decode intensities per pixel (B, H*W, C)
        out = self.decoder(coords, z) # (B, H*W, 1)

        # Reshape back to original image dimensions
        return out, z









class ConvHead(nn.Module):

    def __init__(self, input_dim=64):
        super(ConvHead, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After Conv Layers

        self.conv_head = nn.Sequential(
            ConvResidualBlock(32),

            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            ConvResidualBlock(32),

            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8 -> 4x4

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * self.dim_after_conv * self.dim_after_conv, 64),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv_head(x)
        return out


class DeepMultiHeadRegressor(nn.Module):

    def __init__(self, out_dim=2, input_dim=64):
        super(DeepMultiHeadRegressor, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After Conv Layers

        self.conv1 =  nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        
        self.head1 = ConvHead(input_dim=input_dim)
        self.head2 = ConvHead(input_dim=input_dim)

        self.shared_fc = nn.Sequential(
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            nn.Linear(4, out_dim),
            nn.Sigmoid()
        )
  

    def forward(self, x):
        
        for layer in self.conv1:
            x = layer(x)

        out1 = self.head1(x)
        out2 = self.head2(x)

        out = torch.cat((out1, out2), dim=1)
        out = self.shared_fc(out)
        
        return out



# Small fully connected network
class MLP(nn.Module):

    def __init__(self, input_dim, out_dim=1):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        out = self.fc(x)
        return out
    

    def fit(self, x, y, epochs=1000, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return self