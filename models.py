import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  # Skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual  # Add skip connection
        return out
    

class DeepAutoencoder(nn.Module):
        
    def __init__(self, hidden_units, input_dim=64):
        super(DeepAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After 5 Conv Layers
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            ResidualBlock(256), # Residual Connection

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Flatten and Fully Connected Layers
            nn.Flatten(),
            nn.Linear(512 * self.dim_after_conv * self.dim_after_conv, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, hidden_units),
        )

        # Decoder (Mirroring the Encoder)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512 * self.dim_after_conv * self.dim_after_conv),
            nn.ReLU(),
            nn.Unflatten(1, (512, self.dim_after_conv, self.dim_after_conv)),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2x2 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),

            ResidualBlock(256),  # Residual Connection

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# Model 1: Predict (hour, minute) as 2D labels
class ClockRegressor(nn.Module):
    def __init__(self, out_dim=2, input_dim=64):
        super(ClockRegressor, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After 5 Conv Layers
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            ResidualBlock(64),  # Residual Connection

            nn.Conv2d(64, 512, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # maxpooling
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8 -> 4x4

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # 4x4 -> 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.dim_after_conv * self.dim_after_conv, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Linear(8, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x2

    def get_latent(self, x):
        x = self.conv(x)
        x1 = self.fc1(x)
        return x1


# Model 2: Autoencoder on ClockRegressor's intermediate layer
class ClockRegressorAE(nn.Module):
    def __init__(self, hidden_units=2, input_dim=64):
        super(ClockRegressorAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, hidden_units)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    



