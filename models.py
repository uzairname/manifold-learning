import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        residual = x  # Skip connection
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
    

class DeepAutoencoder(nn.Module):
    input_dim: int
    dim_after_conv: int
        
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



class Autoencoder(nn.Module):
    def __init__(self, hidden_units: int, input_dim: int=65):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = (2 * input_dim) // 64  # After 5 Conv Layers

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            ResidualBlock(256),  # Residual Connection

            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # 4x4 -> 2x2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Flatten(),
            nn.Linear(256 * self.dim_after_conv * self.dim_after_conv, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, hidden_units)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_units, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256 * self.dim_after_conv * self.dim_after_conv),
            nn.ReLU(),
            nn.Unflatten(1, (256, self.dim_after_conv, self.dim_after_conv)),
            nn.Dropout(0.4),

            # Upsample and ConvTranspose Layers
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 2x2 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),  # 4x4 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 4x4 -> 8x8

            nn.ConvTranspose2d(256, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16 -> 32x32

            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output in range [0,1]
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent






class ConvHead(nn.Module):

    def __init__(self, input_dim=64):
        super(ConvHead, self).__init__()

        self.input_dim = input_dim
        self.dim_after_conv = 2 * input_dim // 64  # After Conv Layers

        self.conv_head = nn.Sequential(
            ResidualBlock(32),

            nn.MaxPool2d(kernel_size=2, stride=2), # 32x32 -> 16x16

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            ResidualBlock(32),

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