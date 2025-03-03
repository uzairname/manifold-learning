import torch
import torch.nn as nn

# Autoencoder for 1x128x128 inputs
class DeepAutoencoder(nn.Module):
  def __init__(self, n_hidden):
    super(DeepAutoencoder, self).__init__()
    
    # Encoder: Convolutional layers for feature extraction
    self.encoder_conv = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
    )

    self.encoder_fc = nn.Sequential(
        nn.Linear(512 * 256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, n_hidden)  # Predicting hour and minute
    )

    # Decoder: Fully connected layers for reconstruction
    self.decoder_fc = nn.Sequential(
        nn.Linear(n_hidden, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512 * 256)
    )

    # Decoder: Convolutional layers for image reconstruction
    self.decoder_conv = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)
    )

  def forward(self, x):
    x = self.encoder_conv(x)
    x = torch.flatten(x, start_dim=1)  # Flatten the output
    latent = self.encoder_fc(x)  # Encode to latent space

    x = self.decoder_fc(latent)  # Decode from latent space
    x = x.view(-1, 512, 16, 16)  # Reshape to match decoder input
    x = self.decoder_conv(x)  # Decode to image space
    reconstructed = torch.sigmoid(x)  # Apply sigmoid to ensure pixel values are between 0 and 1
    return reconstructed, latent


# Model 1: Predict (hour, minute) as 2D labels
class ClockRegressor(nn.Module):
    def __init__(self, out_dim=2):
        super(ClockRegressor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)  # Predicting hour and minute
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = torch.sigmoid(x) # Ensure output is between 0 and 1
        return x

