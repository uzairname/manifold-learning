import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import os

from models import DeepAutoencoder
from data import IMG_DIR, IMG_SIZE, ClockDataset

if __name__ == "__main__":

  BATCH_SIZE = 32

  # Load unsupervised dataset for autoencoder
  unsupervised_dataset = ClockDataset(img_dir=IMG_DIR, supervised=False)
  unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=BATCH_SIZE, shuffle=True)

  HIDDEN_UNITS = 2
  LEARNING_RATE = 0.0002

  ae = DeepAutoencoder(n_hidden=HIDDEN_UNITS)
  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(ae.parameters(), lr=LEARNING_RATE)

  # Training loop for autoencoder
  num_epochs = 3
  for epoch in range(num_epochs):
    
      for batch in unsupervised_loader:
          # Forward pass
          reconstructed, _ = ae(batch) # Get reconstructed image, ignore latent
          loss = criterion(reconstructed, batch)
          print(loss.item())
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


  # Save the trained autoencoder model
  os.makedirs('models', exist_ok=True)
  torch.save(ae.state_dict(), 'models/ae.pth')
  print('Autoencoder model saved to ae.pth')


