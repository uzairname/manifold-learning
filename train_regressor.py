import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd

from models import ClockRegressor
from data import IMG_DIR, ClockDataset

if __name__ == "__main__":

  BATCH_SIZE = 32

  # Load supervised dataset
  train_dataset = ClockDataset(img_dir=IMG_DIR, supervised=True)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

  LEARNING_RATE = 0.0002

  regressor = ClockRegressor(out_dim=2)
  criterion = nn.MSELoss()
  optimizer = torch.optim.AdamW(regressor.parameters(), lr=LEARNING_RATE)

  # Training loop for regressor
  num_epochs = 3
  for epoch in range(num_epochs):
      for batch, labels_2d, labels_1d in train_loader:
          # Forward pass
          outputs = regressor(batch)
          loss = criterion(outputs, labels_2d)
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

