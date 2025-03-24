from typing import *
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt



def visualize_data(dataloader: DataLoader):

  data = [i for i in dataloader]

  # Extract all (a,b,c) tuples from train_data

  all_x = []
  all_y = []

  for batch in data:
    all_x.append(batch[0])
    all_y.append(batch[1])

  # Concatenate all batches
  all_x_np = torch.cat(all_x).numpy()
  all_y_np = torch.cat(all_y).numpy()
  
  print("Number of samples:", all_x_np.shape[0])

  # Scatter plot of the first 1000 points
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.scatter(all_x_np[:, 0], all_x_np[:, 1], c=all_y_np, cmap='viridis', alpha=0.7, s=10)
  plt.colorbar(label='c value')
  plt.xlabel('a value')
  plt.ylabel('b value')
  plt.title('Relationship between a, b and y')

  plt.tight_layout()
  plt.show()