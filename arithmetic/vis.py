from typing import *
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt



def visualize_data(dataloader: DataLoader):
  
  data = [i for i in dataloader]
  
  # Extract all (a,b,c) tuples from train_data
  all_a = []
  all_b = []
  all_c = []

  for batch in data:
    all_a.append(batch[0])
    all_b.append(batch[1])
    all_c.append(batch[2])

  # Concatenate all batches
  all_a_tensor = torch.cat(all_a)
  all_b_tensor = torch.cat(all_b)
  all_c_tensor = torch.cat(all_c)

  # Convert to numpy for easier handling
  all_a_np = all_a_tensor.numpy()
  all_b_np = all_b_tensor.numpy()
  all_c_np = all_c_tensor.numpy()
  
  print("Number of samples:", all_a_np.shape[0])

  # Scatter plot of the first 1000 points
  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  plt.scatter(all_a_np.flatten(), all_b_np.flatten(), c=all_c_np.flatten(), 
        cmap='viridis', alpha=0.7, s=10)
  plt.colorbar(label='c value')
  plt.xlabel('a value')
  plt.ylabel('b value')
  plt.title('Relationship between a, b, and c')

  plt.tight_layout()
  plt.show()