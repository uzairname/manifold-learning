import typing 
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, RandomSampler

from utils.utils import is_prime

"""
Datasets of arithmetic problems
"""

@dataclass
class ArithmeticDatasetConfig:
  """
  Configuration for arithmetic datasets
  """
  p: int = 113  # prime number for modular arithmetic
  noise_frac: float = 0.0  # fraction of labels to be noise

class ModArithmeticCpDataset(Dataset):
  """
  Modular arithmetic on C_p
  parameters:
  - p: prime number

  Data points consist of a tuple (x, y) where 
  - x is a sequence "ab" where a and b are integers in [0, p-1] 
  - y is the result of (a + b) mod p, one-hot encoded.
  """
  
  def __init__(self, config: ArithmeticDatasetConfig):
    p = config.p
    assert is_prime(p), f"p={p} is not prime"
    self.p = p
    
    # Generate data
    all_a = torch.arange(p).repeat_interleave(p).reshape(p**2, 1) # token 1
    all_b = torch.arange(p).repeat(p, 1).reshape(p**2, 1) # token 2
    third_token = torch.ones((p**2, 1), dtype=torch.int64)*p # token 3
    
    self.x = torch.cat((all_a, all_b, third_token), dim=1)

    y_int = (torch.arange(p).unsqueeze(1) + torch.arange(p).unsqueeze(0)) % p
    
    # assign noise to some labels
    num_noisy_labels = int(config.noise_frac * y_int.numel())
    noisy_indices = torch.randperm(y_int.numel())[:num_noisy_labels]
    random_labels = torch.randint(0, p, (num_noisy_labels,))
    y_int = y_int.flatten()
    y_int[noisy_indices] = random_labels

    self.y_int = y_int.reshape(p**2)
    self.y_one_hot = F.one_hot(self.y_int, num_classes=p+1).float()

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y_int[idx], self.y_one_hot[idx]


def get_mod_arithmetic_cp_dataloaders(
  data_config: ArithmeticDatasetConfig,
  train_frac: int = 1,
  max_val_size: int = 2**16,
  batch_size: int = 64,
  world_size: int= 1,
  rank: int = None,
  drop_last: bool = False,
):
    """
    Get a dataset of modular arithmetic on C_p
    """
    dataset = ModArithmeticCpDataset(config=data_config)
    
    g = torch.Generator()
    g.manual_seed(42)

    train_size = int(len(dataset) * train_frac)
    test_size = int(min(len(dataset) - train_size, max_val_size))

    train_dataset, val_dataset, _  = random_split(
      dataset,
      [train_size, test_size, len(dataset) - train_size - test_size],
      generator=g
    )

    # Distributed sampler if applicable 
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
      
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=drop_last, num_workers=1, persistent_workers=True)
    test_dataloader = None
    if len(val_dataset) > 0:
      test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=1, persistent_workers=True)

    return train_dataloader, test_dataloader, train_sampler



__all__ = [
  "ArithmeticDatasetConfig",
  "ModArithmeticCpDataset",
  "get_mod_arithmetic_cp_dataloaders",
]