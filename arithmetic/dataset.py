import typing 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler, RandomSampler
import numpy as np
from utils.utils import is_prime
from dataclasses import dataclass



"""
Datasets of arithmetic problems
"""

@dataclass
class ArithmeticDatasetConfig:
  """
  Configuration for arithmetic datasets
  """
  p: int  # prime number for modular arithmetic
  noise_frac: float = 0.0  # fraction of labels to be noise


class ModArithmeticCpDataset(Dataset):
  """
  Modular arithmetic on C_p
  parameters:
  - p: prime number

  Data points consist of a tuple (a, b, y) where 
  - a, b are integers in [0, p), 1 hot encoded
  - y is the result of (a + b) mod p, 1 hot encoded
  """
  
  def __init__(self, config: ArithmeticDatasetConfig):
    p = config.p
    assert is_prime(p), f"p={p} is not prime"
    self.p = p
    
    # Generate data
    all_a = F.one_hot(torch.arange(p), num_classes=p).repeat(1, p).reshape(p**2, p)
    all_b = F.one_hot(torch.arange(p), num_classes=p).repeat(p, 1).reshape(p**2, p)
    all_y = F.one_hot((torch.arange(p).unsqueeze(1) + torch.arange(p).unsqueeze(0)) % p, num_classes=p).reshape(p**2, p)
    
    self.data = torch.cat((all_a, all_b, all_y), dim=1).reshape(p**2, 3, p)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]



def get_mod_arithmetic_cp_dataloaders(
  data_config: ArithmeticDatasetConfig,
  batch_size: int = 32,
  world_size: int= 1,
  rank: int = None,
  val_frac: int = None
):
    """
    Get a dataset of modular arithmetic on C_p
    """
    dataset = ModArithmeticCpDataset(config=data_config)
    
    
    # Datasets
    if val_frac is not None:
        val_size = int(len(dataset) * val_frac)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        val_sampler = RandomSampler(val_dataset)
    else:
        train_dataset = dataset
        val_dataset = None
       
    # Distributed sampler if applicable 
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
      
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=False, num_workers=4)
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, drop_last=False, num_workers=4)
    else:
        val_dataloader = None
        
    return train_dataloader, val_dataloader, train_sampler

