import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, random_split

from utils.trainer import Trainer
from utils.data_types import TrainConfig


@dataclass
class PointsTrainRunConfig(TrainConfig):
  train_frac: float = 0.3



@dataclass
class PointDatasetConfig:
  xy_min: float = -1.0
  xy_max: float = 1.0
  num_points: int = 100


class PointDataset:
  def __init__(self, config: PointDatasetConfig):
    self.config = config

  def __len__(self):
    return self.config.num_points

  def __getitem__(self, idx):
    return torch.rand(2) * (self.config.xy_max - self.config.xy_min) + self.config.xy_min


def get_dataloaders(
  data_config: PointDatasetConfig,
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
    dataset = PointDataset(config=data_config)

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
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=drop_last, num_workers=1, persistent_workers=True)
    test_dataloader = None
    if len(val_dataset) > 0:
      test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=drop_last, num_workers=1, persistent_workers=True)

    return train_dataloader, test_dataloader, train_sampler



class PointsTrainer(Trainer[PointsTrainRunConfig]):
  def __init__(self, c: PointsTrainRunConfig):
    self.metadata = dict(
      train_frac=c.train_frac,
    )
    super().__init__(c)
  
  def get_data(self, c, rank=None, get_val_data=True):
    train_dataloader, val_dataloader, train_sampler = get_dataloaders(
      data_config=c.data_config,
      train_frac=c.train_frac,
      batch_size=c.batch_size,
      world_size=self.world_size,
      rank=rank,
      drop_last=False
    )
    
    if get_val_data:
      val_data = [batch for batch in val_dataloader]
    else:
      val_data = None
      
    return train_dataloader, val_data, train_sampler
  
  def get_inputs_labels(self, batch):
    return batch, batch
  
  
__all__ = [
  "PointsTrainer",
  "PointsTrainRunConfig",
]
