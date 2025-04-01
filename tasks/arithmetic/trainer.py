from dataclasses import dataclass

from utils.trainer import Trainer
from utils.data_types import TrainConfig

from .dataset import get_mod_arithmetic_cp_dataloaders

@dataclass
class ArithmeticTrainRunConfig(TrainConfig):
  val_frac: float = 0.7  # Fraction of data to use for validation


class ArithmeticTrainer(Trainer[ArithmeticTrainRunConfig]):
  def __init__(self, c: ArithmeticTrainRunConfig):
    super().__init__(c)
  
  def get_data(self, c, rank=None, get_val_data=True):
    train_dataloader, val_dataloader, train_sampler = get_mod_arithmetic_cp_dataloaders(
      data_config=c.data_config,
      val_frac=c.val_frac,
      batch_size=c.batch_size,
      world_size=c.world_size,
      rank=rank,
    )
    
    if get_val_data:
      val_data = [batch for batch in val_dataloader]
    else:
      val_data = None
      
    return train_dataloader, val_data, train_sampler
  
  def get_inputs_labels(self, batch, s, c):
    x, y_int, y_one_hot = batch
    return x, y_int
  
  
__all__ = [
  "ArithmeticTrainer",
  "ArithmeticTrainRunConfig",
]