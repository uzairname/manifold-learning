from typing import TypeVar, Generic
import numpy as np
import torch.nn as nn

from dataclasses import asdict, dataclass
from models.other import MLPWithEmbedding
from models.transformer import Transformer

from dotenv import load_dotenv
load_dotenv()

from tasks.arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders
from tasks.arithmetic.utils import BaseTrainRunConfig
from functools import partial

from utils.helpers import CrossEntropyHighPrecision
from utils.trainer import Trainer


@dataclass
class ArithmeticTrainRunConfig(BaseTrainRunConfig):
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
  

if __name__ == "__main__":
  
  p = 113
  val_frac = 0.7
  data_size = int(p ** 2 * (1 - val_frac))
  d_model=32

  use_ln = False

  config = ArithmeticTrainRunConfig(
    model_name=f"arithmetic_transformer",
    checkpoint_dir_name=f'p{p}_d{d_model}_fb_ce',
    model_class=Transformer,
    model_params=dict(
      d_model=d_model,
      d_mlp=d_model * 4,
      n_vocab=p+1,
      max_seq_len=3,
      n_heads=4,
      n_layers=1,
      init_scale=1.0,
      use_ln=use_ln,
    ),
    data_config=ArithmeticDatasetConfig(
      p=p,
      noise_frac=0.0,
    ),
    batch_size=data_size, # Full batch training
    learning_rate=1e-3*4, # Multiply by num of GPUs
    weight_decay=1e-0,
    n_epochs=20000,
    criterion=nn.CrossEntropyLoss(),
    n_evals=128,
    n_checkpoints=32,
    
    val_frac=val_frac,
  )
  
  trainer = ArithmeticTrainer(config)
  trainer.train()
