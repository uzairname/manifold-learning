from dataclasses import asdict
from models.other import MLPWithEmbedding
from models.transformer import Transformer
import numpy as np
import torch.nn as nn

from tasks.arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders
from tasks.arithmetic.utils import TrainRunConfig
from functools import partial

from dotenv import load_dotenv
from utils.helpers import CrossEntropyHighPrecision
from utils.trainer import Trainer

load_dotenv()


class ArithmeticTrainer(Trainer):
  def __init__(
    self,    
    data_config: ArithmeticDatasetConfig = ArithmeticDatasetConfig(),
    val_frac: int = None
  ):
    super().__init__()
    self.data_config = data_config
    self.val_frac = val_frac
  
  def get_data(self, c, rank=None, get_val_data=True):
    train_dataloader, val_dataloader, train_sampler = get_mod_arithmetic_cp_dataloaders(
      data_config=self.data_config,
      val_frac=self.val_frac,
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
    x, y_int, _ = batch
    return x, y_int
  


if __name__ == "__main__":
  
  p = 113
  d_model=128

  model_class_mlp = MLPWithEmbedding
  model_params_mlp = dict(
    n_vocab=p+1, embed_dim=128, sequence_len=3, hidden_dims=[d_model * 4, d_model * 4]
  )

  for init_scale in [0.01]:
      batch_size = 512
      weight_decay = 1e-2
      use_ln = False

      config = TrainRunConfig(
        model_name=f"arithmetic_mlp",
        model_class=Transformer,
        model_params=dict(
          d_model=d_model,
          d_mlp=d_model * 4,
          n_vocab=p+1,
          max_seq_len=3,
          n_heads=4,
          n_layers=1,
          init_scale=init_scale,
          use_ln=use_ln,
        ),
        # model_class=model_class_mlp,
        # model_params=model_params_mlp,
        n_epochs=40000,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=weight_decay,
        criterion=CrossEntropyHighPrecision(),
        n_evals=128,
        n_checkpoints=32,
        experiment_group="trainer"
      )
      
      trainer = ArithmeticTrainer(
        data_config=ArithmeticDatasetConfig(
          p=p,
          noise_frac=0.0,
        ),
        val_frac=0.7
      )
      
      trainer.train(config)
      
      
