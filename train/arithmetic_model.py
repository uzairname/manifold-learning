from models.transformer import Transformer
import numpy as np
import torch.nn as nn

from tasks.arithmetic.dataset import ArithmeticDatasetConfig
from tasks.arithmetic.utils import TrainRunConfig
from tasks.arithmetic import train_arithmetic_model
import functools

from dotenv import load_dotenv
from utils.train import CrossEntropyHighPrecision

load_dotenv()

if __name__ == "__main__":

  p = 113
  d_model=128
  
  for i in range(4):
      batch_size = 512
      weight_decay = 1e-0
      use_ln = False

      if i == 0:
        pass
      if i == 1:
        batch_size = 128
      if i == 2:
        weight_decay = 1e-2
      if i == 3:
        use_ln = True

      config = TrainRunConfig(
        run_name=f"arithmetic_transformer_baseline_337_0.01",
        val_frac=0.7,
        model_class=Transformer,
        model_params=dict(
          d_model=d_model,
          d_mlp=d_model * 4,
          n_vocab=p+1,
          max_seq_len=3,
          n_heads=4,
          n_layers=1,
          init_scale=1,
          use_ln=use_ln,
        ),
        data_config=ArithmeticDatasetConfig(
          p=p,
          noise_frac=0.0,
        ),
        n_epochs=40000,
        batch_size=batch_size,
        learning_rate=1e-3,
        weight_decay=weight_decay,
        criterion=CrossEntropyHighPrecision(),
        n_evals=128,
        n_checkpoints=32,
        experiment_group="E"
      )
      train_arithmetic_model(config)
