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
  
  loss_fn = CrossEntropyHighPrecision()

  for noise_frac in [0]:
      config = TrainRunConfig(
        run_name=f"arithmetic_transformer_baseline",
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
          use_ln=False
        ),
        data_config=ArithmeticDatasetConfig(
          p=p,
          noise_frac=noise_frac,
        ),
        n_epochs=20000,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-0,
        criterion=nn.CrossEntropyLoss(),
        n_evals=128,
        n_checkpoints=32,
        max_gpus=4,
      )
      train_arithmetic_model(config)
