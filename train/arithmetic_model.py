from datasets.arithmetic import ArithmeticDatasetConfig
from models.transformer import Transformer
import numpy as np
import torch.nn as nn

from tasks.arithmetic.utils import TrainRunConfig
from tasks.arithmetic import train_arithmetic_model

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
  batch_size = 512
  p = 113
  d_model=128

  for noise_frac in [0.0]:
    config = TrainRunConfig(
      val_frac=0.7,
      model_class=Transformer,
      model_params=dict(
        d_model=d_model,
        d_mlp=d_model * 4,
        n_vocab=p,
        max_seq_len=2,
        n_heads=4,
        n_layers=1,
        init_scale=0.1,
      ),
      data_config=ArithmeticDatasetConfig(
        p=p,
        noise_frac=noise_frac,
      ),
      n_epochs=1000,
      batch_size=batch_size,
      learning_rate=1e-3,
      weight_decay=1e-0,
      loss_fn=nn.CrossEntropyLoss(),
      n_checkpoints=0,
      n_eval=256,
      experiment_group="B", # Group label for the experiment
    )
    train_arithmetic_model(config)
