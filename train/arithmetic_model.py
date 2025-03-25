from datasets.arithmetic import ArithmeticDatasetConfig
from models.transformer import Transformer
import numpy as np
import torch.nn as nn

from arithmetic.utils import TrainRunConfig
from arithmetic import train_arithmetic_model

from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":
  batch_size = 512
  p = 113

  config = TrainRunConfig(
    val_frac=0.7,
    model_class=Transformer,
    model_params=dict(
      d_model=128,
      n_vocab=p,
      max_seq_len=2,
      n_heads=4,
      n_layers=1,
      d_mlp=512,
    ),
    data_config=ArithmeticDatasetConfig(p=p),
    n_epochs=100,
    batch_size=batch_size,
    learning_rate=1e-3,
    weight_decay=1e-0,
    loss_fn=nn.CrossEntropyLoss(),
    n_checkpoints=16,
    n_eval=64,
  )
  train_arithmetic_model(config)

