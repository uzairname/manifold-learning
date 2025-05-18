import torch.nn as nn
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from models.transformer import Transformer
from tasks.arithmetic import ArithmeticDatasetConfig, ArithmeticTrainRunConfig
from utils.helpers import CrossEntropyHighPrecision
from utils.utils import iife

"""
Configuration to train a standard mod arithmetic transformer, same setup as the paper. runs to 20k epochs
results: No loss spikes, grokking behavior, final train loss 4e-8
"""
@iife
def baseline():
  p = 113
  d_model = 128
  return ArithmeticTrainRunConfig(
    model_name=f"arithmetic_transformer",
    model_class=Transformer,
    model_params=dict(
      d_model=128,
      d_mlp=d_model * 4,
      n_vocab=p+1,
      max_seq_len=3,
      n_heads=4,
      n_layers=1,
      init_scale=1.0,
      use_ln=False,
    ),
    data_config=ArithmeticDatasetConfig(
      p=p,
      noise_frac=0.0
    ),
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-0,
    n_epochs=20000,
    criterion=CrossEntropyHighPrecision(),
    n_evals=128,
    n_checkpoints=32,
    train_frac=0.3,
    checkpoint_dir="baseline"
  )
