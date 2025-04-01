import torch.nn as nn
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from models.transformer import Transformer
from tasks.arithmetic import ArithmeticDatasetConfig, ArithmeticTrainRunConfig, ArithmeticTrainer
from utils.helpers import CrossEntropyHighPrecision


if __name__ == "__main__":
  
  p = 113
  val_frac = 0.7
  n_gpus = 4
  data_size = (int(p ** 2 * (1 - val_frac)) // n_gpus)
  d_model=32

  config = ArithmeticTrainRunConfig(
    model_name=f"arithmetic_transformer",
    checkpoint_dir=f'p{p}_d{d_model}_fb_ce',
    model_class=Transformer,
    model_params=dict(
      d_model=d_model,
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
      noise_frac=0.0,
    ),
    batch_size=512, # Full batch training
    learning_rate=1e-3*4, # Multiply by num of GPUs
    weight_decay=1e-0,
    n_epochs=20000,
    criterion=CrossEntropyHighPrecision(),
    n_evals=128,
    n_checkpoints=32,
    
    val_frac=val_frac,
  )
  
  trainer = ArithmeticTrainer(config)
  trainer.train()
