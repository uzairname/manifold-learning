import torch.nn as nn
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from models.transformer import Transformer
from tasks.arithmetic import ArithmeticDatasetConfig, ArithmeticTrainRunConfig, ArithmeticTrainer
from utils.helpers import CrossEntropyHighPrecision
import run_configs


if __name__ == "__main__":

    c = run_configs.baseline
    c.experiment_group = "train-frac-sweep"
    
    for train_frac in np.arange(0.2, 0.8, 0.1):
      c.train_frac = train_frac
      c.n_checkpoints = 2

      trainer = ArithmeticTrainer(c)
      trainer.train()

