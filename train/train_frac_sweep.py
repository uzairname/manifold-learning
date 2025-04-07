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
    c.n_evals = 512
    
    for train_frac in np.arange(0.4, 0.9, 0.05)**2:
      c.train_frac = train_frac

      trainer = ArithmeticTrainer(c)
      trainer.train()

