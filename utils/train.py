import typing as t
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import neptune

@dataclass
class BaseTrainRunConfig:

  # model
  model_class: nn.Module
  model_partial: t.Optional[t.Callable[[], nn.Module]] = None
  model_params: t.Optional[dict] = None

  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  get_optimizer: t.Optional[t.Callable] = None
  learning_rate: float = 1e-3
  weight_decay: float = 0
  criterion: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1
  
  # multiprocessing
  max_gpus: int = None
  rank: int = None
  distributed: bool = True
  world_size: int = None
  
  # checkpointing
  n_evals: int = 64
  n_checkpoints: int = 0
  save_method: t.Literal["state_dict", "trace", "script"] = "state_dict"
  checkpoint_dir_name: str = "checkpoints"
  
  # logging
  run: t.Optional[neptune.Run] = None
  log: bool = True
  experiment_group: t.Optional[str] = None
  run_name: t.Optional[str] = None
  notes: t.Optional[str] = None
  tags: t.Optional[list[str]] = None

  

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrossEntropyHighPrecision(nn.Module):
  """
  From https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/helpers.py#L105
  """
  def __init__(self):
    super(CrossEntropyHighPrecision, self).__init__()

  def forward(self, logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float32), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss
