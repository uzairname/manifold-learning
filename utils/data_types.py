from typing import Any, Callable, Literal, Optional, TypeVar
from dataclasses import dataclass

import torch.nn as nn
import torch
import time
import neptune

from tqdm import tqdm

from utils.config import MODELS_DIR


@dataclass
class TrainConfig:

  # model
  model_class: nn.Module
  model_partial: Optional[Callable[[], nn.Module]] = None
  model_params: Optional[dict] = None
 
  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  get_optimizer: Optional[Callable] = None
  learning_rate: float = 1e-3
  weight_decay: float = 0
  criterion: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1
  
  # data
  data_config: Optional[Any] = None

  # multiprocessing
  max_gpus: int = None
  distributed: bool = True
  world_size: int = None

  # checkpointing
  checkpoint_base_dir: str = MODELS_DIR
  n_evals: int = 64
  n_checkpoints: int = 0
  save_method: Literal["state_dict", "trace", "script"] = "state_dict"
  checkpoint_dir: str = "checkpoints"

  # logging
  log: bool = True
  experiment_group: Optional[str] = None
  model_name: Optional[str] = None
  notes: Optional[str] = None
  tags: Optional[list[str]] = None
  
  
TC = TypeVar('TrainConfig', bound=TrainConfig, default=TrainConfig, covariant=True)


@dataclass
class TrainRunState:
  '''
  State of a training run within the training loop
  '''
  model: torch.nn.Module # underlying model, use for evaluation
  training_model: torch.nn.Module # model to train, possibly wrapped in DDP
  train_dataloader: torch.utils.data.DataLoader
  val_data: list

  device: torch.device
  is_primary: bool
  total_steps: int
  eval_frequency: int
  checkpoint_steps: list
  checkpoint_dir: str

  t: tqdm = None
  step: int = 0
  epoch: int = 0
  batch_idx: int = 0
  checkpoint_num: int = 0
  val_loss: float = None
  running_loss: float = 0
  start_time: float = time.time()

  run: Optional[neptune.Run] = None