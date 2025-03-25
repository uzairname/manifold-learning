import typing
from dataclasses import dataclass
import torch.nn as nn
import neptune

@dataclass
class BaseTrainRunConfig:
  # model
  model_class: nn.Module
  model_partial: typing.Callable = None
  model_params: dict = None

  # multiprocessing
  max_gpus: int = None
  rank: int = None
  distributed: bool = True
  world_size: int = None

  # logging
  run: neptune.Run | None = None
  log: bool = True
  experiment_group: str = None
  name: str = None
  notes: str = None
  tags: list[str] = None

  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  optimizer: typing.Callable | None = None
  learning_rate: float = None
  weight_decay: float = None
  loss_fn: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1

  # checkpointing
  n_eval: int = 64
  n_checkpoints: int = None
  save_method: typing.Literal["state_dict", "trace", "script"] = "state_dict"
  label: str = ""