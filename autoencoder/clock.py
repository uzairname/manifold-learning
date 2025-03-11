import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from datasets.clock import IMG_SIZE, ClockConfig, ClockDataset, ClockDatasetConfig

import torch.nn as nn
import wandb.wandb_run
import torch
import io

import typing
from dataclasses import dataclass, field


@dataclass
class TrainRunConfig:
  # model
  model_class: nn.Module
  type: typing.Literal["autoencoder", "encoder", "decoder"]
  model_partial: typing.Callable = None
  latent_dim: int = 2
  img_size: int = IMG_SIZE
  model_params: dict = None

  # multiprocessing
  max_gpus: int = None
  rank: int = None
  distributed: bool = True
  world_size: int = None

  # logging
  log_wandb: bool = True
  run: wandb.wandb_run.Run | None = None
  name: str = None
  notes: str = None
  tags: list[str] = None

  # data
  dataset_config: ClockDatasetConfig = None
  data_config: ClockConfig = None
  val_size: int = None

  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  optimizer: typing.Callable = None
  learning_rate: float = 1e-4
  weight_decay: float = 0.0
  loss_fn: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1

  # checkpointing
  n_checkpoints: int = None
  save_path_suffix: str = None
  save_method: typing.Literal["state_dict", "trace", "script"] = "state_dict"


def get_dataloaders(
  data_config: ClockConfig=ClockConfig(),
  dataset_config: ClockDatasetConfig=ClockDatasetConfig(),
  val_size: int=None,
  batch_size: int=64,
  world_size: int=1,
  rank: int=None,
  use_workers: bool=True,
):
  """
  Get the clock dataset split into training and validation sets.
  """
  # Dataset
  dataset = ClockDataset(dataset_config=dataset_config, clock_config=data_config)

  # Split into train and val
  if val_size is None:
      val_size = np.min((dataset_config.data_size//8, 2**12))
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

  # Get sampler and dataloader for train data
  if rank is not None:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
  else:
    train_sampler = torch.utils.data.RandomSampler(train_dataset)

  if use_workers:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=4, prefetch_factor=4, persistent_workers=True)
  else:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=0)
  # increase num_workers if GPU is underutilized

  # Get sampler and dataloader for val data
  val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # No need for shuffling
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, drop_last=True, num_workers=4, persistent_workers=True)

  assert len(train_dataloader) > 0, f"Train dataloader is empty (batch_size={batch_size}, dataset size={len(train_dataset)}, world_size={world_size})"
  assert val_size == 0 or len(val_dataloader) > 0, "Validation dataloader is empty"
  return train_dataloader, val_dataloader, train_sampler, val_sampler


def eval_model(
  type_: typing.Literal['encoder', 'autoencoder', 'decoder'],
  model: nn.Module,
  val_data: typing.List,
  device: str,
  criterion: nn.Module,
  latent_dim: int=2,
):
  val_loss_eval = 0
  model.eval()
  for i, batch in enumerate(val_data):
      _, clean_imgs, labels2d, labels1d = batch
      labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d

      if type_ == "encoder":
          input = clean_imgs.to(device)
          output = labels.to(device)
      elif type_ == "decoder":
          input = labels.to(device)
          output = clean_imgs.to(device)
      elif type_ == "autoencoder":
          input = clean_imgs.to(device)
          output = clean_imgs.to(device)

      with torch.no_grad():
          pred = model(input)
          loss = criterion(pred, output)
          val_loss_eval += loss.item()
  val_loss_eval /= len(val_data)

  return val_loss_eval


def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  criterion: nn.Module,
  val_data: typing.List,
):
  # Unwrap from DDP and save the model using TorchScript trace
  temp = io.BytesIO()
  torch.save(model.module.state_dict(), temp)
  temp.seek(0)
  state_dict = torch.load(temp)
  model_copy = c.model_partial().to(device)
  model_copy.load_state_dict(state_dict)

  # Trace and save the model
  model_copy.eval()
  if c.save_method == "trace":
    dummy_input = torch.randn(1, c.latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.img_size, c.img_size).to(device)
    scripted_model = torch.jit.trace(model_copy, dummy_input)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "script":
    scripted_model = torch.jit.script(model_copy)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "state_dict":
    scripted_model = model_copy
    torch.save(scripted_model.state_dict(), path)
  else:
    raise ValueError(f"Unknown save method {c.save_method}")

  # Evaluate the model test loss
  val_loss_eval = eval_model(c.type, model_copy, val_data, device, criterion=criterion, latent_dim=c.latent_dim)

  return val_loss_eval