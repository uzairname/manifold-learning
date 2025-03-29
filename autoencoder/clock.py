from datasets.clock import IMG_SIZE, ClockConfig, ClockDatasetConfig

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


def eval_model(
  type_: typing.Literal['encoder', 'autoencoder', 'decoder'],
  model: nn.Module,
  val_data: typing.List,
  device: str,
  criterion: nn.Module,
  latent_dim: int=2,
):
  val_loss = 0
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
          val_loss += loss.item()
  val_loss /= len(val_data)

  return val_loss


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
    dummy_input = torch.randn(1, c.latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.dataset_config.img_size, c.dataset_config.img_size).to(device)
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