from datasets.clock import ClockConfig, ClockDatasetConfig

import torch.nn as nn
import wandb.wandb_run
import torch
import io

import typing
from dataclasses import dataclass
import time
import copy


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
  group: str = None
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


def copy_model(trained_model: nn.Module, init_model: typing.Callable[[], nn.Module], device: str) -> nn.Module:
    model_copy = init_model()
    model_copy.load_state_dict(trained_model.state_dict())
    return model_copy.to(device)
  


def eval_model(
  c: TrainRunConfig,
  ddp_model: nn.Module,
  val_data: typing.List,
  device: str,
):
  """
  Evaluates a ddp model from a training run.
  """
  # model_ = copy_model(ddp_model.module, c.model_partial, device)

  model_ = ddp_model.module

  model_.eval()
  
  val_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(val_data):
        _, clean_imgs, labels2d, labels1d = batch
        labels = labels1d.unsqueeze(1) if c.latent_dim == 1 else labels2d

        if c.type == "encoder":
            input = clean_imgs.to(device)
            output = labels.to(device)
        elif c.type == "decoder":
            input = labels.to(device)
            output = clean_imgs.to(device)
        elif c.type == "autoencoder":
            input = clean_imgs.to(device)
            output = clean_imgs.to(device)

            pred = model_(input)
            loss = c.loss_fn(pred, output)
            val_loss += loss.item()

  val_loss /= len(val_data)

  model_.train()
  
  return val_loss



def eval_and_save_model(
  c: TrainRunConfig,
  ddp_model: nn.Module,
  device: str,
  path: str,
  val_data: typing.List,
):
  """
  Evaluates and saves a ddp model from a training run.
  """
  # Unwrap from DDP and save the model using TorchScript trace
  # model_ = copy_model(ddp_model.module, c.model_partial, device)

  model_ = ddp_model.module
  model_.eval()
  # Trace and save the model
  if c.save_method == "trace":
    dummy_input = torch.randn(1, c.latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.dataset_config.img_size, c.dataset_config.img_size).to(device)
    scripted_model = torch.jit.trace(model_, dummy_input)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "script":
    scripted_model = torch.jit.script(model_)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "state_dict":
    torch.save(model_.state_dict(), path)
  else:
    raise ValueError(f"Unknown save method {c.save_method}")

  # Evaluate the model test loss
  val_loss_eval = eval_model(c, ddp_model, val_data, device)

  return val_loss_eval
