from datasets.clock import ClockConfig, ClockDatasetConfig, get_dataloaders

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb.wandb_run
import torch
import io
import os
import json

import typing
from dataclasses import dataclass
import time
import copy
import neptune

from utils.config import MODELS_DIR  


@dataclass
class TrainRunConfig:
  # model
  model_class: nn.Module
  type: typing.Literal["autoencoder", "encoder", "decoder"]
  model_partial: typing.Callable = None
  model_params: dict = None

  # multiprocessing
  max_gpus: int = None
  rank: int = None
  distributed: bool = True
  world_size: int = None

  # logging
  log: bool = True
  run: wandb.wandb_run.Run | None = None
  experiment_group: str = None
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
  learning_rate: float = None
  weight_decay: float = None
  loss_fn: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1

  # checkpointing
  n_eval: int = 64
  n_checkpoints: int = None
  save_method: typing.Literal["state_dict", "trace", "script"] = "state_dict"
  label: str = ""


def copy_model(trained_model: nn.Module, init_model: typing.Callable[[], nn.Module], device: str) -> nn.Module:
    model_copy = init_model()
    model_copy.load_state_dict(trained_model.state_dict())
    return model_copy.to(device)
  


def eval_model(
  model: nn.Module,
  type_: str,
  latent_dim: int,
  loss_fn: nn.Module,
  val_data: typing.List,
  device: str,
):
  """
  Evaluates a model from a training run.
  """
  # model_ = copy_model(ddp_model.module, c.model_partial, device)
  model.eval()
  
  val_loss = 0
  with torch.no_grad():
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

        pred = model(input)
        loss = loss_fn(pred, output)
        val_loss += loss.item()

  val_loss /= len(val_data)

  model.train()
  
  return val_loss



def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  val_data: typing.List,
):
  """
  Evaluates and saves a model from a training run.
  """
  # Unwrap from DDP and save the model using TorchScript trace
  # model_ = copy_model(ddp_model.module, c.model_partial, device)
  latent_dim = c.model_params['latent_dim']

  model.eval()
  # Trace and save the model
  if c.save_method == "trace":
    dummy_input = torch.randn(1, latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.dataset_config.img_size, c.dataset_config.img_size).to(device)
    scripted_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "script":
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "state_dict":
    torch.save(model.state_dict(), path)
  else:
    raise ValueError(f"Unknown save method {c.save_method}")

  # Evaluate the model test loss
  val_loss = eval_model(model=model, type_=c.type, latent_dim=latent_dim, loss_fn=c.loss_fn, val_data=val_data, device=device)

  return val_loss



@dataclass
class ModelCheckpoint:
  id: int
  type: typing.Literal['encoder', 'autoencoder', 'decoder']
  model: nn.Module
  latent_dim: int
  img_size: int
  dataloader: DataLoader
  val_dataloader: DataLoader
  

def load_model_and_dataset(
  model_class: nn.Module,
  model_dir: str,
  checkpoint=None,
  device='cuda'
)-> ModelCheckpoint:
  """
  Loads a clock model by state dict and architecture
  """

  if checkpoint is None:
    model_path = os.path.join(model_dir, f"final.pt")
  else:
    model_path = os.path.join(model_dir, f"{checkpoint}.pt")
  
  state_dict = torch.load(model_path, map_location=device)
  if 'model' in state_dict:
    state_dict = state_dict['model']

  with open(os.path.join(model_dir, 'model_params.json'), 'r') as f:
    checkpoint_data = json.load(f)
    model_params = checkpoint_data.get('model_params', {})
    type_ = checkpoint_data['type']
    dataset_config = checkpoint_data['dataset_config']
    data_config = checkpoint_data['data_config']
    
  dataloader, val_dataloader, _, _ = get_dataloaders(
    data_config=ClockConfig(**data_config),
    dataset_config=ClockDatasetConfig(**dataset_config),
    batch_size=64,
  )

  model = model_class(**model_params).to(device)

  model.load_state_dict(state_dict)
  model.eval()

  return ModelCheckpoint(
    id=checkpoint,
    type=type_,
    model=model,
    latent_dim=model_params['latent_dim'],
    img_size=model_params['img_size'],
    dataloader=dataloader,
    val_dataloader=val_dataloader,
  )
