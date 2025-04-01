from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig, get_dataloaders

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
from utils.data_types import TrainConfig 


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
  c: TrainConfig,
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
  val_loss = eval_model(model=model, type_=c.type, latent_dim=latent_dim, loss_fn=c.criterion, val_data=val_data, device=device)

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
  val_loss: float
  step: int
  

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
    run_data = json.load(f)
    model_params = run_data.get('model_params', {})
    type_ = run_data['type']
    dataset_config = run_data['dataset_config']
    data_config = run_data['data_config']
    
  if checkpoint is not None:
    with open(os.path.join(model_dir, f"{checkpoint}.json"), 'r') as f:
      checkpoint_data = json.load(f)
      step = checkpoint_data.get('step', None)
      val_loss = checkpoint_data.get('val_loss', None)
  else:
    step = None
    val_loss = None


  dataloader, val_dataloader, _, _ = get_dataloaders(
    data_config=ClockDatasetConfig(**data_config),
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
    val_loss=val_loss,
    step=step,
  )
