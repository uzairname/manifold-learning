from typing import *
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json

from tasks.arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders
from utils.train import BaseTrainRunConfig


@dataclass
class TrainRunConfig(BaseTrainRunConfig):
  data_config: ArithmeticDatasetConfig = None
  val_frac: int = None
  
  
  
@dataclass
class ModelCheckpoint:
  model: nn.Module
  dataloader: DataLoader
  id: Optional[str] = None
  step: Optional[int] = None
  epoch: Optional[int] = None
  val_loss: Optional[float] = None
  


def eval_model(
  model: nn.Module,
  loss_fn: nn.Module,
  val_data: list,
  device: str,
):
  """
  Evaluates a model from a training run.
  """
  model.eval()
  
  val_loss = 0
  with torch.no_grad():
    for batch in val_data:
        x, y, _ = batch
        x = x.to(device)
        y = y.to(device)
        
        pred = model(x) # take the last token prediction
        loss = loss_fn(pred, y)
        
        val_loss += loss.item()
        
  return val_loss / len(val_data)


def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  val_data: list,
):
  """
  Evaluates and saves a model from a training run.
  """

  model.eval()
  # save the model
  if c.save_method == "script":
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "state_dict":
    torch.save(model.state_dict(), path)
  else:
    raise ValueError(f"Unknown save method {c.save_method}")

  # Evaluate the model test loss
  val_loss = eval_model(model=model, loss_fn=c.criterion, val_data=val_data, device=device)

  return val_loss




def load_model_and_dataset(
  model_class: nn.Module,
  model_dir: str,
  checkpoint=None,
  device='cuda',
  batch_size=512
)-> ModelCheckpoint:
  """
  Loads a clock model by state dict and architecture
  """

  if checkpoint is None:
    cp_path = os.path.join(model_dir, f"final.pt")
    cp_info_path = os.path.join(model_dir, "final.json")
  else:
    cp_path = os.path.join(model_dir, f"{checkpoint}.pt")
    cp_info_path = os.path.join(model_dir, f"{checkpoint}.json")
  
  state_dict = torch.load(cp_path, map_location=device)
  if 'model' in state_dict:
    state_dict = state_dict['model']

  with open(os.path.join(model_dir, 'model_params.json'), 'r') as f:
    run_data = json.load(f)
    model_params = run_data.get('model_params', {})
    data_config_dict = run_data['data_config']
    data_config = ArithmeticDatasetConfig(**data_config_dict)
    
  if checkpoint is not None:
    with open(os.path.join(model_dir, f"{checkpoint}.json"), 'r') as f:
      checkpoint_data = json.load(f)
      step = checkpoint_data.get('step', None)
      val_loss = checkpoint_data.get('val_loss', None)
  else:
    step = None
    val_loss = None


  dataloader, _, _ = get_mod_arithmetic_cp_dataloaders(data_config=data_config, batch_size=batch_size)

  model = model_class(**model_params).to(device)

  model.load_state_dict(state_dict)
  model.eval()

  return ModelCheckpoint(
    id=checkpoint,
    model=model,
    dataloader=dataloader,
    val_loss=val_loss,
    step=step,
  )