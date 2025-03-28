from typing import *
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json

from tasks.arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders
from utils.data_types import TrainRunConfig


  
  
@dataclass
class ModelCheckpoint:
  model: nn.Module
  dataloader: DataLoader
  id: Optional[str] = None
  step: Optional[int] = None
  epoch: Optional[int] = None
  val_loss: Optional[float] = None
  


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