from typing import *
from dataclasses import dataclass
from pydantic import BaseModel, ValidationError
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tasks.arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders


class CheckpointMetadata(BaseModel):
  step: Optional[int] = None
  epoch: Optional[int] = None
  val_loss: Optional[float] = None
  batch: Optional[int] = None

@dataclass
class ModelCheckpoint:
  model: nn.Module
  dataloader: DataLoader
  id: Optional[str] = None
  data: Optional[CheckpointMetadata] = None




def load_model_checkpoint(
  model_class: nn.Module,
  get_dataloaders: Callable[[], Tuple[DataLoader, DataLoader, Optional[torch.utils.data.Sampler]]],
  checkpoint_dir: str,
  checkpoint=None,
  device='cuda',
  batch_size=512
)-> ModelCheckpoint:
  """
  Loads a clock model by state dict and architecture
  """
  
  checkpoint_name = checkpoint if checkpoint is not None else "final"
  model_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
  info_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")

  state_dict = torch.load(model_path, map_location=device)
  if 'model' in state_dict:
    state_dict = state_dict['model']

  with open(os.path.join(checkpoint_dir, 'model_params.json'), 'r') as f:
    run_data = json.load(f)
    model_params = run_data.get('model_params', {})
    data_config_dict = run_data['data_config']
    
  with open(info_path, 'r') as f:
    metadata = CheckpointMetadata.model_validate(json.load(f))

  dataloader, _, _ = get_dataloaders(data_config=data_config_dict, batch_size=batch_size)

  model = model_class(**model_params).to(device)

  model.load_state_dict(state_dict)
  model.eval()

  return ModelCheckpoint(
    model=model,
    dataloader=dataloader,
    id=checkpoint_name,
    data=metadata
  )
  
__all__ = [
  "ModelCheckpoint",
  "load_model_checkpoint",
]