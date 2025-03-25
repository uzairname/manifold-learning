from dataclasses import dataclass
import typing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.arithmetic import ArithmeticDatasetConfig
from utils.train import BaseTrainRunConfig


@dataclass
class TrainRunConfig(BaseTrainRunConfig):
  data_config: ArithmeticDatasetConfig = None
  val_frac: int = None
  


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
        x, y = batch
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
  val_loss = eval_model(model=model, loss_fn=c.loss_fn, val_data=val_data, device=device)

  return val_loss
