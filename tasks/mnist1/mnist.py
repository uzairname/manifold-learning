from dataclasses import dataclass
import io
import wandb
from scripts.delete_unused_columns import wandb_run
import torch
import torch.nn as nn
import typing
import torchvision
from torchvision import transforms 
import torch.distributed as dist
from torch.utils.data import random_split, DataLoader, DistributedSampler
import logging

INPUT_SHAPE = (1, 28, 28)


@dataclass
class TrainRunConfig:
  # model
  model_class: nn.Module
  model_partial: typing.Callable = None
  model_params: dict = None

  # multiprocessing
  rank: int = None
  max_gpus: int = None
  distributed: bool = True
  world_size: int = None

  # logging
  log_wandb: bool = True
  run: wandb_run.Run | None = None
  name: str = None
  notes: str = None
  tags: list[str] = None

  # data
  val_size: int = None

  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  learning_rate: float = 1e-4
  weight_decay: float = 0.0
  loss_fn: nn.Module = nn.CrossEntropyLoss()
  accumulation_steps: int = 1

  # checkpointing
  n_checkpoints: int = None
  save_path_suffix: str = None
  save_method: typing.Literal["state_dict", "trace", "script"] = "state_dict"



def get_mnist_dataloaders(
  batch_size: int=64,
  world_size: int=1,
  rank: int=None,
):
  """
  Get dataloaders for MNIST dataset.
  """
  
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
  
  dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
  
  # DDP samplers
  if rank is None:
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
  else:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

  # Dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)
  
  return train_loader, val_loader, train_sampler, val_sampler


def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  loss_fn: nn.Module,
  val_dataloader: DataLoader,
):
  """
  Evaluate the model on the validation set and save the model
  """
  
  temp = io.BytesIO()
  torch.save(model.module.state_dict(), temp)
  temp.seek(0)
  state_dict = torch.load(temp)
  model_copy = c.model_partial().to(device)
  model_copy.load_state_dict(state_dict)
  
  # Set model to evaluation mode
  model_copy.eval()
  
  # Initialize variables for tracking loss and accuracy
  total_loss = 0.0
  correct = 0
  total = 0
  # Disable gradient computation for evaluation
  with torch.no_grad():
    for batch in val_dataloader:
      inputs, targets = batch[0].to(device), batch[1].to(device)
      # Forward pass
      outputs = model_copy(inputs)
      # Compute loss
      loss = loss_fn(outputs, targets)
      total_loss += loss.item()
      # Compute accuracy
      _, predicted = torch.max(outputs.data, 1)
      total += targets.size(0)
      correct += (predicted == targets).sum().item()
  
  # Calculate average loss and accuracy
  avg_loss = total_loss / len(val_dataloader)
  accuracy = correct / total
  
  if c.rank == 0:
    if c.save_method == "state_dict":
      torch.save(model_copy.state_dict(), path)
    elif c.save_method == "trace":
      example_input = torch.randn(1, *INPUT_SHAPE).to(device)
      traced_model = torch.jit.trace(model_copy, example_input)
      traced_model.save(path)
    elif c.save_method == "script":
      scripted_model = torch.jit.script(model_copy)
      scripted_model.save(path)
      
  return avg_loss, accuracy
