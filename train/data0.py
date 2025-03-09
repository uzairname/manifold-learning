from models.decoders import ResNetDecoder3
from train_utils.train import get_dataloaders
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import numpy as np

import typing
from dataclasses import dataclass
import functools

import os
import io
import time
import wandb
import wandb.wandb_run
import logging
from tqdm import tqdm

from config import MODELS_DIR
from utils import mkdir_empty
from train_utils.multiprocessing_utils import process_group_cleanup, process_group_setup
from datasets.clock import IMG_SIZE, ClockConfig, ClockDataset

@dataclass
class TrainRunConfig:
  # model
  model_class: nn.Module
  type: typing.Literal["autoencoder", "encoder", "decoder"]
  model_partial: typing.Callable = None
  latent_dim: int = 2
  img_size: int = IMG_SIZE
  model_args: dict = None
  
  # multiprocessing
  rank: int = None
  world_size: int = None

  # logging
  log_wandb: bool = True
  run: wandb.wandb_run.Run | None = None
  name: str = None
  notes: str = None
  tags: list[str] = None
  
  # data
  data_size: int = 2**20
  val_size: int = None
  data_config: dict = None
  augment: bool = False
  
  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  learning_rate: float = 1e-4
  weight_decay: float = 1e-4
  loss_fn: nn.Module = nn.MSELoss()
  
  # checkpointing
  n_checkpoints: int = None
  save_path_suffix: str = None

def train_clock_model(c: TrainRunConfig):
    torch.cuda.empty_cache()

    c.world_size = torch.cuda.device_count()
    c.model_partial = functools.partial(c.model_class, img_size=c.img_size, latent_dim=c.latent_dim, **(c.model_args or {}))
    
    if c.name is None:
      c.name = c.model_class.__name__
      
    print(f"Training model {c.name} with {c.world_size} GPUs")

    mp.spawn(
      train_process,
      args=(c,),
      nprocs=c.world_size,
      join=True
    )
    
    print('Done training')


def train_process(rank, c: TrainRunConfig):
  c.rank = rank
  
  process_group_setup(rank, c.world_size)

  try:
    _train(c)
  except Exception as e:
    logging.error(f"Error in rank {rank}:")
    logging.error(e, exc_info=True)
  finally:
    dist.barrier(device_ids=[rank])
    process_group_cleanup()
    print(f"Cleaned up {rank}")
    
    if c.run is not None:
      c.run.finish()


def get_dataloaders(
  data_config: ClockConfig=ClockConfig(),
  data_size: int=2**14,
  val_size: int=None,
  img_size: int=128,
  augment: bool=False,
  batch_size: int=64,
  world_size: int=1,
  rank: int=None,
):
  """
  Get the clock dataset split into training and validation sets.
  """
  # Dataset
  dataset = ClockDataset(device='cpu', len=data_size, img_size=img_size, augment=augment, config=data_config)

  # Split into train and val
  if val_size is None:
      val_size = np.min((data_size//8, 2**12))
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

  # Get sampler and dataloader for train data
  if rank is not None:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
  else:
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=6, persistent_workers=True) 
  # increase num_workers if GPU is underutilized

  # Get sampler and dataloader for val data
  val_sampler = torch.utils.data.SequentialSampler(val_dataset)  # No need for shuffling
  val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True, drop_last=True, num_workers=6, persistent_workers=True)
  
  assert len(train_dataloader) > 0, f"Train dataloader is empty (batch_size={batch_size}, dataset size={len(train_dataset)}, world_size={world_size})"
  assert val_size == 0 or len(val_dataloader) > 0, "Validation dataloader is empty"

  return train_dataloader, val_dataloader, train_sampler, val_sampler


def _train(c: TrainRunConfig):
    torch.manual_seed(42)     
    device = torch.device(f"cuda:{c.rank}")
    
    # Get dataloaders
    # Dataset
    dataset = ClockDataset(device='cpu', len=c.data_size, img_size=c.img_size, augment=c.augment, config=c.data_config)

    # Get sampler and dataloader for train data
    train_sampler = DistributedSampler(dataset, num_replicas=c.world_size, rank=c.rank, shuffle=True)
    train_dataloader = DataLoader(dataset, batch_size=c.batch_size, sampler=train_sampler, pin_memory=True, drop_last=True, num_workers=6, persistent_workers=True) 
    

    n_epochs = c.n_epochs
    # Total number of batches this gpu will see
    total_steps = len(train_dataloader) * n_epochs
    train_samples = len(train_dataloader.dataset)
    log_total_train_samples = np.round(np.log2(len(train_dataloader.dataset) * n_epochs)).astype(int)
    batches_per_gpu_epoch = len(train_dataloader)
    
    logging.info(f"total steps {total_steps}")
    logging.info(f"Data size {c.data_size}, batch size {c.batch_size}, epochs {n_epochs}")
    logging.info(f"Training on 2^{log_total_train_samples} total samples, ~{batches_per_gpu_epoch} batches per epoch per GPU")

    # Initialize model and wrap with DistributedDataParallel
    model = c.model_partial().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])

    criterion = c.loss_fn
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop
    val_loss_eval = None
    running_loss = 0
    
    with tqdm(total=total_steps, disable=(c.rank != 0)) as t:
      for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        logging.info("d")
        for i, batch in enumerate(train_dataloader):
            step = i + epoch * len(train_dataloader)
          
            imgs, clean_imgs, labels2d, labels1d = batch
            labels = labels1d.unsqueeze(1) if c.latent_dim == 1 else labels2d
            if c.type == "encoder":
                input = imgs.to(device)
                output = labels.to(device)
            elif c.type == "decoder":
                input = labels.to(device)
                output = clean_imgs.to(device)
            elif c.type == "autoencoder":
                input = imgs.to(device)
                output = clean_imgs.to(device)
                
            # Save model checkpoint before optimization step
            # Forward pass with AMP broadcast for mixed precision
            with torch.amp.autocast(device_type='cuda'):
              pred = ddp_model(input)
              loss = criterion(pred, output)
              
            # Log the loss before scaling and backward pass 
            loss_tensor = torch.tensor(loss.item(), device=device)
            # Average loss across GPUs
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            
            # Log loss every 16 steps
            running_loss += loss_tensor.item()
            if (c.rank == 0) and (step % 16 == 0):
                avg_loss = running_loss / 16
                t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={val_loss_eval:.5f}" if val_loss_eval is not None else ""))
                t.set_postfix(epoch=f"{epoch}/{n_epochs}",batch=f"{i}/{batches_per_gpu_epoch}", gpus=f"{c.world_size}")
                if c.run is not None:
                    c.run.log({"train_loss": avg_loss, "steps": step})
                running_loss = 0
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if c.rank == 0:
              t.update(1) 

    # Evaluate and save final loss              

def eval_model(
  type_: typing.Literal['encoder', 'autoencoder', 'decoder'],
  model: nn.Module,
  val_data: typing.List,
  device: str,
  criterion: nn.Module=nn.MSELoss(),
  latent_dim: int=2,
):
  val_loss_eval = 0
  model.eval()
  for i, batch in enumerate(val_data):
      imgs, clean_imgs, labels2d, labels1d = batch
      labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d

      if type_ == "encoder":
          input = imgs.to(device)
          output = labels.to(device)
      elif type_ == "decoder":
          input = labels.to(device)
          output = clean_imgs.to(device)
      elif type_ == "autoencoder":
          input = imgs.to(device)
          output = clean_imgs.to(device)

      with torch.no_grad():
          pred = model(input)
          loss = criterion(pred, output)
          val_loss_eval += loss.item()
  val_loss_eval /= len(val_data)

  val_loss_train = 0
  model.train()
  for i, batch in enumerate(val_data):
    imgs, clean_imgs, labels2d, labels1d = batch
    labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d

    if type_ == "encoder":
        input = imgs.to(device)
        output = labels.to(device)
    elif type_ == "decoder":
        input = labels.to(device)
        output = clean_imgs.to(device)
    elif type_ == "autoencoder":
        input = imgs.to(device)
        output = clean_imgs.to(device)

    with torch.no_grad():
        pred = model(input)
        loss = criterion(pred, output)
        val_loss_train += loss.item()

  val_loss_train /= len(val_data)
  
  return val_loss_eval, val_loss_train
        
        
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
  # dummy_input = torch.randn(1, c.latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.img_size, c.img_size).to(device)
  scripted_model = torch.jit.script(model_copy)
  torch.jit.save(scripted_model, path)
  
  # Evaluate the model test loss
  val_loss_eval, val_loss_train = eval_model(c.type, model_copy, val_data, device, criterion=criterion, latent_dim=c.latent_dim)
  
  return val_loss_eval, val_loss_train

"""
torchrun --nproc_per_node=4 train_ae.py
"""


if __name__ == "__main__":
  
  for cls in [ ResNetDecoder3 ]:
    for total_samples in [2**27]:
        data_size=total_samples # for infinite data, 1 epoch

        for batch_size in [32]:
          config = TrainRunConfig(
            type="decoder",
            name=cls.__name__,
            model_class=cls,
            model_args=dict(
              resnet_start_channels=512,
            ),
            loss_fn=nn.MSELoss(),
            img_size=128,
            data_size=data_size,
            n_epochs=total_samples//data_size,
            batch_size=batch_size,
            latent_dim=2,
            learning_rate=3e-4*batch_size/128,
            weight_decay=3e-4,
            data_config=ClockConfig(
                minute_hand_len=1,
                minute_hand_start=0.5,
                miute_hand_thickness=0.1,
                hour_hand_len=0.5,
                hour_hand_start=0,
                hour_hand_thickness=0.1
            ),
            n_checkpoints=16,
            augment=True,
            save_path_suffix=f"",
          )
          train_clock_model(config)
