import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import numpy as np

import typing
from dataclasses import dataclass

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
  latent_dim: int = 2
  img_size: int = IMG_SIZE
  model_kwargs: dict = None
  
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
  n_epochs: int = 4
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
  data_config: ClockConfig,
  data_size: int=2**20,
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
  assert len(val_dataloader) > 0, "Validation dataloader is empty"

  return train_dataloader, val_dataloader, train_sampler, val_sampler
  
def _train(c: TrainRunConfig):
    torch.manual_seed(42)
            
    if (c.latent_dim > 2): raise ValueError("Latent dim must be 1 or 2")
      
    device = torch.device(f"cuda:{c.rank}")
    
    # Get dataloaders
    train_dataloader, val_dataloader, train_sampler, _ = get_dataloaders(
        data_config=c.data_config,
        data_size=c.data_size,
        val_size=c.val_size,
        img_size=c.img_size,
        augment=c.augment,
        batch_size=c.batch_size,
        world_size=c.world_size,
        rank=c.rank
    )
    
    # Create validation. Shape (N, B, C, H, W)
    if c.rank == 0:
        val_data = [batch for batch in val_dataloader]
    else:
        val_data = None

    n_epochs = c.n_epochs
    # Total number of batches this gpu will see
    total_steps = len(train_dataloader) * n_epochs
    train_samples = len(train_dataloader.dataset)
    log_total_train_samples = int(np.log2(len(train_dataloader.dataset) * n_epochs))
    batches_per_gpu_epoch = len(train_dataloader)
    
    logging.info(f"Data size {c.data_size}, batch size {c.batch_size}, epochs {n_epochs}")
    logging.info(f"Training on 2^{log_total_train_samples} total samples, ~{batches_per_gpu_epoch} batches per epoch per GPU")
    logging.info(f"Validation on {len(val_dataloader.dataset)} samples")

    # Initialize model and wrap with DistributedDataParallel
    model = c.model_class(img_size=c.img_size, latent_dim=c.latent_dim, **(c.model_kwargs or {})).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])

    criterion = c.loss_fn
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Save file name
        
    # Checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, c.name, f"{c.latent_dim}-i{c.img_size}-d{log_total_train_samples}{c.save_path_suffix or ''}")
    if c.rank == 0:
      mkdir_empty(checkpoint_dir)
    
    logging.info(f"Saving model checkpoints to {checkpoint_dir}")
    checkpoint_every = total_steps // c.n_checkpoints if c.n_checkpoints is not None else None
    mkdir_empty(checkpoint_dir)
    
    if c.log_wandb and c.rank == 0:
      c.run = wandb.init(
        name=c.name,
        project="manifold-learning",
        config={
            "latent-dim": c.latent_dim,
            "learning-rate": c.learning_rate,
            "weight-decay": c.weight_decay,
            "img-size": c.img_size,
            "model-class": c.model_class.__name__,
            "model-kwargs": c.model_kwargs,
            "n-epochs": c.n_epochs,
            "train-samples-per-epoch": train_samples, # Unique train samples, per epoch.
            "log-total-train-samples": log_total_train_samples, # Train samples times number of epochs
            "n-params": sum(p.numel() for p in c.model_class(latent_dim=c.latent_dim, img_size=c.img_size, **(c.model_kwargs or {})).parameters()),
            "batch-size": c.batch_size,
            "type": c.type,
            "augment": c.augment,
            "loss-fn": c.loss_fn,
        },
        notes=c.notes,
        tags=c.tags
      )
    
      wandb.define_metric("steps")
      wandb.define_metric("val_loss", step_metric="steps")
      wandb.define_metric("train_loss", step_metric="steps")
      
    
    # Training loop
    val_loss_eval = None
    checkpoint_num = 0
    running_loss = 0
    
    with tqdm(total=total_steps, disable=(c.rank != 0)) as t:
      start_time = time.time()
      for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
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
                
            # Save model checkpoint
            if c.rank == 0 and checkpoint_every is not None and (step % checkpoint_every == 0):
              val_loss_eval, val_loss_train = eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, f"{checkpoint_num}.pt"), criterion, val_data, checkpoint_num=checkpoint_num)
              checkpoint_num += 1
              
              if c.run is not None:
                c.run.log({"val_loss": val_loss_eval, "steps": step})

            if c.rank == 0:
              t.update(1) 
      time_taken = time.time() - start_time

    if c.run is not None:
      c.run.summary["time_taken"] = time_taken
    # Evaluate and save final loss              
    if c.rank == 0:
        eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, "final.pt"), criterion, val_data)
        logging.info(f'Model checkpoints saved to {checkpoint_dir}')
      

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
  checkpoint_num: int = None,
):
  # Unwrap from DDP and save the model using TorchScript trace
  temp = io.BytesIO()
  torch.save(model.module.state_dict(), temp)
  temp.seek(0)
  state_dict = torch.load(temp)
  model_copy = c.model_class(img_size=c.img_size, latent_dim=c.latent_dim, **(c.model_kwargs or {})).to(device)
  model_copy.load_state_dict(state_dict)

  # Trace and save the model
  model_copy.eval()
  dummy_input = torch.randn(1, c.latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.img_size, c.img_size).to(device)
  scripted_model = torch.jit.trace(model_copy, dummy_input)
  torch.jit.save(scripted_model, path)
  
  # Evaluate the model test loss
  val_loss_eval, val_loss_train = eval_model(c.type, model_copy, val_data, device, criterion=criterion, latent_dim=c.latent_dim)
  
  return val_loss_eval, val_loss_train

"""
torchrun --nproc_per_node=4 train_ae.py
"""
