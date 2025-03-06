import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

import numpy as np
import os

import wandb.wandb_run

from datasets.clock import IMG_SIZE, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR
import io
import logging
from dataclasses import dataclass
import typing
from train.multiprocessing_utils import process_group_cleanup, process_group_setup

@dataclass
class TrainRunConfig:
  # model
  model_class: nn.Module
  type: typing.Literal["autoencoder", "encoder", "decoder"]
  latent_dim: int = 2
  img_size: int = IMG_SIZE
  
  rank: int = None

  run: wandb.wandb_run.Run | None = None
  name: str = None
  note: str = None
  
  # data
  data_size: int = 2**20
  data_config: dict = None
  
  # hyperparameters
  batch_size: int = 64
  learning_rate: float = 1e-4
  weight_decay: float = 0.001
  
  # checkpointing
  checkpoint_every: int = None



def train_clock_model(c: TrainRunConfig):
  
    torch.cuda.empty_cache()
    world_size = torch.cuda.device_count()
    
    if c.name is None:
      c.name = c.model_class.__name__
      
    print(f"Training model {c.name} with {world_size} GPUs")
    
    if c.rank == 0:
      c.run = wandb.init(
        name=c.name,
        project="manifold-learning",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": c.learning_rate,
            "img-size": c.img_size,
            "model_class": c.model_class.__name__,
            "log-data-size": np.log2(c.data_size),
            "n-params": sum(p.numel() for p in c.model_class(img_size=c.img_size).parameters()),
            "batch-size": c.batch_size,
            "note": c.note,
        },
      )
      
    else:
      c.run = None
  
    mp.spawn(
      train_process,
      args=(world_size, c),
      nprocs=world_size,
      join=True
    )



def train_process(rank, world_size, c: TrainRunConfig):
  
  c.rank = rank
  
  process_group_setup(rank, world_size)

  try:
    _train(c)
  except Exception as e:
    logging.error(f"Error in rank {rank}:")
    logging.error(e, exc_info=True)
  finally:
    if c.run is not None:
      c.run.finish()
      
    dist.barrier()
    process_group_cleanup()
    
  

def _train(c: TrainRunConfig):
  
    torch.manual_seed(42)
    
    if (c.latent_dim > 2): raise ValueError("Latent dim must be 1 or 2")
      
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{c.rank}")

    # Load dataset with DistributedSampler
    dataset = ClockDataset(len=c.data_size, img_size=c.img_size, supervised=True, augment=False, config=c.data_config)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=c.rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=c.batch_size, sampler=sampler, pin_memory=True, drop_last=True, num_workers=4) # increase num_workers if GPU is underutilized

    # Initialize model and wrap with DistributedDataParallel
    model = c.model_class(img_size=c.img_size, latent_dim=c.latent_dim, device=device).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop

    log_data_size = int(np.log2(np.max((c.data_size, 1))))
    path = os.path.join(MODELS_DIR, c.name, f'{c.latent_dim}-i{c.img_size}-d{log_data_size}.pt')
    os.makedirs(os.path.join(MODELS_DIR, c.name), exist_ok=True)

    print("training")
    running_loss = 0
    t = tqdm(enumerate(dataloader), total=len(dataloader), disable=(c.rank != 0))
    for i, batch in t:
        imgs, labels2d, labels1d = batch
        labels = labels1d if c.latent_dim == 1 else labels2d

        if c.type == "encoder":
            input = imgs.to(device)
            output = labels2d.to(device)
        elif c.type == "decoder":
            input = labels2d.to(device)
            output = imgs.to(device)
        elif c.type == "autoencoder":
            input = imgs.to(device)
            output = imgs.to(device)
            
        # Forward pass with AMP broadcast for mixed precision
        with torch.amp.autocast(device_type='cuda'):
          pred = ddp_model(input)
          loss = criterion(pred, output)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_tensor = torch.tensor(loss.item(), device=device)
        # Average loss across GPUs
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size 

        running_loss += loss_tensor.item()
        if (c.rank == 0) and ((i + 1) % 16 == 0):
            avg_loss = running_loss / 16
            t.set_description_str(f"Training AE... Loss={avg_loss:.5f}")
            t.set_postfix(batches=f"{i+1}/{len(dataloader)}", gpus=f"{world_size}")
            if c.run is not None:
                c.run.log({"loss": avg_loss}, step=i*c.batch_size*world_size)
            running_loss = 0


    if c.rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Unwrap from DDP and save the model using TorchScript trace
        temp = io.BytesIO()
        torch.save(ddp_model.module.state_dict(), temp)
        temp.seek(0)
        state_dict = torch.load(temp)
        model_copy = c.model_class(latent_dim=c.latent_dim, img_size=c.img_size, device=device).to(device)
        model_copy.load_state_dict(state_dict)

        # Trace and save the model
        model_copy.eval()
        scripted_model = torch.jit.trace(model_copy, torch.randn(1, 1, c.img_size, c.img_size).to(device))
        torch.jit.save(scripted_model, path)

        logging.info(f'AE model saved to {path}')
        


"""
torchrun --nproc_per_node=4 train_ae.py
"""
