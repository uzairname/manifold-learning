import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
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
from train_utils.multiprocessing_utils import process_group_cleanup, process_group_setup
from utils import mkdir_empty, silentremove


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
  data_config: dict = None
  augment: bool = False
  
  # hyperparameters
  batch_size: int = 64
  learning_rate: float = 1e-4
  weight_decay: float = 0.001
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

    if c.log_wandb:
      c.run = wandb.init(
        name=c.name,
        project="manifold-learning",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": c.learning_rate,
            "weight_decay": c.weight_decay,
            "img-size": c.img_size,
            "model_class": c.model_class.__name__,
            "model_kwargs": c.model_kwargs,
            "log-data-size": np.log2(c.data_size),
            "n-params": sum(p.numel() for p in c.model_class(img_size=c.img_size).parameters()),
            "batch-size": c.batch_size,
            "type": c.type,
            "augment": c.augment,
            "loss-fn": c.loss_fn,
        },
        notes=c.notes,
        tags=c.tags
      )

    mp.spawn(
      train_process,
      args=(c,),
      nprocs=c.world_size,
      join=True
    )
    
    print('Processes finished')
    if c.run is not None:
      c.run.finish()



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
  

def _train(c: TrainRunConfig):
    torch.manual_seed(42)
    
            
    if (c.latent_dim > 2): raise ValueError("Latent dim must be 1 or 2")
      
    device = torch.device(f"cuda:{c.rank}")

    # Load dataset with DistributedSampler
    dataset = ClockDataset(len=c.data_size, img_size=c.img_size, augment=c.augment, config=c.data_config)
    sampler = DistributedSampler(dataset, num_replicas=c.world_size, rank=c.rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=c.batch_size, sampler=sampler, pin_memory=True, drop_last=True, num_workers=4) # increase num_workers if GPU is underutilized

    # Initialize model and wrap with DistributedDataParallel
    model = c.model_class(img_size=c.img_size, latent_dim=c.latent_dim, **(c.model_kwargs or {})).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])

    criterion = c.loss_fn
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Save file name
    log_data_size = int(np.log2(np.max((c.data_size, 1))))
        
    # Checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, c.name, f"{c.latent_dim}-i{c.img_size}-d{log_data_size}{c.save_path_suffix or ''}")
    if c.rank == 0:
      mkdir_empty(checkpoint_dir)
    
    logging.info(f"Saving model checkpoints to {checkpoint_dir}")
    checkpoint_every = len(dataloader) // c.n_checkpoints if c.n_checkpoints is not None else None
    mkdir_empty(checkpoint_dir)

    # Training loop
    running_loss = 0
    checkpoint_num = 0
    t = tqdm(enumerate(dataloader), total=len(dataloader), disable=(c.rank != 0))
    for i, batch in t:
        imgs, clean_imgs, labels2d, labels1d = batch
        
        labels = labels1d if c.latent_dim == 1 else labels2d

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
        
        # Log loss
        running_loss += loss_tensor.item()
        if (c.rank == 0) and ((i + 1) % 16 == 0):
            avg_loss = running_loss / 16
            t.set_description_str(f"Training {c.name}... Loss={avg_loss:.5f}")
            t.set_postfix(batches=f"{i+1}/{len(dataloader)}", gpus=f"{c.world_size}")
            if c.run is not None:
                c.run.log({"loss": avg_loss})
            running_loss = 0
            
            
        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
            
        # Save model checkpoint
        if c.rank == 0 and checkpoint_every is not None and (i + 1) % checkpoint_every == 0:

            eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, f"{checkpoint_num}.pt"), criterion, dataloader, checkpoint_num=checkpoint_num)
            checkpoint_num += 1
            


    # Evaluate and save final loss              
    if c.rank == 0:
        eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, "final.pt"), criterion, dataloader)
        logging.info(f'Model checkpoints saved to {checkpoint_dir}')
      

        
    
        
def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  criterion: nn.Module,
  dataloader: DataLoader,
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
  if c.run is None or checkpoint_num is not None:
    return
  
  logging.info(f"Evaluating model")
  
  # choose 16 random batches
  model_copy.eval()
  data_iter = iter(dataloader)

  n_batches = 16
  test_loss = 0
  for i, batch in enumerate(data_iter):
      if i >= n_batches:
          break
      imgs, clean_imgs, labels2d, labels1d = batch
      labels = labels1d if c.latent_dim == 1 else labels2d

      if c.type == "encoder":
          input = imgs.to(device)
          output = labels.to(device)
      elif c.type == "decoder":
          input = labels.to(device)
          output = clean_imgs.to(device)
      elif c.type == "autoencoder":
          input = imgs.to(device)
          output = clean_imgs.to(device)

      with torch.no_grad():
          pred = model_copy(input)
          loss = criterion(pred, output)
          test_loss += loss.item()
          
  test_loss /= n_batches

  logging.info(f"Test loss: {test_loss}")
  c.run.summary["test_loss"] = test_loss


"""
torchrun --nproc_per_node=4 train_ae.py
"""
