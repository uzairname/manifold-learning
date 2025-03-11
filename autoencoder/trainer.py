import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import json
from dataclasses import asdict
import functools

import os
import psutil
import time
import wandb
import wandb.wandb_run
import logging
from tqdm import tqdm

from config import MODELS_DIR
from autoencoder.clock import TrainRunConfig, eval_and_save_model, get_dataloaders
from utils.train_utils import log_gradient_norms
from utils.utils import mkdir_empty
from utils.multiprocessing_utils import process_group_cleanup, process_group_setup


DATASET_NAME = "clock"

def train_clock_model(c: TrainRunConfig):
    torch.cuda.empty_cache()

    c.world_size = c.world_size or torch.cuda.device_count()
    c.model_partial = functools.partial(c.model_class, img_size=c.img_size, latent_dim=c.latent_dim, **(c.model_params or {}))
    c.dataset_config.img_size = c.img_size
    
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
  try:
    c.rank = rank
    process_group_setup(rank, c.world_size)
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


def _train(c: TrainRunConfig):
    torch.manual_seed(42)     
    device = torch.device(f"cuda:{c.rank}")
    torch.autograd.set_detect_anomaly(True)
    
    # Get dataloaders
    train_dataloader, val_dataloader, train_sampler, _ = get_dataloaders(
        data_config=c.data_config,
        dataset_config=c.dataset_config,
        val_size=c.val_size,
        batch_size=c.batch_size,
        world_size=c.world_size,
        rank=c.rank,
        use_workers=True
    )
    
    # Create validation. Shape (N, B, C, H, W)
    if c.rank == 0:
        val_data = [batch for batch in val_dataloader]
    else:
        val_data = None

    n_epochs = c.n_epochs
    
    # Total number of batches this gpu will see
    total_steps = len(train_dataloader) * n_epochs
    
    # Number of unique samples in the training set
    train_samples = len(train_dataloader.dataset)
    
    # Number of total samples seen by the model (train_samples * n_epochs) is 2^log_total_train_samples
    log_total_train_samples = np.round(np.log2(len(train_dataloader.dataset) * n_epochs)).astype(int)
    
    batches_per_gpu_epoch = len(train_dataloader)
    
    # Checkpoint every n optimization steps/batches
    checkpoint_every = np.max((total_steps // c.n_checkpoints, 1)) if c.n_checkpoints is not None else None
    
    # Initialize model and wrap with DistributedDataParallel
    model = c.model_partial().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])

    if c.optimizer is not None:
      optimizer = c.optimizer(model)
    else:
      optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    criterion = c.loss_fn

    scaler = torch.amp.GradScaler()

    # Checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, c.name, f"{c.latent_dim}-i{c.img_size}-d{log_total_train_samples}{c.save_path_suffix or ''}")
    if c.rank == 0:
      mkdir_empty(checkpoint_dir)
      if c.save_method == "state_dict":
        # save json of model args, img_size, latent_dim to the checkpoint dir
        with open(os.path.join(checkpoint_dir, "model_params.json"), "w") as f:
          json.dump({
            "data_config": asdict(c.data_config) if c.data_config is not None else None,
            "model_params": c.model_params,
            "img_size": c.img_size,
            "latent_dim": c.latent_dim,
        }, f)
          
          
    logging.info(f"Saving model checkpoints to {checkpoint_dir}")
    
    if c.log_wandb and c.rank == 0:
      c.run = wandb.init(
        name=c.name,
        project="manifold-learning",
        config={
            "latent-dim": c.latent_dim,
            "data-config": c.data_config,
            "dataset-config": c.dataset_config,
            "model-params": c.model_params,
            "learning-rate": c.learning_rate,
            "weight-decay": c.weight_decay,
            "img-size": c.img_size,
            "model-class": c.model_class.__name__,
            "n-epochs": c.n_epochs,
            "train-samples-per-epoch": train_samples, # Unique train samples, per epoch.
            "log-total-train-samples": log_total_train_samples, # Train samples times number of epochs
            "n-params": sum(p.numel() for p in c.model_partial().parameters()),
            "batch-size": c.batch_size,
            "type": c.type,
            "loss-fn": c.loss_fn,
            "dataset": DATASET_NAME,
        },
        notes=c.notes,
        tags=c.tags
      )
    
      wandb.define_metric("steps")
      wandb.define_metric("time")
      wandb.define_metric("val_loss", step_metric="steps")
      wandb.define_metric("train_loss", step_metric="steps")
      
    # wandb.watch(ddp_model, log="all", log_freq=16)
      
    logging.info(f"total steps {total_steps}")
    logging.info(f"Data size {c.dataset_config.data_size}, batch size {c.batch_size}, epochs {n_epochs}")
    logging.info(f"Training on 2^{log_total_train_samples} total samples, ~{batches_per_gpu_epoch} batches per epoch per GPU")
    logging.info(f"Validation on {len(val_dataloader.dataset)} samples")
    
        
    # Training loop
    val_loss_eval = None
    checkpoint_num = 0
    running_loss = 0
    start_time = time.time()
    with tqdm(total=total_steps, disable=(c.rank != 0)) as t:
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
                
            # Save model checkpoint before optimization step
            if c.rank == 0 and checkpoint_every is not None and (step % checkpoint_every == 0):
              val_loss_eval = eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, f"{checkpoint_num}.pt"), criterion, val_data)
              checkpoint_num += 1
              if c.run is not None:
                c.run.log({"val_loss": val_loss_eval, "steps": step, "time": time.time() - start_time})
                

            # Forward pass with AMP broadcast for mixed precision
            with torch.amp.autocast(device_type='cuda'):
              pred = ddp_model(input)
              loss = criterion(pred, output)

            # Average loss across all GPUs for logging
            loss_tensor = loss.clone().detach()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Log gradient norms
            if (c.rank == 0) and (step % 16 == 0):
              log_gradient_norms(c.run, ddp_model, time=time.time() - start_time)
              # log image
              # if c.run is not None:
              #   c.run.log({"image": wandb.Image(imgs[0].cpu())})
                

            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Log the loss
            running_loss += loss_tensor.item()
            if (c.rank == 0) and (step % 16 == 0):
                mem = psutil.virtual_memory()
                avg_loss = running_loss / 16
                t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={val_loss_eval:.5f}" if val_loss_eval is not None else ""))
                t.set_postfix(epoch=f"{epoch}/{n_epochs}",batch=f"{i}/{batches_per_gpu_epoch}", gpus=f"{c.world_size}", mem=f"{mem.percent:.2f}%")
                if c.run is not None:
                    c.run.log({"train_loss": avg_loss, "steps": step, "time": time.time() - start_time})
                running_loss = 0

            if c.rank == 0:
              t.update(1) 
              
    time_taken = time.time() - start_time
    

    if c.run is not None:
      c.run.summary["time_taken"] = time_taken
    # Evaluate and save final loss              
    if c.rank == 0:
        eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, "final.pt"), criterion, val_data)
        if c.run is not None:
            c.run.log({"val_loss": val_loss_eval, "steps": step})
        logging.info(f'Model checkpoints saved to {checkpoint_dir}')
      

"""
torchrun --nproc_per_node=4 train.py
"""
