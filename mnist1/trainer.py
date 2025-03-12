
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import json
import functools

import os
import psutil
import time
import wandb
import wandb.wandb_run
import logging
from tqdm import tqdm

from mnist1.mnist import TrainRunConfig, eval_and_save_model, get_mnist_dataloaders
from utils.logging import setup_logging
from utils.train_utils import log_gradient_norms
from utils.utils import mkdir_empty
from utils.multiprocessing_utils import process_group_cleanup, process_group_setup

MODELS_DIR="saved_models/mnist"
DATASET_NAME="mnist"

def train_mnist_model(c: TrainRunConfig):
  
  
    c.world_size = min(c.max_gpus or torch.cuda.device_count(), torch.cuda.device_count())
    c.distributed = c.world_size > 1
    c.model_partial = functools.partial(c.model_class, **(c.model_params or {}))
    
    if c.name is None:
      c.name = c.model_class.__name__
    
    print(f"Training MNIST model {c.name} with {c.world_size} GPUs")

    torch.cuda.empty_cache()
    if c.distributed:
      mp.spawn(
        train_process,
        args=(c,),
        nprocs=c.world_size,
        join=True,
      )
    else:
      setup_logging()
      c.rank = None
      _train(c)
    
    if c.run is not None:
      c.run.finish()
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



def _train(c: TrainRunConfig):
    torch.manual_seed(42) 
  
    if c.distributed:
      device = torch.device(f"cuda:{c.rank}")
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
      
    is_primary = c.rank == 0 or not c.distributed
    
    torch.autograd.set_detect_anomaly(True)
    
    # Get dataloaders
    train_dataloader, val_dataloader, train_sampler, _ = get_mnist_dataloaders(
        batch_size=c.batch_size,
        world_size=c.world_size,
        rank=c.rank,
    )

    n_epochs = c.n_epochs
    # Total number of batches this gpu will see
    total_steps = len(train_dataloader) * n_epochs
    train_samples = len(train_dataloader.dataset)
    log_total_train_samples = np.round(np.log2(len(train_dataloader.dataset) * n_epochs)).astype(int)
    batches_per_gpu_epoch = len(train_dataloader)
    
    logging.info(f"total steps {total_steps}")
    logging.info(f"Training on 2^{log_total_train_samples} total samples, ~{batches_per_gpu_epoch} batches per epoch per GPU")
    logging.info(f"Validation on {len(val_dataloader.dataset)} samples")

    # Initialize model and wrap with DistributedDataParallel
    model = c.model_partial().to(device)
    if c.distributed:
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])
    else:
      ddp_model = model

    criterion = c.loss_fn
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler(device.type)

    # Save file name
    # Checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, c.name, f"model{c.save_path_suffix or ''}")
    if is_primary:
      mkdir_empty(checkpoint_dir)
      if c.save_method == "state_dict":
        # save json of model args, img_size, latent_dim to the checkpoint dir
        with open(os.path.join(checkpoint_dir, "model_params.json"), "w") as f:
          json.dump({
            "model_params": c.model_params,
        }, f)
    
    logging.info(f"Saving model checkpoints to {checkpoint_dir}")
    checkpoint_every = np.max((total_steps // c.n_checkpoints, 1)) if c.n_checkpoints is not None else None
    
    if c.log_wandb and is_primary:
      c.run = wandb.init(
        name=c.name,
        project="manifold-learning",
        config={
            "learning-rate": c.learning_rate,
            "weight-decay": c.weight_decay,
            "model-class": c.model_class.__name__,
            "model-kwargs": c.model_params,
            "n-epochs": c.n_epochs,
            "train-samples-per-epoch": train_samples, # Unique train samples, per epoch.
            "log-total-train-samples": log_total_train_samples, # Train samples times number of epochs
            "n-params": sum(p.numel() for p in c.model_partial().parameters()),
            "batch-size": c.batch_size,
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
    
      wandb.watch(ddp_model, log="all", log_freq=16)
    
    # Training loop
    val_loss = None
    checkpoint_num = 0
    running_loss = 0
    
    with tqdm(total=total_steps, disable=not is_primary) as t:
      start_time = time.time()
      for epoch in range(n_epochs):
        if c.distributed:
          train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
          step = i + epoch * len(train_dataloader)
          
          # Get the batch
          imgs, labels = batch
          imgs = imgs.to(device)
          labels = labels.to(device)
          
          # randomize some labels in the batch
          randomize_prob = 0.1
          for j in range(len(labels)):
            if np.random.rand() < randomize_prob:
              labels[j] = torch.randint(0, 10, (1,)).item()
          

          # Save model checkpoint before optimization step
          if is_primary and checkpoint_every is not None and (step % checkpoint_every == 0):
            val_loss, acc = eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, f"{checkpoint_num}.pt"), criterion, val_dataloader)
            val_loss = None
            checkpoint_num += 1
            if c.run is not None:
              c.run.log({"val_loss": val_loss, "steps": step, "time": time.time() - start_time})

          print(f"evaluated model. {c.rank}")

          use_mp = True
          
          if use_mp:
            # Forward pass with AMP broadcast for mixed precision
            with torch.amp.autocast(device_type=device.type):
              pred = ddp_model(imgs)
              loss = criterion(pred, labels)
          else:
            pred = ddp_model(imgs)
            loss = criterion(pred, labels)
            

          # Average loss across all GPUs for logging
          loss_tensor = loss.clone().detach()
          if c.distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
          

          if use_mp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
          else:
            loss.backward()

          if is_primary and (step % 16 == 0):
            # log gradients
            log_gradient_norms(c.run, ddp_model, time=time.time() - start_time)
            # log input image
            
          if use_mp:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
          else:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
            optimizer.step()
            optimizer.zero_grad()

          # Log the loss and memory usage
          running_loss += loss_tensor.item()
          if is_primary and (step % 1 == 0):
              mem = psutil.virtual_memory()
              avg_loss = running_loss / 16
              t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={val_loss:.5f}" if val_loss is not None else ""))
              t.set_postfix(epoch=f"{epoch}/{n_epochs}",batch=f"{i}/{batches_per_gpu_epoch}", gpus=f"{c.max_gpus}", mem=f"{mem.percent:.2f}%")
              if c.run is not None:
                  c.run.log({"train_loss": avg_loss, "steps": step, "time": time.time() - start_time})
              running_loss = 0

          if is_primary:
            t.update(1)
            

      time_taken = time.time() - start_time
    

    if c.run is not None:
      c.run.summary["time_taken"] = time_taken
    # Evaluate and save final loss              
    if c.rank == 0:
        eval_and_save_model(c, ddp_model, device, os.path.join(checkpoint_dir, "final.pt"), criterion, val_dataloader)
        if c.run is not None:
            c.run.log({"val_loss": val_loss, "steps": step})
        logging.info(f'Model checkpoints saved to {checkpoint_dir}')
      

"""
torchrun --nproc_per_node=4 train.py
"""
