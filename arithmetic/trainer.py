import json
import logging
import os
import time
import psutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

import numpy as np
from dataclasses import asdict, dataclass
import functools

import neptune
from neptune.utils import stringify_unsupported
from tqdm import tqdm

from utils.config import MODELS_DIR
from utils.multiprocessing_utils import process_group_cleanup, process_group_setup
from utils.train_utils import BaseTrainRunConfig, setup_logging
from arithmetic.dataset import ArithmeticDatasetConfig, get_mod_arithmetic_cp_dataloaders
from utils.utils import mkdir_empty


@dataclass
class TrainRunConfig(BaseTrainRunConfig):
  data_config: ArithmeticDatasetConfig = ArithmeticDatasetConfig()
  val_frac=0.0


def train_arithmetic_model(c: TrainRunConfig):
  
  # Determine the world size
  c.world_size = min(c.max_gpus or torch.cuda.device_count(), torch.cuda.device_count())
  c.distributed = c.world_size > 1
  
  # Set model partial and name
  c.model_partial = functools.partial(c.model_class, **(c.model_params or {}))
  
  if c.name is None:
    c.name = c.model_class.__name__
    
  print(f"Training model {c.name} with {c.world_size} GPUs")

  torch.cuda.empty_cache()

  if c.distributed:
    mp.spawn(
      train_process,
      args=(c,),
      nprocs=c.world_size,
      join=True,
    )
  else:
    # 1 or 0 gpus
    setup_logging()
    c.rank = None
    _train(c)
  
  if c.run is not None:
    c.run.stop()
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
    process_group_cleanup()
    
    if c.run is not None:
      c.run.stop()
      
def _train(c: TrainRunConfig):
  
  is_primary = c.rank == 0 or not c.distributed
  latent_dim = c.model_params['latent_dim']   

  if c.distributed:
    device = torch.device(f"cuda:{c.rank}")
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  torch.manual_seed(42)  
  torch.autograd.set_detect_anomaly(True)

  # Get dataloaders
  train_dataloader, val_dataloader, train_sampler, _ = get_mod_arithmetic_cp_dataloaders(
      data_config=c.data_config,
      val_frac=c.val_frac,
      batch_size=c.batch_size,
      world_size=c.world_size,
      rank=c.rank,
  )
  
  total_steps = len(train_dataloader) * c.n_epochs
  approx_unique_samples = len(train_dataloader.dataset) * c.batch_size
  checkpoint_steps = np.linspace(0, total_steps, c.n_checkpoints, endpoint=False, dtype=int)
  
  eval_frequency = np.max((total_steps // c.n_eval, 1))

  # Initialize model and wrap with DistributedDataParallel
  model = c.model_partial().to(device)
  model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
  if c.distributed:
    training_model = nn.parallel.DistributedDataParallel(model, device_ids=[c.rank])
  else:
    training_model = model

  if c.optimizer is not None:
    optimizer = c.optimizer(model)
  else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    
  criterion = c.loss_fn

  scaler = torch.amp.GradScaler(device=device.type)

  # Checkpoints and logging setup
  label = c.label or ""
  checkpoint_dir = os.path.join(MODELS_DIR, c.name, (f"{c.experiment_group or ''}-{latent_dim}-i{c.data_config.img_size}-d{log_total_train_samples}-{label}"))
  if is_primary:
    mkdir_empty(checkpoint_dir)
    if c.save_method == "state_dict":
      # save json of model args, img_size, latent_dim to the checkpoint dir
      with open(os.path.join(checkpoint_dir, "model_params.json"), "w") as f:
        json.dump({
          "data_config": asdict(c.data_config) if c.data_config is not None else None,
          "model_params": c.model_params,
        }, f, indent=4)

  logging.info(f"Saving model checkpoints to {checkpoint_dir}")
  
  if c.log and is_primary:
    c.run = neptune.init_run(
      project = os.getenv("NEPTUNE_PROJECT"),
      api_token = os.getenv("NEPTUNE_API_TOKEN"),
    )
    
    c.run['config'] = stringify_unsupported({
      "group": c.experiment_group,
      "name": c.name,
      "label": c.label,
      "dataset-config": asdict(c.data_config) if c.data_config is not None else {},
      "n_params": sum(p.numel() for p in model.parameters()),
      "model_params": c.model_params,
      "learning-rate": c.learning_rate,
      "weight-decay": c.weight_decay,
      "n-epochs": c.n_epochs,
      "batch-size": c.batch_size,
      "loss-fn": c.loss_fn.__class__.__name__,
      "checkpoint-dir": checkpoint_dir,
    })
    
  logging.info(f"total steps {total_steps}")
  logging.info(f"~{approx_unique_samples} samples, {c.n_epochs} epochs")
  

  val_loss = None
  checkpoint_num = 0
  running_loss = 0
  start_time = time.time()
  
  with tqdm(total=total_steps, disable=not is_primary, ncols=170) as t:
    for epoch in range(c.n_epochs):
      if c.distributed:
        train_sampler.set_epoch(epoch)
      for i, batch in enumerate(train_dataloader):

          # Logging
          step = i + epoch * len(train_dataloader)
          # Log the loss
          if is_primary and (step % 16 == 0) and step > 0:
              mem = psutil.virtual_memory()
              avg_loss = running_loss / 16
              t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={val_loss:.5f}" if val_loss is not None else ""))
              t.set_postfix(epoch=f"{epoch}/{c.n_epochs}",batch=f"{i}/{len(train_dataloader)}", gpus=f"{c.world_size}", mem=f"{mem.percent:.2f}%")
              if c.run is not None:
                  c.run['train/train_loss'].append(
                    value=avg_loss,
                    step=time.time() - start_time
                  )
              running_loss = 0
        

          if is_primary and (step % eval_frequency == 0):   
            # Evaluate the model
            val_loss = eval_model(model=training_model.module if c.distributed else training_model, loss_fn=c.loss_fn, val_dataloader=val_dataloader, device=device)
            if c.run is not None:
              c.run["train/val_loss"].append(
                value=val_loss,
                step=time.time() - start_time
              )

            # log image
            if c.run is not None and log_images:
              c.run['images'].append(neptune.types.File.as_image(imgs[0].cpu()))

          # Save model checkpoint
          if is_primary and (step == checkpoint_steps[checkpoint_num]):
            val_loss = eval_and_save_model(c, training_model.module if c.distributed else training_model, device, os.path.join(checkpoint_dir, f"{checkpoint_num}.pt"), val_data)
            with open(os.path.join(checkpoint_dir, f"{checkpoint_num}.json"), "w") as f:
              json.dump({
                "step": step  ,
                "epoch": epoch,
                "batch": i,
                "val_loss": val_loss,
              }, f, indent=4)
            checkpoint_num += 1


          # Optimization step

          imgs, clean_imgs, labels2d, labels1d = batch

          labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d
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
          with torch.amp.autocast(device_type=device.type):
            pred = training_model(input)
            loss = criterion(pred, output)

          # Average loss across all GPUs for logging
          loss_tensor = loss.clone().detach()
          if c.distributed:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
          running_loss += loss_tensor.item()
          
          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)

          # Log gradient norms
          if is_primary and (step % 16 == 0):
            log_norms(run=c.run, model=training_model, time=time.time() - start_time)
              
          torch.nn.utils.clip_grad_norm_(training_model.parameters(), 0.2)
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad()

          if is_primary:
            t.update(1) 
                        
  time_taken = time.time() - start_time

  if c.run is not None:
    c.run['summary/time_taken'] = time_taken
  # Evaluate and save final loss              
  if is_primary:
      val_loss = eval_and_save_model(c, training_model.module if c.distributed else training_model.module, device, os.path.join(checkpoint_dir, "final.pt"), val_data)
      if c.run is not None:
        c.run['summary/val_loss'] = val_loss

      logging.info(f'Model checkpoints saved to \033[92m{checkpoint_dir}\033[0m')
  
