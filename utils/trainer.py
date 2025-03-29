from abc import abstractmethod, ABC
import typing
from dataclasses import asdict
from functools import partial
import json
import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

import neptune
import psutil
from neptune.utils import stringify_unsupported
from tqdm import tqdm

from utils.config import MODELS_DIR
from utils.data_types import TrainRunState, TrainRunConfig
from utils.multiprocessing import process_group_cleanup, process_group_setup
from utils.helpers import eval_model, set_all_seeds, copy_model
from utils.logging import setup_logging
from utils.utils import mkdir_empty


class Trainer(ABC):
  '''
  Class for doing training runs
  '''
  
  def __init__(self):
    self.run: neptune.Run = None
  
  def train(self, c: TrainRunConfig):
    torch.cuda.empty_cache()
    set_all_seeds()
    torch.autograd.set_detect_anomaly(True)
    self.run = None

    # Determine the world size based on config and gpus available
    c.world_size = min(c.max_gpus or torch.cuda.device_count(), torch.cuda.device_count())
    c.distributed = c.world_size > 1
    

    # Set model partial and run name
    c.model_partial = partial(c.model_class, **(c.model_params or {}))
    c.model_name = c.model_name or c.model_class.__name__
    print(f"Training model {c.model_name} with {c.world_size} GPUs")

    if c.distributed:
      mp.spawn(
        self.train_process,
        args=(c,),
        nprocs=c.world_size,
        join=True,
      )
    else:
      setup_logging()
      self._train(c)
    
    if self.run is not None:
      self.run.stop()
    print('Done training')


  def train_process(self, rank, c: TrainRunConfig):
    try:
      process_group_setup(rank, c.world_size)
      self._train(c, rank)
    except Exception as e:
      logging.error(f"Error in rank {rank}:")
      logging.error(e, exc_info=True)
    finally:
      process_group_cleanup()

  @abstractmethod
  def get_data(self, c: TrainRunConfig, rank=None, get_val_data=True) -> typing.Tuple[torch.utils.data.DataLoader, typing.List, torch.utils.data.Sampler]:
    '''
    Get the training and validation data
    Returns a tuple of:
    - train_dataloader: the training data
    - val_data: the validation data
    - train_sampler: the sampler for the training data

    '''
    pass
  
  
  @abstractmethod
  def get_inputs_labels(
    self,
    batch: torch.Tensor,
    s: TrainRunState,
    c: TrainRunConfig,
  ) -> torch.Tensor:
    '''
    Get the input and labels for the model.
    '''
    pass
  
  
  def _train(self, c: TrainRunConfig, rank=None):
    
    # Device
    is_primary = rank == 0 or not c.distributed
    if c.distributed:
      device = torch.device(f"cuda:{rank}")
    else:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataloader, val_data, train_sampler = self.get_data(c, rank=rank, get_val_data=is_primary)
    
    # Model
    model = c.model_partial().to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if c.distributed:
      training_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
      training_model = model

    # Optimizer and criterion
    if c.get_optimizer is not None:
      optimizer = c.get_optimizer(model)
    else:
      optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)

    criterion = c.criterion
    scaler = torch.amp.GradScaler(device=device.type)

    # Training loop
    total_steps = len(train_dataloader) * c.n_epochs

    s = TrainRunState(
      model=model,
      training_model=training_model,
      train_dataloader=train_dataloader,
      val_data=val_data,
      device=device,
      is_primary=is_primary,
      total_steps=total_steps,
      eval_frequency=None if not c.n_evals else max((total_steps // c.n_evals, 1)),
      checkpoint_steps=np.linspace(0, total_steps-1, c.n_checkpoints).astype(int).tolist(),
      checkpoint_dir=os.path.join(MODELS_DIR, c.model_name, c.checkpoint_dir_name),
    )
    self.setup_tracking(s, c)

    with tqdm(total=total_steps, disable=not is_primary, ncols=170) as t:
      s.t = t
      for epoch in range(c.n_epochs):
        s.epoch = epoch
        if c.distributed:
          train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            s.batch_idx = i
            s.step = i + epoch * len(train_dataloader)
            self.log_metrics(s, c)

            # Optimization step
            x, y = self.get_inputs_labels(batch, s, c)
            x = x.to(device)
            y = y.to(device)

            # Forward pass with AMP broadcast for mixed precision
            with torch.amp.autocast(device_type=device.type):
              pred = training_model(x)
              loss = criterion(pred, y)

            loss_tensor = loss.clone().detach()
            if c.distributed:
              dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            s.running_loss += loss_tensor.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            if is_primary and (s.step % 16 == 0):
              self.log_norms(s, self.run)

            torch.nn.utils.clip_grad_norm_(training_model.parameters(), 0.2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if is_primary:
              t.update(1) 
  
  
  def setup_tracking(
    self,
    s: TrainRunState,
    c: TrainRunConfig,
  ):
    '''
    Sets up experiment tracking and the checkpoint directory
    '''
    
    approx_unique_samples = len(s.train_dataloader.dataset) * c.batch_size
    
    if c.log and s.is_primary:
      # Experiment tracking
      self.run = neptune.init_run(
        project = os.getenv("NEPTUNE_PROJECT"),
        api_token = os.getenv("NEPTUNE_API_TOKEN"),
      )
      
      config = {
        "group": c.experiment_group,
        "name": c.model_name,
        "label": c.checkpoint_dir_name,
        "data_config": asdict(c.data_config) if c.data_config is not None else None,
        "n_params": sum(p.numel() for p in s.model.parameters()),
        "model_params": c.model_params,
        "learning-rate": c.learning_rate,
        "weight-decay": c.weight_decay,
        "n-epochs": c.n_epochs,
        "batch-size": c.batch_size,
        "loss-fn": c.criterion.__class__.__name__,
        "checkpoint-dir": s.checkpoint_dir,
      }
      
      self.run['config'] = stringify_unsupported(config)
      
      logging.info(f"total steps {s.total_steps}")
      logging.info(f"~{approx_unique_samples} samples, {c.n_epochs} epochs")
  
      # Checkpoints
      mkdir_empty(s.checkpoint_dir)
      if c.save_method == "state_dict":
        with open(os.path.join(s.checkpoint_dir, "model_params.json"), "w") as f:
          json.dump({
            "data_config": asdict(c.data_config) if c.data_config is not None else None,
            "model_params": c.model_params,
          }, f, indent=4)

      logging.info(f"Saving model checkpoints to {s.checkpoint_dir}")

        
  def log_metrics(
    self, 
    s: TrainRunState,
    c: TrainRunConfig,
  ):
    '''
    Log metrics to neptune during training
    '''

    if not s.is_primary:
      return
    
    if (s.step % 16 == 0) and s.step > 0:
      mem = psutil.virtual_memory()
      avg_loss = s.running_loss / 16
      s.t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={s.val_loss:.5f}" if s.val_loss is not None else ""))
      s.t.set_postfix(epoch=f"{s.epoch}/{c.n_epochs}",batch=f"{s.batch_idx}/{len(s.train_dataloader)}", gpus=f"{c.world_size}", mem=f"{mem.percent:.2f}%")

      if self.run is not None:
          self.run['train/train_loss'].append(
            value=avg_loss,
            step=s.step
          )
      s.running_loss = 0

    # Evaluate the model
    if s.eval_frequency and (s.step % s.eval_frequency == 0):
      s.val_loss = self.eval_model(c, s)
      if self.run is not None:
        self.run["train/val_loss"].append(
          value=s.val_loss,
          step=s.step
        )

    # Save model checkpoint
    if s.step in s.checkpoint_steps:
      self.save_checkpoint(s, c)

      
  def log_norms(self, s: TrainRunState, run: neptune.Run):
    """
    Logs the weight norms and gradient norms of the model.
    """

    norms = []

    for name, param in s.model.named_parameters():
      norm = torch.norm(param, p=2).item()  # Compute L2 norm
      norms.append(norm)

      if param.grad is not None:
          grad_norm = param.grad.norm().item()
          threshold = 1e3
          if grad_norm >= threshold:
              logging.warning(f"Gradient norm exceeded {threshold} in layer: {name} | Norm: {grad_norm}")

    mean_norm = sum(norms) / len(norms) if norms else None

    grad_norms = []
    # get min, max, mean, std of gradient norms
    for name, param in s.model.named_parameters():
      if param.grad is not None:
          grad_norm = param.grad.norm().item()
          grad_norms.append(grad_norm)
    grad_norms = np.array(grad_norms)

    if run is not None:
      run['train/grad_norms/std'].append(value=np.std(grad_norms), step=s.step)
      run['train/grad_norms/mean'].append(value=np.mean(grad_norms), step=s.step)
      run['train/grad_norms/max'].append(value=np.max(grad_norms), step=s.step)
      if mean_norm is not None:
        run['train/weight_norms/mean'].append(value=mean_norm, step=s.step)


  def eval_model(self, c: TrainRunConfig, s: TrainRunState):
    return eval_model(s.model, c.criterion, s.val_data, s.device, partial(self.get_inputs_labels, s=s, c=c))

  def save_checkpoint(
    self,
    s: TrainRunState,
    c: TrainRunConfig,
  ):
    """
    Evaluates and saves a model from a training run.
    """

    # Save model
    path = os.path.join(s.checkpoint_dir, f"{s.checkpoint_num}.pt")
    s.model.eval()
    if c.save_method == "script":
      scripted_model = torch.jit.script(s.model)
      torch.jit.save(scripted_model, path)
    elif c.save_method == "state_dict":
      torch.save(s.model.state_dict(), path)
    else:
      raise ValueError(f"Unknown save method {c.save_method}")
  
    # Record checkpoint info
    val_loss = self.eval_model(c, s)
    with open(os.path.join(s.checkpoint_dir, f"{s.checkpoint_num}.json"), "w") as f:
      json.dump({
        "step": s.step,
        "epoch": s.epoch,
        "batch": s.batch_idx,
        "val_loss": val_loss,
      }, f, indent=4)

    s.checkpoint_num += 1
