from abc import abstractmethod, ABC
from typing import Tuple, Generic, TypeVar, List
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

from utils.data_types import TC, TrainRunState, TrainConfig
from utils.logger import TrainRunLogger
from utils.multiprocessing import process_group_cleanup, process_group_setup
from utils.helpers import eval_model, set_all_seeds, copy_model
from utils.logging import setup_logging
from utils.utils import mkdir_empty


class Trainer(ABC, Generic[TC]):

  def __init__(self, train_config: TC):
    """
    Class for training a model. Handles distributed training, logging, and saving checkpoints.
    
    Type Parameters:
      TC: The type of the training configuration, which must inherit from `TrainConfig`.

    Args:
      train_config (TC): The object containing all hyperparameters and settings for training.

    Attributes:
      c (TC): The training configuration object.
      run (neptune.Run): Placeholder for a neptune.ai training run to log to, initialized as None.
    """

    self.c = train_config
    self.logger = TrainRunLogger(self)
  
  def train(self):
    """
    Train the model
    """
    
    c = self.c

    set_all_seeds()
    torch.cuda.empty_cache()
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
      mp.spawn(self.train_process, args=(c,), nprocs=c.world_size, join=True)
    else:
      setup_logging()
      self._train(c)
    if self.run is not None:
      self.run.stop()

    print('Done training')


  def train_process(self, rank, c: TC):
    try:
      process_group_setup(rank, c.world_size)
      self._train(c, rank)
    except Exception as e:
      logging.error(f"Error in rank {rank}:")
      logging.error(e, exc_info=True)
    finally:
      process_group_cleanup()

  @abstractmethod
  def get_data(self, c: TC, rank=None, get_val_data=True) -> Tuple[torch.utils.data.DataLoader, List, torch.utils.data.Sampler]:
    """
    Retrieves the training and validation data.

    Args:
      c (TC): The training configuration object.
      rank (int, optional): The rank of the current process in distributed training. Defaults to None.
      get_val_data (bool, optional): Whether to retrieve validation data. Defaults to True.

    Returns:
      Tuple[torch.utils.data.DataLoader, List, torch.utils.data.Sampler]: 
        - train_dataloader: The DataLoader for the training data.
        - val_data: The validation data.
        - train_sampler: The sampler for the training data.
    """
    pass
  
  
  @abstractmethod
  def get_inputs_labels(
    self,
    batch: torch.Tensor,
    s: TrainRunState,
    c: TC
  ) -> torch.Tensor:
    """
    Extracts the inputs and labels from a batch of data.

    Args:
      batch (torch.Tensor): A batch of data from the DataLoader.
      s (TrainRunState): The current state of the training run.
      c (TC): The training configuration object.

    Returns:
      torch.Tensor: A tuple containing the inputs and labels for the model.
    """
    pass
  
  
  def _train(self, c: TC, rank=None):
    """
    Individual process training function. 
    This is called by each spawned process in distributed training or directly in single GPU/CPU mode.
    """
    
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
      checkpoint_dir=os.path.join(c.checkpoint_base_dir, c.model_name, c.checkpoint_dir),
    )
    self.logger.setup_tracking(s, c)

    with tqdm(total=total_steps, disable=not is_primary, ncols=170) as t:
      s.t = t
      for epoch in range(c.n_epochs):
        s.epoch = epoch
        if c.distributed:
          train_sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            s.batch_idx = i
            s.step = i + epoch * len(train_dataloader)
            self.logger.log_metrics(s, c)

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
              self.logger.log_norms(s, self.run)

            torch.nn.utils.clip_grad_norm_(training_model.parameters(), 0.2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if is_primary:
              t.update(1) 
  
