from typing import TYPE_CHECKING, Generic
if TYPE_CHECKING:
  from .trainer import Trainer  # Avoid circular import

import os
from dataclasses import asdict
from functools import partial

import torch
import numpy as np

import json
import logging
import psutil
import neptune
from neptune.utils import stringify_unsupported

from utils.data_types import TC, TrainRunState
from utils.helpers import eval_model
from utils.utils import mkdir_empty


class TrainRunLogger(Generic[TC]):
  
  def __init__(self, trainer: 'Trainer'):
    """
    Utility functions for the Trainer class to handle logging and saving checkpoints durig a training run
    """
    self.run: neptune.Run = None
    self.trainer = trainer


  def setup_tracking(
    self,
    s: TrainRunState,
    c: TC,
  ):
    """
    Sets up experiment tracking and the checkpoint directory.

    This method initializes the experiment tracking system (e.g., Neptune) and prepares the directory
    for saving model checkpoints. It also logs configuration details and training metadata.

    Args:
      s (TrainRunState): The current state of the training run.
      c (TrainConfig): The training configuration.
    """

    approx_unique_samples = len(s.train_dataloader.dataset) * c.batch_size

    if c.log and s.is_primary:
      # Experiment tracking
      self.run = neptune.init_run(
        project=os.getenv("NEPTUNE_PROJECT"),
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
      )

      self.run['config'] = stringify_unsupported(dict(
        n_params=sum(p.numel() for p in s.model.parameters()),
        **asdict(c)
      ))

      logging.info(f"Total steps: {s.total_steps}")
      logging.info(f"~{approx_unique_samples} samples, {c.n_epochs} epochs")

      # Checkpoints
      if s.checkpoint_dir is not None:
        mkdir_empty(s.checkpoint_dir)
        if c.save_method == "state_dict":
          with open(os.path.join(s.checkpoint_dir, "model_params.json"), "w") as f:
            json.dump({
              "data_config": asdict(c.data_config) if c.data_config is not None else None,
              "model_params": c.model_params,
            }, f, indent=4)

        logging.info(f"Saving model checkpoints to {s.checkpoint_dir}")
      else:
        logging.info("checkpoint_dir is None. No checkpoints will be saved")


  def log_metrics(
    self,
    s: TrainRunState,
    c: TC,
  ):
    """
    Logs training metrics to Neptune during the training process.

    This method logs metrics such as training loss, validation loss, and memory usage.
    It also evaluates the model and saves checkpoints at specified intervals.

    Args:
      s (TrainRunState): The current state of the training run.
      c (TrainConfig): The training configuration.
    """

    if not s.is_primary:
      # Avoid duplicate logs in distributed training
      return

    if (s.step % 16 == 0) and s.step > 0:
      mem = psutil.virtual_memory()
      avg_loss = s.running_loss / 16
      s.t.set_description_str(f"Loss={avg_loss:.5f}" + (f" Val Loss={s.val_loss:.5f}" if s.val_loss is not None else ""))
      s.t.set_postfix(epoch=f"{s.epoch}/{c.n_epochs}", batch=f"{s.batch_idx}/{len(s.train_dataloader)}", gpus=f"{self.trainer.world_size}", mem=f"{mem.percent:.2f}%")

      if self.run is not None:
        self.run['train/train_loss'].append(
          value=avg_loss,
          step=s.step
        )
      s.running_loss = 0

    # Evaluate the model
    if s.eval_frequency and (s.step % s.eval_frequency == 0):
      s.val_loss = self.eval_model(s, c)
      if self.run is not None:
        self.run["train/val_loss"].append(
          value=s.val_loss,
          step=s.step
        )

    # Save model checkpoint
    if s.step in s.checkpoint_steps:
      self.save_checkpoint(s, c)


  def log_norms(self, s: TrainRunState):
    """
    Logs the weight norms and gradient norms of the model.

    This method computes and logs the L2 norms of model weights and gradients. It also
    warns if any gradient norm exceeds a predefined threshold.

    Args:
      s (TrainRunState): The current state of the training run.
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
    # Get min, max, mean, std of gradient norms
    for name, param in s.model.named_parameters():
      if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
    grad_norms = np.array(grad_norms)

    if self.run is not None:
      self.run['train/grad_norms/std'].append(value=np.std(grad_norms), step=s.step)
      self.run['train/grad_norms/mean'].append(value=np.mean(grad_norms), step=s.step)
      self.run['train/grad_norms/max'].append(value=np.max(grad_norms), step=s.step)
      if mean_norm is not None:
        self.run['train/weight_norms/mean'].append(value=mean_norm, step=s.step)


  def eval_model(self, s: TrainRunState, c: TC) -> float:
    """
    Evaluates the model using the configured val dataset and criterion

    Args:
      s (TrainRunState): The current state of the training run.
      c (TrainConfig): The training configuration.

    Returns:
      float: The validation loss.
    """
    return eval_model(s.model, c.criterion, s.val_data, s.device, partial(self.trainer.get_inputs_labels))


  def save_checkpoint(
    self,
    s: TrainRunState,
    c: TC,
  ):
    """
    Saves a model checkpoint to the configured directory
    
    Can save model as a TorchScript or state dict based on the configuration.
    Also stores metadata about the checkpoint in a JSON file.
    
    Args:
      s (TrainRunState): The current state of the training run.
      c (TrainConfig): The training configuration.
    """

    if not s.is_primary or s.checkpoint_dir is None:
      # Avoid duplicate saves in distributed training
      return
    
    # Determine file names
    checkpoint_name = s.checkpoint_num
    if s.checkpoint_num >= c.n_checkpoints:
      checkpoint_name = 'final'
  
    model_path = os.path.join(s.checkpoint_dir, f"{checkpoint_name}.pt")
    info_path = os.path.join(s.checkpoint_dir, f"{checkpoint_name}.json")

    # Save model
    s.model.eval()
    if c.save_method == "script":
      scripted_model = torch.jit.script(s.model)
      torch.jit.save(scripted_model, model_path)
    elif c.save_method == "state_dict":
      torch.save(s.model.state_dict(), model_path)
    else:
      raise ValueError(f"Unknown save method {c.save_method}")

    # Record checkpoint info
    val_loss = self.eval_model(s, c)
    with open(info_path, "w") as f:
      json.dump({
        "step": s.step,
        "epoch": s.epoch,
        "batch": s.batch_idx,
        "val_loss": val_loss,
      }, f, indent=4)

    s.checkpoint_num += 1
