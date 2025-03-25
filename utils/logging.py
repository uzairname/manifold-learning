import logging

import numpy as np
import torch
import torch.nn as nn
from wandb.wandb_run import Run


def setup_logging(rank:int=None):
    logger = logging.getLogger()
    is_primary = rank == 0 or rank is None
    logger.setLevel(logging.INFO if is_primary else logging.WARNING)  # Only Rank 0 logs INFO

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler
    handler = logging.StreamHandler()

    # Custom log formatter that works without 'extra' dict
    formatter = logging.Formatter(f"%(asctime)s [Rank {rank}] %(message)s" if rank is not None else f"%(asctime)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_norms(model: nn.Module, run: Run=None, step=None):
  """
  Logs the weight norms and gradient norms of the model.
  """

  norms = []

  for name, param in model.named_parameters():
    if 'weight' in name:  # Focus on weight parameters
        norm = torch.norm(param, p=2).item()  # Compute L2 norm
        norms.append(norm)

    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        threshold = 1e3
        if grad_norm >= threshold:
            logging.warning(f"Gradient norm exceeded {threshold} in layer: {name} | Norm: {grad_norm}")

  mean_norm = sum(norms) / len(norms) if norms else 0

  grad_norms = []
  # get min, max, mean, std of gradient norms
  for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
  grad_norms = np.array(grad_norms)
  std_grad = np.std(grad_norms)
  mean_grad = np.mean(grad_norms)
  max_grad = np.max(grad_norms)

  if run is not None:
    run['train/grad_norms/std'].append(
      value=std_grad,
      step=step
    )
    run['train/grad_norms/mean'].append(
      value=mean_grad,
      step=step
    )
    run['train/grad_norms/max'].append(
      value=max_grad,
      step=step
    )
    run['train/weight_norms/mean'].append(
      value=mean_norm,
      step=step
    )