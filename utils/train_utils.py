import logging

import numpy as np
import torch.nn as nn
from wandb.wandb_run import Run


def log_gradient_norms(run: Run | None, model: nn.Module, time=None):

  # logging.info("Logging gradient distribution")
  for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()  
        threshold = 1e3
        if grad_norm >= threshold:
            logging.warning(f"Gradient norm exceeded {threshold} in layer: {name} | Norm: {grad_norm}")

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
  min_grad = np.min(grad_norms)

  if run is not None:
    run.log({f"grad_norm_std": std_grad,
             f"grad_norm_mean": mean_grad,
             f"grad_norm_max": max_grad,
              "time": time})
