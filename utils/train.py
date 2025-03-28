import typing as t
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import neptune

@dataclass
class BaseTrainRunConfig:

  # model
  model_class: nn.Module
  model_partial: t.Optional[t.Callable[[], nn.Module]] = None
  model_params: t.Optional[dict] = None

  # hyperparameters
  n_epochs: int = 1
  batch_size: int = 64
  get_optimizer: t.Optional[t.Callable] = None
  learning_rate: float = 1e-3
  weight_decay: float = 0
  criterion: nn.Module = nn.MSELoss()
  accumulation_steps: int = 1
  
  # multiprocessing
  max_gpus: int = None
  rank: int = None
  distributed: bool = True
  world_size: int = None
  
  # checkpointing
  n_evals: int = 64
  n_checkpoints: int = 0
  save_method: t.Literal["state_dict", "trace", "script"] = "state_dict"
  checkpoint_dir_name: str = "checkpoints"
  
  # logging
  run: t.Optional[neptune.Run] = None
  log: bool = True
  experiment_group: t.Optional[str] = None
  run_name: t.Optional[str] = None
  notes: t.Optional[str] = None
  tags: t.Optional[list[str]] = None

  

def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CrossEntropyHighPrecision(nn.Module):
  '''
  From https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/helpers.py#L105
  '''
  def __init__(self):
    super(CrossEntropyHighPrecision, self).__init__()

  def forward(self, logits, labels):
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss



class HookPoint(nn.Module):
    '''
    from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/23d2c64dd1f8a5ca65efaf27e15c2b2cd47dedf1/transformers.py#L113
    A helper class to get access to intermediate activations (inspired by Garcon)
    It's a dummy module that is the identity function by default
    I can wrap any intermediate activation in a HookPoint and get a convenient way to add PyTorch hooks
    '''
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []
    
    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name
    
    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output, 
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")
    
    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")
    
    def forward(self, x):
        return x