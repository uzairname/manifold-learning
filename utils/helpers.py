from typing import Optional, List, Tuple, Any, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random



def eval_model(
  model: nn.Module,
  criterion: nn.Module,
  val_data: List,
  device: str,
  get_inputs_labels: Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]
):
  """
  Evaluates a model from a training run.
  """
  model.eval()
  val_loss = 0
  with torch.no_grad():
    for batch in val_data:
        x, y = get_inputs_labels(batch)
        x = x.to(device)
        y = y.to(device)
        pred = model(x) # take the last token prediction
        loss = criterion(pred, y)
        val_loss += loss.item()
  return val_loss / len(val_data)



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


def copy_model(trained_model: nn.Module, init_model: Callable[[], nn.Module], device: str) -> nn.Module:
    model_copy = init_model()
    model_copy.load_state_dict(trained_model.state_dict())
    return model_copy.to(device)

