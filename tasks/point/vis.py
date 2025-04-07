import typing
import torch

from utils.checkpoints import ModelCheckpoint



def map_inputs(cp: ModelCheckpoint, get_inputs_labels: typing.Callable, forward_fn: typing.Callable, limit=None, device='cuda'):
  """
  Given a forward function that takes an input according to the type and returns an output of batches,
  this function maps the inputs to the outputs.
  """
  count = 0
  with torch.no_grad():
    for batch in cp.dataloader:
      inputs, labels = get_inputs_labels(batch)
      
      outs = forward_fn(inputs)
        
      for i in range(outs.size(0)):
        yield inputs[i], labels[i], outs[i]

        count += 1
        if limit is not None and count >= limit:
          return



