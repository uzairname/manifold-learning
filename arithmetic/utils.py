import typing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from arithmetic.trainer import TrainRunConfig


def copy_model(trained_model: nn.Module, init_model: typing.Callable[[], nn.Module], device: str) -> nn.Module:
    model_copy = init_model()
    model_copy.load_state_dict(trained_model.state_dict())
    return model_copy.to(device)


def eval_model(
  model: nn.Module,
  loss_fn: nn.Module,
  val_dataloader: DataLoader,
  device: str,
):
  """
  Evaluates a model from a training run.
  """
  # model_ = copy_model(ddp_model.module, c.model_partial, device)
  model.eval()
  
  val_loss = 0
  with torch.no_grad():
    for batch in val_dataloader:
        _, clean_imgs, labels2d, labels1d = batch
        labels = labels1d.unsqueeze(1) if model.latent_dim == 1 else labels2d

        if model.type == "encoder":
            input = clean_imgs.to(device)
            output = labels.to(device)
        elif model.type == "decoder":
            input = labels.to(device)
            output = clean_imgs.to(device)
        elif model.type == "autoencoder":
            input = clean_imgs.to(device)
            output = clean_imgs.to(device)

        pred = model(input)
        loss = loss_fn(pred, output)
        val_loss += loss.item()
  
  return val_loss


def eval_and_save_model(
  c: TrainRunConfig,
  model: nn.Module,
  device: str,
  path: str,
  val_dataloader: DataLoader,
):
  """
  Evaluates and saves a model from a training run.
  """
  # Unwrap from DDP and save the model using TorchScript trace
  # model_ = copy_model(ddp_model.module, c.model_partial, device)
  latent_dim = c.model_params['latent_dim']

  model.eval()
  # Trace and save the model
  if c.save_method == "trace":
    # dummy_input = torch.randn(1, latent_dim).to(device) if c.type == "decoder" else torch.randn(1, 1, c.dataset_config.img_size, c.dataset_config.img_size).to(device)
    # scripted_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "script":
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, path)
  elif c.save_method == "state_dict":
    torch.save(model.state_dict(), path)
  else:
    raise ValueError(f"Unknown save method {c.save_method}")

  # Evaluate the model test loss
  val_loss = eval_model(model=model, loss_fn=c.loss_fn, val_dataloader=val_dataloader, device=device)

  return val_loss
