import typing
from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig, get_dataloaders
from models.autoencoders import ConvINRAutoencoder, MLPResnetAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

from utils.data_types import TrainConfig
from utils.trainer import Trainer



def get_optimizer(model: nn.Module):
  
  encoder_params = model.encoder.parameters()
  decoder_params = model.decoder.parameters()
  
  return torch.optim.AdamW(
    [
      {"params": encoder_params, "lr": 1e-4, "weight_decay": 1e-2},
      {"params": decoder_params, "lr": 1e-3, "weight_decay": 1e-4},
    ],
  )


if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    for total_samples in [2**18]:
      
      # finite data
      data_size=total_samples
      img_size = 64

      config = TrainConfig(
        checkpoint_dir="test",
        model_class=cls,
        model_params=dict(
          latent_dim=2,
          img_size=img_size,
          encoder_args=dict(
            channels=[64, 64],
            fc_dims=[32],
          ),
          decoder_args=dict(
            fc_size=64,
            resnet_start_channels=256,
          ),
        ),
        data_config=ClockDatasetConfig(
          minute_hand_start=0,
          minute_hand_width=0.1,
          hour_hand_width=0.2,
          data_size=data_size,
          img_size=img_size,
          augment=dict(
            noise_std=0.1,
            blur=2,
          )
        ),
        n_epochs=total_samples//data_size,
        batch_size=512,
        get_optimizer=partial(get_optimizer),
        criterion=nn.SmoothL1Loss(),
        n_checkpoints=16,
        max_gpus=4,
      )
      
      trainer = ClockTrainer(
        type = "autoencoder",
      )
      
      trainer.train(config)
