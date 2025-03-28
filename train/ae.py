from tasks.clock.dataset import ClockConfig, ClockDatasetConfig
from models.autoencoders import ConvINRAutoencoder, MLPResnetAutoencoder
from tasks.clock.utils import TrainRunConfig
from tasks.clock import train_clock_model
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

def optimizer(model: nn.Module):
  
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
    for total_samples in [2**22]:
      
      # finite data
      data_size=total_samples

      config = TrainRunConfig(
          model_class=cls,
          type="autoencoder",
          model_params=dict(
            latent_dim=2,
            encoder_args=dict(
              channels=[64, 64],
              fc_dims=[32],
            ),
            decoder_args=dict(
              fc_size=64,
              resnet_start_channels=256,
            ),
          ),
          data_config=ClockConfig(
            minute_hand_start=0,
            minute_hand_width=0.1,
            hour_hand_width=0.2,
          ),
          dataset_config=ClockDatasetConfig(
            data_size=data_size,
            img_size=64,
            augment=dict(
              noise_std=0.1,
              blur=2,
            )
          ),
          n_epochs=total_samples//data_size,
          batch_size=512,
          optimizer=partial(optimizer),
          loss_fn=nn.SmoothL1Loss(),
          n_checkpoints=16,
          max_gpus=4,
      )
      train_clock_model(config)
