from datasets.clock import ClockConfig, ClockDatasetConfig
from models.autoencoders import MLPResnetAutoencoder
from autoencoder.clock import TrainRunConfig
from autoencoder import train_clock_model
import torch
import torch.nn as nn
import torch.optim as optim


def optimizer(model: nn.Module):
  
  encoder_params = model.encoder.parameters()
  decoder_params = model.decoder.parameters()
  
  return torch.optim.AdamW(
    [
      {"params": encoder_params, "lr": 1e-3, "weight_decay": 1e-2},
      {"params": decoder_params, "lr": 1e-2, "weight_decay": 1e-3},
    ],
  )

if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    
    config = TrainRunConfig(
        model_class=cls,
        type="autoencoder",
        latent_dim=2,
        model_params=dict(
          encoder_args=dict(
            n_conv_blocks=2,
            channels=[1, 32, 64],
            fc_dims=[256, 128],
          ),
          decoder_args=dict(
            fc_size=256,
            resnet_start_channels=128,
          ),
        ),
        data_config=ClockConfig(),
        dataset_config=ClockDatasetConfig(
          data_size=2**24,
          img_size=64,
          augment=dict(
            noise_std=0.01,
          ),
        ),
        batch_size=256,
        optimizer=optimizer,
        loss_fn=nn.SmoothL1Loss(),
        n_checkpoints=16,
    )
  
    train_clock_model(config)
