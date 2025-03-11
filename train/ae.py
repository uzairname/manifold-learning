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
      {"params": encoder_params, "lr": 1e-2, "weight_decay": 1e-2},
      {"params": decoder_params, "lr": 1e-3, "weight_decay": 1e-3},
    ],
  )
  

if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    
    config = TrainRunConfig(
        model_class=cls,
        type="autoencoder",
        loss_fn=nn.SmoothL1Loss(),
        latent_dim=2,
        img_size=64,
        model_params=dict(
          encoder_args=dict(
            fc_dims=[512, 256],
            n_conv_blocks=2,
            channels=[1, 32, 64, 128],
          ),
          decoder_args=dict(
            resnet_start_channels=384,
            fc_size=1024,
          ),
        ),
        data_config=ClockConfig(
          hour_hand_width=0.1,
          minute_hand_width=0.75,
          minute_hand_start=1/3,
        ),
        dataset_config=ClockDatasetConfig(
          data_size=2**14,
          augment=dict(
            noise_std=0.01,
            # blur=1.0
          )
        ),
        batch_size=256,
        optimizer=optimizer,
        n_checkpoints=32,
    )
  
    train_clock_model(config)
