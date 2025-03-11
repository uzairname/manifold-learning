from datasets.clock import ClockConfig
from models.autoencoders import MLPResnetAutoencoder
from autoencoder.clock import TrainRunConfig
from autoencoder import train_clock_model
import torch
import torch.nn as nn


if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    accumulation_steps = 1
    config = TrainRunConfig(
        model_class=cls,
        type="autoencoder",
        model_params=dict(
          encoder_args=dict(
            fc_size=512,
          ),
          decoder_args=dict(
            resnet_start_channels=384,
            fc_size=1024,
            conv_start_channels=64,
          ),
        ),
        data_config=ClockConfig(
          hour_hand_width=0.2,
          minute_hand_width=0.2,
          angle_quantization=1,
        ),
        latent_dim=2,
        batch_size=128,
        img_size=256,
        data_size=2**25,
        augment=dict(
          noise_std=0.05
        ),
        learning_rate=world_size*2e-5*accumulation_steps,
        accumulation_steps=accumulation_steps,
        weight_decay=1e-2,
        n_checkpoints=16,
        loss_fn=nn.SmoothL1Loss(),
    )
  
    train_clock_model(config)
