from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig
from models.autoencoders import MLPResnetAutoencoder
from tasks.clock.utils import TrainRunConfig
from tasks.clock import train_clock_model
import torch
import torch.nn as nn
from functools import partial
import numpy as np


def quantization_scheduler(idx: int, alpha=17, beta=2):
  """
  scheduler for quantization
  """
  return min(int(2**(beta+idx*2**(-1*alpha))), 256)


def hand_width_scheduler(x, slope=1e5):
  """
  Returns a value for hand width, ranging from ~0.2 to ~0.05
  """
  mean = -3 * torch.tensor(x/slope).sigmoid()
  return np.random.lognormal(mean, 0.05)


if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    accumulation_steps = 1
    alpha=16
    
    config = TrainRunConfig(
        model_class=cls,
        name=f"MLPResnetAutoencoder-handwidth",
        type="autoencoder",
        latent_dim=2,
        img_size=128,
        model_params=dict(
          encoder_args=dict(
            fc_size=256,
          ),
          decoder_args=dict(
            resnet_start_channels=128,
            fc_size=512,
          ),
        ),
        data_config=ClockDatasetConfig(
          hour_hand_width=0.2,
          minute_hand_width=0.2,
        ),
        dataset_config=ClockDatasetConfig(
          augment=dict(
            noise_std=0.2
          ),
          data_size=2**22,
          # quantization_scheduler=partial(quantization_scheduler, alpha=alpha),
          hand_width_scheduler=partial(hand_width_scheduler, slope=1e5),
        ),
        batch_size=128,
        learning_rate=world_size*1e-3*accumulation_steps,
        accumulation_steps=accumulation_steps,
        weight_decay=1e-4,
        n_checkpoints=16,
        loss_fn=nn.SmoothL1Loss(),
    )
  
    train_clock_model(config)
