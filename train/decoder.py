from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig
from models.decoders import ActivationType, ImplicitNeuralDecoder,  ResNetDecoder3, ResNetDecoder4
import numpy as np
import torch.nn as nn

from tasks.clock import train_clock_model
from tasks.clock.utils import TrainRunConfig


if __name__ == "__main__":
  for total_samples in [2**21]:
    for fc_size, resnet_start_channels in [[128, 256]]:
      data_size=total_samples # for infinite data, 1 epoch
      batch_size = 512

      config = TrainRunConfig(
        type="decoder",
        model_class=ResNetDecoder3,
        model_params=dict(
          latent_dim=2,
          fc_size=fc_size,
          resnet_start_channels=resnet_start_channels,
        ),
        data_config=ClockDatasetConfig(
          minute_hand_start=0.5,
          minute_hand_end=1,
          minute_hand_width=0.2,
          hour_hand_start=0,
          hour_hand_end=0.5,
          hour_hand_width=0.2,
        ),
        dataset_config=ClockDatasetConfig(
          data_size=data_size,
          img_size=64,
        ),
        n_epochs=total_samples//data_size,
        batch_size=batch_size,
        learning_rate=1e-3*batch_size/128,
        weight_decay=1e-4,
        loss_fn=nn.SmoothL1Loss(),
        n_checkpoints=16,
        n_eval=4,
      )
      train_clock_model(config)

