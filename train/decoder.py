from datasets.clock import ClockConfig, ClockDatasetConfig
from models.decoders import ResNetDecoder, ResNetDecoder2, ResNetDecoder3
from autoencoder import train_clock_model
import numpy as np
import torch.nn as nn

from autoencoder.clock import TrainRunConfig


if __name__ == "__main__":
  for cls in [ ResNetDecoder3 ]:
    for fc_size, resnet_start_channels in [[1024, 384]]:
      for total_samples in [2**22]:
          data_size=total_samples # for infinite data, 1 epoch

          for batch_size in [128]:
            config = TrainRunConfig(
              type="decoder",
              model_class=cls,
              model_params=dict(
                resnet_start_channels=resnet_start_channels,
                fc_size=fc_size,
                conv_start_channels=64,
              ),
              img_size=128,
              data_config=ClockConfig(
                hour_hand_width=0.2,
                minute_hand_width=0.2
              ),
              dataset_config=ClockDatasetConfig(
                augment=dict(
                  noise_std=0.01
                ),
                data_size=data_size,
              ),
              n_epochs=total_samples//data_size,
              batch_size=batch_size,
              latent_dim=2,
              learning_rate=1e-4,
              weight_decay=1e-4,
              loss_fn=nn.SmoothL1Loss(),
              n_checkpoints=16,
              save_path_suffix=f"b",
            )
            train_clock_model(config)

