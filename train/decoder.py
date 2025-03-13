from datasets.clock import ClockConfig, ClockDatasetConfig
from models.decoders import ActivationType, ImplicitNeuralDecoder, ResNetDecoder, ResNetDecoder2, ResNetDecoder3
from clock import train_clock_model
import numpy as np
import torch.nn as nn

from clock.utils import TrainRunConfig


if __name__ == "__main__":
  for cls in [ ImplicitNeuralDecoder ]:
    for fc_size, resnet_start_channels in [[1024, 384]]:
      for fc_dims in [[256, 512, 1024]]:
        for use_film in [True, False]:
          for activation in [ActivationType.sine]:
            for total_samples in [2**19]:
              # for dropout in [True, False]:
                data_size=total_samples # for infinite data, 1 epoch
                batch_size = 256

                config = TrainRunConfig(
                  experiment_group="Film",
                  name="ImplicitNeuralDecoder",
                  type="decoder",
                  model_class=cls,
                  model_params=dict(
                    fc_dims=fc_dims,
                    pe_dim=8,
                    activation=activation,
                    use_film=True,
                  ),
                  # dict(
                  #     resnet_start_channels=resnet_start_channels,
                  #     fc_size=fc_size,
                  #     dropout=dropout,
                  #   )
                  data_config=ClockConfig(),
                  dataset_config=ClockDatasetConfig(
                    data_size=data_size,
                    img_size=64,
                  ),
                  n_epochs=total_samples//data_size,
                  batch_size=batch_size,
                  latent_dim=2,
                  learning_rate=1e-3*batch_size/128,
                  weight_decay=1e-4,
                  loss_fn=nn.SmoothL1Loss(),
                  n_checkpoints=16,
                )
                train_clock_model(config)

