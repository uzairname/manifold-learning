from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder, ResNetDecoder2, ResNetDecoder3
from train_utils import TrainRunConfig, train_clock_model
import numpy as np
import torch.nn as nn


if __name__ == "__main__":
  for cls in [ ResNetDecoder3 ]:
    for fc_size, resnet_start_channels in [[1024, 384]]:
      for total_samples in [2**22]:
          data_size=total_samples # for infinite data, 1 epoch

          for batch_size in [128]:
            config = TrainRunConfig(
              type="decoder",
              name=cls.__name__,
              model_class=cls,
              model_args=dict(
                resnet_start_channels=resnet_start_channels,
                fc_size=fc_size,
                conv_start_channels=64,
              ),
              loss_fn=nn.MSELoss(),
              img_size=128,
              data_size=data_size,
              n_epochs=total_samples//data_size,
              batch_size=batch_size,
              latent_dim=2,
              learning_rate=1e-4,
              # learning_rate=1e-3*batch_size/128,
              weight_decay=1e-4,
              data_config=ClockConfig(
                  minute_hand_len=1,
                  minute_hand_start=0,
                  miute_hand_thickness=0.05,
                  hour_hand_len=0.5,
                  hour_hand_start=0,
                  hour_hand_thickness=0.1
              ),
              n_checkpoints=16,
              augment=True,
              save_path_suffix=f"b",
            )
            train_clock_model(config)

