from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder, ResNetDecoder3
from train_utils import TrainRunConfig, train_clock_model
import numpy as np
import torch.nn as nn
import functools

if __name__ == "__main__":
  for cls in [ ResNetDecoder3 ]:
    for total_samples in [2**27]:
        data_size=total_samples # for infinite data, 1 epoch

        for batch_size in [32]:
          config = TrainRunConfig(
            type="decoder",
            name=cls.__name__,
            model_class=cls,
            model_args=dict(
              resnet_start_channels=512,
            ),
            loss_fn=nn.MSELoss(),
            img_size=128,
            data_size=data_size,
            n_epochs=total_samples//data_size,
            batch_size=batch_size,
            latent_dim=2,
            learning_rate=3e-4*batch_size/128,
            weight_decay=3e-4,
            data_config=ClockConfig(
                minute_hand_len=1,
                minute_hand_start=0.5,
                miute_hand_thickness=0.1,
                hour_hand_len=0.5,
                hour_hand_start=0,
                hour_hand_thickness=0.1
            ),
            n_checkpoints=16,
            augment=True,
            save_path_suffix=f"",
          )
          train_clock_model(config)

