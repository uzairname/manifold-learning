from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder
from train_utils import TrainRunConfig, train_clock_model
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
  for cls in [ ResNetDecoder ]:
      config = TrainRunConfig(
        type="decoder",
        model_class=cls,
        model_kwargs=dict(
          use_fc=False,
          conv_only_decoder=False,
          dilation_rate=1
        ),
        img_size=256,
        data_size=2**20,
        batch_size=128,
        latent_dim=2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        data_config=ClockConfig(
            minute_hand_len=1,
            minute_hand_start=0.5,
            miute_hand_thickness=0.1,
            hour_hand_len=0.5,
            hour_hand_start=0,
            hour_hand_thickness=0.1
        ),
      )
      train_clock_model(config)

