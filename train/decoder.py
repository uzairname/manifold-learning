from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder
from train_utils import TrainRunConfig, train_clock_model
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
  for cls in [ ResNetDecoder ]:
    for total_samples in [2**18]:
      for data_size in np.logspace(9, 18, 2, base=2).astype(int):
        for batch_size in [32, 128]:
          config = TrainRunConfig(
            type="decoder",
            model_class=cls,
            model_kwargs=dict(
              use_fc=False,
              conv_only_decoder=False,
              dilation_rate=1
            ),
            img_size=128,
            data_size=data_size,
            n_epochs=total_samples//data_size,
            batch_size=batch_size,
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
            n_checkpoints=16,
            save_path_suffix=f"data_size{data_size}",
          )
          train_clock_model(config)

