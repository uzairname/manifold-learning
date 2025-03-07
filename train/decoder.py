from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder
from train_utils import TrainRunConfig, train_clock_model
import numpy as np
import torch.nn as nn

if __name__ == "__main__":
  
    for cls in [ ResNetDecoder ]:
      for loss_fn in [ nn.MSELoss() ]:
        for use_fc in [False]:
          for conv_only_decoder in [True, False]:
            for weight_decay in [1e-4, 1e-3]:
      
              config = TrainRunConfig(
                type="decoder",
                model_class=cls,
                model_kwargs=dict(
                  use_fc=use_fc,
                  conv_only_decoder=conv_only_decoder
                ),
                loss_fn=loss_fn,
                img_size=128,
                data_size=2**21,
                batch_size=128,
                latent_dim=2,
                learning_rate=1e-4,
                weight_decay=weight_decay,
                data_config=ClockConfig(
                    minute_hand_len=1,
                    minute_hand_start=0.5,
                    miute_hand_thickness=0.1,
                    hour_hand_len=0.5,
                    hour_hand_start=0,
                    hour_hand_thickness=0.1
                ),
                save_path_suffix=f"_{'fc' if use_fc else 'nofc'}_{'conv' if conv_only_decoder else 'convt'}",
              )
              train_clock_model(config)

