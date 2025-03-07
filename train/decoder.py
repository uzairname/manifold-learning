from datasets.clock import ClockConfig
from models.decoders import ResNetDecoder
from train_utils import TrainRunConfig, train_clock_model
import numpy as np

if __name__ == "__main__":
  
    for cls in [ ResNetDecoder ]:
      
      config = TrainRunConfig(
        model_class=cls,
        type="decoder",
        img_size=128,
        data_size=2**20,
        batch_size=128,
        latent_dim=2,
        learning_rate=1e-4,
        weight_decay=1e-2,
        data_config=ClockConfig(
            minute_hand_len=1,
            minute_hand_start=0.5,
            miute_hand_thickness=0.1,
            hour_hand_len=0.5,
            hour_hand_start=0,
            hour_hand_thickness=0.1
        ),
        augment=True,
        save_path_suffix="a"
      )
      train_clock_model(config)
