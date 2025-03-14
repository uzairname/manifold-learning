from datasets.clock import ClockConfig, ClockDatasetConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from clock.utils import TrainRunConfig
from clock.utils import TrainRunConfig
from clock import train_clock_model
import numpy as np

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for total_samples in [2**19]:
      for learning_rate in [1e-4]:
        for data_size in [2**10]:
          for weight_decay in [1e-1]:

            config = TrainRunConfig(
                dataset_config=ClockDatasetConfig(
                  data_size=data_size,
                  img_size=64,
                ),
                type="encoder",
                model_class=cls,
                model_params=dict(
                  latent_dim=1,
                  channels=[32, 32],
                  fc_dims=[512, 256, 128, 64],
                  sigmoid=True
                ),
                data_config=ClockConfig(),
                batch_size=128,
                n_epochs=total_samples//data_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_checkpoints=16,
                experiment_group="",
            )
        
            train_clock_model(config)


