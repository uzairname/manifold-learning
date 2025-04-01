from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from tasks.clock.utils import TrainConfig
from tasks.clock.utils import TrainConfig
from tasks.clock import train_clock_model
import numpy as np

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for total_samples in [2**19]:
<<<<<<< Updated upstream
      for learning_rate in [1e-4]:
        for data_size in [2**10]:
          for weight_decay in [1e-1]:

            config = TrainConfig(
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
                data_config=ClockDatasetConfig(),
                batch_size=128,
                n_epochs=total_samples//data_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                n_checkpoints=16,
                experiment_group="",
            )
        
            train_clock_model(config)
=======
      for learning_rate in [1e-2, 1e-4]:
        for noise_std, blur in [[0.001, 0], [0.05, 2], [0.01, 4]]:
          data_size = total_samples

          config = TrainConfig(
              dataset_config=ClockDatasetConfig(
                data_size=data_size,
                img_size=64,
                augment=dict(
                  noise_std=noise_std,
                  blur=blur
                )
              ),
              type="encoder",
              model_class=cls,
              model_params=dict(
                fc_dims=[256, 128],
                n_conv_blocks=2,
                channels=[1, 32, 64],
                sigmoid=True
              ),
              data_config=ClockConfig(),
              latent_dim=2,
              batch_size=256,
              n_epochs=total_samples//data_size,
              learning_rate=1e-3,
              weight_decay=1e-2,
              n_checkpoints=16,
          )
      
          train_clock_model(config)
>>>>>>> Stashed changes


