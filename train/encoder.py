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
        for noise_std, blur in [[0.001, 0]]:
          data_size = total_samples

          config = TrainRunConfig(
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
              learning_rate=learning_rate,
              weight_decay=1e-2,
              n_checkpoints=16,
              experiment_group="test",
          )
      
          train_clock_model(config)




# /mnt/home/moham147/experiments/manifold-learning/saved_models/ResNetDecoder3
