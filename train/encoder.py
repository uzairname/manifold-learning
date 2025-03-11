from datasets.clock import ClockConfig, ClockDatasetConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from autoencoder.clock import TrainRunConfig
from autoencoder.clock import TrainRunConfig
from autoencoder import train_clock_model
import numpy as np

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for total_samples in [2**22]:
      data_size = total_samples

      config = TrainRunConfig(
          type="encoder",
          model_class=cls,
          model_params=dict(
            fc_dims=[1024, 512, 256],
            sigmoid=True,
            n_conv_blocks=3,
            channels=[1, 32, 64, 128],
          ),
          dataset_config=ClockDatasetConfig(
            data_size=data_size,
          ),
          data_config=ClockConfig(),
          latent_dim=2,
          batch_size=256,
          img_size=256,
          n_epochs=total_samples//data_size,
          learning_rate=1e-2,
          weight_decay=1e-2,
          n_checkpoints=16,
      )
  
      train_clock_model(config)




# /mnt/home/moham147/experiments/manifold-learning/saved_models/ResNetDecoder3
