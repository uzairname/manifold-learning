from datasets.clock import ClockConfig, ClockDatasetConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from autoencoder.clock import TrainRunConfig
from autoencoder.clock import TrainRunConfig
from autoencoder import train_clock_model
import numpy as np

if __name__ == "__main__":

  for cls in [ MLPEncoder]:
    for total_samples in [2**20]:
      data_size = total_samples

      for fc_size in [512]:
          config = TrainRunConfig(
              type="encoder",
              model_class=cls,
              # model_args=dict(
              #   fc_size=fc_size,
              #   sigmoid=True,
              # ),
              dataset_config=ClockDatasetConfig(
                data_size=data_size,
              ),
              data_config=ClockConfig(),
              latent_dim=2,
              batch_size=128,
              img_size=128,
              n_epochs=total_samples//data_size,
              learning_rate=1e-4,
              weight_decay=1e-3,
              n_checkpoints=16,
          )
      
          train_clock_model(config)

# /mnt/home/moham147/experiments/manifold-learning/saved_models/ResNetDecoder3
