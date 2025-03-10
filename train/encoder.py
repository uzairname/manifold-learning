from datasets.clock import ClockConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig
import numpy as np

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for total_samples in [2**22]:
      data_size = total_samples

      for fc_size in [512]:
          config = TrainRunConfig(
              model_class=cls,
              type="encoder",
              model_args=dict(
                fc_size=fc_size,
              ),
              latent_dim=2,
              batch_size=128,
              img_size=128,
              data_size=data_size,
              n_epochs=total_samples//data_size,
              data_config=ClockConfig(),
              learning_rate=1e-4,
              weight_decay=1e-3,
              n_checkpoints=16,
          )
      
          train_clock_model(config)

# /mnt/home/moham147/experiments/manifold-learning/saved_models/ResNetDecoder3
