from tasks.point.trainer import PointsTrainer, PointsTrainRunConfig
from tasks.point.trainer import PointDatasetConfig
from models.autoencoders import MLPAutoencoder
import torch.nn as nn


if __name__ == "__main__":
  
  config = PointsTrainRunConfig(
    checkpoint_dir="points",
    model_class=MLPAutoencoder,
    model_params=dict(
      encoder_dims=[2,32,32,1],
      decoder_dims=[1,32,32,2],
      decoder_activation='tanh'
    ),
    data_config=PointDatasetConfig(
      num_points=10000
    ),
    criterion=nn.MSELoss(),
  )
  
  trainer = PointsTrainer(config)
  
  trainer.train()
  
