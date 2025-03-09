from datasets.clock import ClockConfig
from models.encoders import ConvMLPEncoder, MLPEncoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig
import numpy as np

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for total_samples in [2**19]:
      data_size = total_samples
    
      config = TrainRunConfig(
          model_class=cls,
          type="encoder",
          model_kwargs=dict(
          ),
          latent_dim=1,
          batch_size=128,
          img_size=128,
          data_size=data_size,
          val_size=np.min((data_size//8, 2**12)),
          n_epochs=total_samples//data_size,
          data_config=ClockConfig(
              minute_hand_len=1,
              minute_hand_start=0.5,
              miute_hand_thickness=0.1,
              hour_hand_len=0.5,
              hour_hand_start=0,
              hour_hand_thickness=0.1
          ),
          learning_rate=1e-4,
          weight_decay=1e-3,
          augment=False,
          n_checkpoints=16,
      )
    
      train_clock_model(config)

# /mnt/home/moham147/experiments/manifold-learning/saved_models/ResNetDecoder3