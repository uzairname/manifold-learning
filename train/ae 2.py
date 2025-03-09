from datasets.clock import ClockConfig
from models.autoencoders import MLPResnetAutoencoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig


if __name__ == "__main__":
  
    for cls in [ MLPResnetAutoencoder ]:
      config = TrainRunConfig(
          model_class=cls,
          type="autoencoder",
          latent_dim=2,
          batch_size=128,
          img_size=128,
          data_size=2**24,
          data_config=ClockConfig(
              minute_hand_len=1,
              minute_hand_start=0.5,
              miute_hand_thickness=0.1,
              hour_hand_len=0.5,
              hour_hand_start=0,
              hour_hand_thickness=0.1
          ),
          augment=False,
          learning_rate=1e-4,
          weight_decay=1e-3,
          n_checkpoints=16,
      )
    
      train_clock_model(config)

