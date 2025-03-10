from datasets.clock import ClockConfig
from models.autoencoders import MLPResnetAutoencoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig


if __name__ == "__main__":
    for cls in [ MLPResnetAutoencoder ]:
      config = TrainRunConfig(
          model_class=cls,
          type="autoencoder",
          model_args=dict(
            encoder_args=dict(
                resnet_start_channels=384,
                fc_size=1024,
                conv_start_channels=64,
                activation='sigmoid',
              ),
          ),
          latent_dim=2,
          batch_size=128,
          img_size=128,
          data_size=2**26,
          data_config=ClockConfig(),
          augment=True,
          learning_rate=1e-4,
          weight_decay=1e-3,
          n_checkpoints=16,
      )
    
      train_clock_model(config)

