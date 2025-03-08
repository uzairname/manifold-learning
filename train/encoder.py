from datasets.clock import ClockConfig
from models.encoders import ConvMLPEncoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig
import wandb

if __name__ == "__main__":

  for cls in [ ConvMLPEncoder ]:
    for dropout_position in [0]:
      for n_epochs in [4]:
        config = TrainRunConfig(
            model_class=cls,
            name="ConvMLPEncoder",
            type="encoder",
            model_kwargs=dict(
                dropout_position=dropout_position,
            ),
            latent_dim=2,
            batch_size=512,
            img_size=128,
            data_size=2**19,
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
