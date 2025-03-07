from datasets.clock import ClockConfig
from models.encoders import MLPEncoderReLU, MLPEncoderAdjustedSigmoid, MLPEncoderSigmoid, TwoHeadConvEncoder, ConvSelfAttentionEncoder, WideConvResEncoder
from train_utils.train import TrainRunConfig
from train_utils import train_clock_model, TrainRunConfig


if __name__ == "__main__":
 for cls in [ MLPEncoderSigmoid ]:
    for augment in [ True, False ]:
      for weight_decay in [1e-2]:
  
        config = TrainRunConfig(
            model_class=cls,
            type="encoder",
            latent_dim=2,
            batch_size=128,
            img_size=128,
            data_size=2**23,
            data_config=ClockConfig(
                minute_hand_len=1,
                minute_hand_start=0.5,
                miute_hand_thickness=0.1,
                hour_hand_len=0.5,
                hour_hand_start=0,
                hour_hand_thickness=0.1
            ),
            learning_rate=1e-4,
            weight_decay=weight_decay,
            augment=augment,
            n_checkpoints=4,
            notes="Bn momentum 0.5, testing augment",
            save_path_suffix=f"augment-{augment}"
        )
      
        train_clock_model(config)

 