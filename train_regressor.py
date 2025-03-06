from datasets.clock import ClockConfig
from models.encoders import MultiHeadRegressor
from train.train import TrainRunConfig
from train import train_clock_model, TrainRunConfig


if __name__ == "__main__":
    
    config = TrainRunConfig(
        model_class=MultiHeadRegressor,
        type="encoder",
        latent_dim=2,
        img_size=64,
        data_size=2**22,
        data_config=ClockConfig(
            minute_hand_len=1,
            minute_hand_start=0.5,
            miute_hand_thickness=0.1,
            hour_hand_len=0.5,
            hour_hand_start=0,
            hour_hand_thickness=0.1
        ),
        batch_size=512,
        learning_rate=1e-4,
        note="With sigmoid"
    )
    
    train_clock_model(config)

