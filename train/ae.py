import typing
from tasks.clock.dataset import ClockDatasetConfig, ClockDatasetConfig, get_dataloaders
from models.autoencoders import ConvINRAutoencoder, MLPResnetAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

from utils.data_types import TrainRunConfig
from utils.trainer import Trainer


class ClockTrainer(Trainer):
  
  def __init__(
    self, 
    type: typing.Literal["autoencoder", "encoder", "decoder"] = "autoencoder", 
    val_size=None
  ):
    super().__init__()
    self.type = type
    self.val_size = val_size
  
  def get_data(self, c, rank=None, get_val_data=True):
    train_dataloader, val_dataloader, train_sampler, _= get_dataloaders(
      data_config=c.data_config,
      val_size=self.val_size,
      batch_size=c.batch_size,
      world_size=c.world_size,
      rank=rank,
      use_workers=True
    )
    
    val_data = [batch for batch in val_dataloader] if get_val_data else None  
        
    return train_dataloader, val_data, train_sampler
  
  
  def get_inputs_labels(self, batch, s, c):
    imgs, clean_imgs, labels2d, labels1d = batch
    
    latent_dim = c.model_params["latent_dim"]

    labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d
    if self.type == "encoder":
        input = imgs.to(s.device)
        label = labels.to(s.device)
    elif self.type == "decoder":
        input = labels.to(s.device)
        label = clean_imgs.to(s.device)
    elif self.type == "autoencoder":
        input = imgs.to(s.device)
        label = clean_imgs.to(s.device)
        
    return input, label


def optimizer(model: nn.Module):
  
  encoder_params = model.encoder.parameters()
  decoder_params = model.decoder.parameters()
  
  return torch.optim.AdamW(
    [
<<<<<<< Updated upstream
      {"params": encoder_params, "lr": 1e-4, "weight_decay": 1e-2},
      {"params": decoder_params, "lr": 1e-3, "weight_decay": 1e-4},
    ],
  )

=======
      {"params": encoder_params, "lr": 1e-3, "weight_decay": 1e-2},
      {"params": decoder_params, "lr": 1e-2, "weight_decay": 1e-3},
    ],
  )
>>>>>>> Stashed changes

if __name__ == "__main__":
  
  world_size = torch.cuda.device_count()

  for cls in [ MLPResnetAutoencoder ]:
    for total_samples in [2**18]:
      
      # finite data
      data_size=total_samples
      img_size = 64

      config = TrainRunConfig(
        checkpoint_dir_name="test",
        model_class=cls,
<<<<<<< Updated upstream
=======
        type="autoencoder",
        latent_dim=2,
>>>>>>> Stashed changes
        model_params=dict(
          latent_dim=2,
          img_size=img_size,
          encoder_args=dict(
<<<<<<< Updated upstream
            channels=[64, 64],
            fc_dims=[32],
          ),
          decoder_args=dict(
            fc_size=64,
            resnet_start_channels=256,
          ),
        ),
        data_config=ClockDatasetConfig(
          minute_hand_start=0,
          minute_hand_width=0.1,
          hour_hand_width=0.2,
          data_size=data_size,
          img_size=img_size,
          augment=dict(
            noise_std=0.1,
            blur=2,
          )
        ),
        n_epochs=total_samples//data_size,
        batch_size=512,
        get_optimizer=partial(optimizer),
        criterion=nn.SmoothL1Loss(),
        n_checkpoints=16,
        max_gpus=4,
      )
      
      trainer = ClockTrainer(
        type = "autoencoder",
      )
      
      trainer.train(config)
=======
            n_conv_blocks=2,
            channels=[1, 32, 64],
            fc_dims=[256, 128],
          ),
          decoder_args=dict(
            fc_size=256,
            resnet_start_channels=128,
          ),
        ),
        data_config=ClockConfig(),
        dataset_config=ClockDatasetConfig(
          data_size=2**24,
          img_size=64,
          augment=dict(
            noise_std=0.01,
          ),
        ),
        batch_size=256,
        optimizer=optimizer,
        loss_fn=nn.SmoothL1Loss(),
        n_checkpoints=16,
    )
  
    train_clock_model(config)
>>>>>>> Stashed changes
