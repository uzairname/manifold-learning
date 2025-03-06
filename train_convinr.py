
import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import numpy as np

from datasets.clock import ClockConfig
from models import ConvInrAutoencoder
from train.multiprocessing_utils import process_group_setup, process_group_cleanup
from train.train_ae import train_ae

import logging
import wandb

def train_process(rank, world_size, distributed=True):
      
    process_group_setup(rank, world_size)
    
    model_class = ConvInrAutoencoder
    lr = 1e-4
    img_size = 64
    data_size = 2**20
    data_config = ClockConfig(
        minute_hand_len=1,
        minute_hand_start=0.5,
        miute_hand_thickness=0.1,
        hour_hand_len=0.5,
        hour_hand_start=0,
        hour_hand_thickness=0.1
    )
    name=model_class.__name__
    n_params = sum(p.numel() for p in model_class(img_size=img_size).parameters())

    if rank == 0:
        run = wandb.init(
          name=name,
          project="manifold-learning",
          # Track hyperparameters and run metadata.
          config={
              "learning_rate": lr,
              "img-size": img_size,
              "model_class": model_class.__name__,
              "log-data-size": np.log2(data_size),
              "n-params": n_params,
          },
        )
    else:
      run = None

    try:
      train_ae(
          model_class=model_class,
          rank=rank,
          run=run,
          learning_rate=lr,
          img_size=img_size,
          data_size=data_size,
          latent_dim=2,
          data_config=data_config,
          save_dir=name
      )
    except Exception as e:
      logging.error(f"Error in rank {rank}:")
      logging.error(e, exc_info=True)
    finally:
      dist.barrier()
      process_group_cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("available gpus:", world_size)
    torch.cuda.empty_cache()
    
    mp.spawn(
        train_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

