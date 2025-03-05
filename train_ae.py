import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

import numpy as np
import os

from models import Autoencoder, ConvInrAutoencoder, DeepAutoencoder
from datasets.clock import IMG_SIZE, ClockConfig, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR
import io
import logging

from multiprocessing_utils import process_group_cleanup, process_group_setup
import wandb


def train_ae(
    model_class: nn.Module,
    rank=None,
    run = None,
    img_size=IMG_SIZE, 
    data_size=2**20, 
    latent_dim=2,
    data_config=None,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.001,
    checkpoint_every=None,
    save_dir="models"
):
    torch.manual_seed(42)
      
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Load dataset with DistributedSampler
    dataset = ClockDataset(len=data_size, img_size=img_size, supervised=True, augment=False, config=data_config)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, drop_last=True, num_workers=4) # increase num_workers if GPU is underutilized

    # Initialize model and wrap with DistributedDataParallel
    model = model_class(img_size=img_size, latent_dim=latent_dim, device=device).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop

    log_data_size = int(np.log2(np.max((data_size, 1))))
    path = os.path.join(MODELS_DIR, save_dir, f'ae-{latent_dim}-i{img_size}-d{log_data_size}.pt')
    os.makedirs(os.path.join(MODELS_DIR, save_dir), exist_ok=True)

    print("training")
    running_loss = 0
    t = tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0))
    for i, batch in t:
        inputs = batch[0]
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP broadcast for mixed precision
        with torch.amp.autocast(device_type='cuda'):
          predicted, _ = ddp_model(inputs)
          loss = criterion(predicted, inputs)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_tensor = torch.tensor(loss.item(), device=device)
        # Average loss across GPUs
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size 

        running_loss += loss_tensor.item()
        if (rank == 0) and ((i + 1) % 16 == 0):
            avg_loss = running_loss / 16
            t.set_description_str(f"Training AE... Loss={avg_loss:.4f}")
            t.set_postfix(batches=f"{i+1}/{len(dataloader)}", gpus=f"{world_size}")
            if run is not None:
                run.log({"loss": avg_loss})
            running_loss = 0


    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Unwrap from DDP and save the model using TorchScript trace
        temp = io.BytesIO()
        torch.save(ddp_model.module.state_dict(), temp)
        temp.seek(0)
        state_dict = torch.load(temp)
        model_copy = model_class(latent_dim=latent_dim, img_size=img_size).to(device)
        model_copy.load_state_dict(state_dict)

        # Trace and save the model
        model_copy.eval()
        scripted_model = torch.jit.trace(model_copy, torch.randn(1, 1, img_size, img_size).to(device))
        torch.jit.save(scripted_model, path)

        logging.info(f'AE model saved to {path}')
        
    if run is not None:
        run.finish()
    

def train_ae_process(rank, world_size, distributed=True):
      
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
        train_ae_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


"""
torchrun --nproc_per_node=4 train_ae.py
"""
