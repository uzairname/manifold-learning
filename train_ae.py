import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

import numpy as np
import os

from models import Autoencoder
from datasets.clock import IMG_SIZE, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR
import io
import logging

from multiprocessing_utils import process_group_setup


def train_ae(
    rank,
    img_size=IMG_SIZE, 
    data_size=2**20, 
    hidden_units=2,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.001,
    checkpoint_every=None,
    tag="models"
):
    torch.manual_seed(42)
  
    # rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Load dataset with DistributedSampler
    dataset = ClockDataset(len=data_size, img_size=img_size, supervised=True, augment=False)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)

    # Initialize model and wrap with DistributedDataParallel
    model = Autoencoder(hidden_units=hidden_units, input_dim=img_size).to(device)

    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop
    # t = tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0))
    running_loss = 0
    for i, batch in enumerate(dataloader):
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

        # Synchronize loss across GPUs
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size  # Average loss across all GPUs

        running_loss += loss_tensor.item()
        if rank == 0 and (i + 1) % 16 == 0:
            avg_loss = running_loss / 16
            # t.set_description_str(f"Training AE... Loss={avg_loss:.4f}")
            # t.set_postfix(batches=f"{i+1}/{len(dataloader)}", gpus=f"{world_size}")
            tqdm.write(f"Training AE... Loss={avg_loss:.4f} | Batches {i+1}/{len(dataloader)} | GPUs {world_size}")
            running_loss = 0  # Reset loss counter

        # save checkpoint

    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Record log of data size.
        log_data_size = int(np.log2(np.max((data_size, 1))))

        os.makedirs(os.path.join(MODELS_DIR, tag), exist_ok=True)
        path = os.path.join(MODELS_DIR, tag, f'ae-{hidden_units}-i{img_size}-d{log_data_size}.trace.pt')
        
        # Save the model using TorchScript trace
        # Unwrap from DDP
        temp = io.BytesIO()
        torch.save(ddp_model.module.state_dict(), temp)
        temp.seek(0)
        state_dict = torch.load(temp)
        model_copy = Autoencoder(hidden_units=hidden_units, input_dim=img_size).to(device)
        model_copy.load_state_dict(state_dict)

        # Trace and save the model
        model_copy.eval()
        scripted_model = torch.jit.trace(model_copy, torch.randn(1, 1, img_size, img_size).to(device))
        torch.jit.save(scripted_model, path)

        logging.info(f'AE model saved to {path}')

def train_ae_process(rank, world_size):
    
    process_group_setup(rank, world_size)

    try:
      train_ae(
          rank,
          img_size=64,
          data_size=2**14,
          hidden_units=2,
          tag='small-autoencoder'
      )
    except Exception as e:
      logging.error(f"Error in rank {rank}:")
      logging.error(e, exc_info=True)
    finally:
      dist.barrier()
      dist.destroy_process_group()


if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()
    print("available gpus:", world_size)

    mp.spawn(
        train_ae_process,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


"""
torchrun --nproc_per_node=4 train_ae.py
"""
