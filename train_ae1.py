import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os

from models import ClockRegressor, ClockRegressorAE, DeepAutoencoder
from clock_dataset import IMG_SIZE, ClockDataset
import time
from tqdm import tqdm, trange

MODELS_DIR = "models"
IMG_SIZE = 128

os.environ["OMP_NUM_THREADS"] = "1"

def main():
    torch.manual_seed(42)
    
    # Initialize process group for distributed training
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    BATCH_SIZE = 64
     # 1 or 2
    HIDDEN_UNITS = 2
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.001

    # Load dataset with DistributedSampler
    # data_size = 2**14
    data_size=2**23
    dataset = ClockDataset(len=data_size, supervised=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)

    # Initialize model and wrap with DistributedDataParallel
    model = DeepAutoencoder(hidden_units=HIDDEN_UNITS, input_dim=IMG_SIZE).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop
    t = tqdm(enumerate(dataloader), total=len(dataloader))
    runnnig_loss = 0
    for i, batch in t:
        inputs = batch[0]
        inputs = inputs.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP broadcast for mixed precision
        with torch.amp.autocast(device_type='cuda'):
          predicted, _ = model(inputs)
          loss = criterion(predicted, inputs)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Synchronize loss across GPUs
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size  # Average loss across all GPUs

        runnnig_loss += loss_tensor.item()
        if i % 16 == 0:
          t.set_description_str(f"Training... Loss={runnnig_loss / 16:.4f}")
          runnnig_loss = 0

    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # count number of parameters, to nearest 10^x
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_params = int(np.log10(num_params))
        log_data_size = int(np.log10(np.max((data_size, 1))))
        path = os.path.join(MODELS_DIR, f'ae-{HIDDEN_UNITS}-d{log_data_size}-p{log_params}.pt')
        torch.save(model.module, path)

        print(f'AE model saved to {path}')

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


"""
torchrun --nproc_per_node=4 train_ae.py
"""
