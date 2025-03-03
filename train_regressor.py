import torch
import torch.nn as nn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os

from models import ClockRegressor, ClockRegressorAE
from clock_dataset import IMG_SIZE, ClockDataset
import time
from tqdm import tqdm

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

    BATCH_SIZE = 32
     # 1 or 2
    OUT_DIM = 2
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.0001

    # Load dataset with DistributedSampler
    dataset = ClockDataset(supervised=True, device='cpu')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # Initialize model and wrap with DistributedDataParallel
    model = ClockRegressor(out_dim=OUT_DIM, input_dim=IMG_SIZE).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    num_epochs = 10
    loss_tensor = None
    for epoch in tqdm(range(num_epochs), desc=f"Loss: {loss_tensor.item() if loss_tensor is not None else 'N/A'}"):
        sampler.set_epoch(epoch)  # Ensure shuffling is different each epoch
        for batch in dataloader:
            inputs, labels2d, labels1d = batch
            inputs, labels2d, labels1d = inputs.to(device), labels2d.to(device), labels1d.to(device)

            optimizer.zero_grad()

            # Forward pass with AMP broadcast for mixed precision
            with torch.amp.autocast(device_type='cuda'):
              predicted = model(inputs)
              loss = criterion(predicted, labels1d if OUT_DIM == 1 else labels2d)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Synchronize loss across GPUs
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size  # Average loss across all GPUs

        if rank == 0:
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_tensor.item():.4f}')

    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # count number of parameters, to nearest 10^x
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_params = int(np.log10(num_params))
        path = os.path.join(MODELS_DIR, f'regressor-{OUT_DIM}-e{num_epochs}-p{log_params}.pth')
        torch.save(model.module.state_dict(), path)

        print(f'Regressor model saved to {path}')

    dist.destroy_process_group()


if __name__ == "__main__":
    main()


"""
torchrun --nproc_per_node=4 train_regressor.py
"""
