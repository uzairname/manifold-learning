import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

import numpy as np
import os

from models.clock_models import DeepMultiHeadRegressor
from datasets.clock import IMG_SIZE, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR


os.environ["OMP_NUM_THREADS"] = "1"

def train_clock_regressor(img_size=IMG_SIZE, data_size=2**20):
    torch.manual_seed(42)
    
    # Initialize process group for distributed training
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    BATCH_SIZE = 64
    OUT_DIM = 2
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.001

    # Load dataset with DistributedSampler
    dataset = ClockDataset(len=data_size, img_size=IMG_SIZE, supervised=True, augment=False)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)

    # Initialize model and wrap with DistributedDataParallel
    model = DeepMultiHeadRegressor(out_dim=OUT_DIM, input_dim=IMG_SIZE).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop
    t = tqdm(enumerate(dataloader), total=len(dataloader))
    runnnig_loss = 0
    for i, batch in t:
        inputs, labels2d, labels1d = batch
        labels = labels2d if OUT_DIM == 2 else labels1d

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP broadcast for mixed precision
        with torch.amp.autocast(device_type='cuda'):
          predicted = model(inputs)
          loss = criterion(predicted, labels)

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
          t.set_description_str(f"Training Regressor... Loss={runnnig_loss / 16:.4f}")
          t.set_postfix(batches=f"{(i)}/{len(dataloader)} (1/{world_size})")
          runnnig_loss = 0

    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # count number of parameters, to nearest 10^x
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_params = int(np.log2(num_params))
        log_data_size = int(np.log2(np.max((data_size, 1))))
        path = os.path.join(MODELS_DIR, f'reg-{OUT_DIM}-d{log_data_size}-p{log_params}-i{img_size}.pt')
        torch.save(model.module, path)

        print(f'Regression model saved to {path}')

    dist.destroy_process_group()


if __name__ == "__main__":

    train_clock_regressor(
       img_size=128, 
       data_size=2**23
    )


"""
torchrun --nproc_per_node=4 train_regressor.py
"""
