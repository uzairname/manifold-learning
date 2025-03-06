import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

import numpy as np
import os

from datasets.clock import IMG_SIZE, ClockConfig, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR
import io
import logging


def train_regressor(
    model_class: nn.Module,
    rank=None,
    run = None,
    img_size=IMG_SIZE, 
    data_size=2**20, 
    out_dim=2,
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
    model = model_class(img_size=img_size, out_dim=out_dim, device=device).to(device)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # AMP initialization
    scaler = torch.amp.GradScaler()

    # Training loop

    log_data_size = int(np.log2(np.max((data_size, 1))))
    path = os.path.join(MODELS_DIR, save_dir, f'{out_dim}-i{img_size}-d{log_data_size}.pt')
    os.makedirs(os.path.join(MODELS_DIR, save_dir), exist_ok=True)

    print("training")
    running_loss = 0
    t = tqdm(enumerate(dataloader), total=len(dataloader), disable=(rank != 0))
    for i, batch in t:
        inputs, labels2d, labels1d = batch
        input = labels2d if out_dim == 2 else labels1d
        input, label = inputs.to(device), input.to(device)

        optimizer.zero_grad()

        # Forward pass with AMP broadcast for mixed precision
        with torch.amp.autocast(device_type='cuda'):
          predicted = ddp_model(input)
          loss = criterion(predicted, label)

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
            t.set_description_str(f"Training Regressor... Loss={avg_loss:.4f}")
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
        model_copy = model_class(out_dim=out_dim, img_size=img_size, device=device).to(device)
        model_copy.load_state_dict(state_dict)

        # Trace and save the model
        model_copy.eval()
        scripted_model = torch.jit.trace(model_copy, torch.randn(1, 1, img_size, img_size).to(device))
        torch.jit.save(scripted_model, path)

        logging.info(f'Regression model saved to {path}')
        
    if run is not None:
        run.finish()


"""
torchrun --nproc_per_node=4 train_ae.py
"""
