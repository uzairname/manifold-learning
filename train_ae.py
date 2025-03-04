import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import os

from models import DeepAutoencoder, Autoencoder
from datasets.clock import IMG_SIZE, ClockDataset
from tqdm import tqdm
from config import MODELS_DIR


os.environ["OMP_NUM_THREADS"] = "1"

def train_ae(
    img_size=IMG_SIZE, 
    data_size=2**20, 
    hidden_units=2,
    batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.001
):
    torch.manual_seed(42)

    print("available gpus", torch.cuda.device_count())
    
    # Initialize process group for distributed training
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
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
    t = tqdm(enumerate(dataloader), total=len(dataloader))
    runnnig_loss = 0
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

        # Synchronize loss across GPUs
        loss_tensor = torch.tensor(loss.item(), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= world_size  # Average loss across all GPUs

        runnnig_loss += loss_tensor.item()
        if i % 16 == 0:
          t.set_description_str(f"Training AE... Loss={runnnig_loss / 16:.4f}")
          t.set_postfix(batches=f"{i}/{len(dataloader)}", gpus=f"{world_size}")
          runnnig_loss = 0

    if rank == 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        # Record log of n params and data size, to nearest 2^x
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        log_params = int(np.log2(num_params))
        log_data_size = int(np.log2(np.max((data_size, 1))))
        path = os.path.join(MODELS_DIR, f'ae-{hidden_units}-d{log_data_size}-p{log_params}-i{img_size}.pt')

        # Save the model using TorchScript

        # Unwrap from DDP
        torch.save(ddp_model.module.state_dict(), "temp.pth")
        state_dict = torch.load("temp.pth")
        model_copy = Autoencoder(hidden_units=hidden_units, input_dim=img_size).to(device)
        model_copy.load_state_dict(state_dict)

        # Script and save the model
        scripted_model = torch.jit.script(model_copy)
        torch.jit.save(scripted_model, path)

        print(f'AE model saved to {path}')

    dist.destroy_process_group()


if __name__ == "__main__":
    train_ae(
      img_size=256,
      data_size=2**23,
      hidden_units=2
    )


"""
torchrun --nproc_per_node=4 train_ae.py
"""
