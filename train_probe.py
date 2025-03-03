

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models import ClockRegressor
from clock_dataset import IMG_DIR, ClockDataset
import time
from models import ClockRegressorAE
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import trange

            
# Define dataset for extracted features
class FeatureDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

OUT_DIM = 2
MODELS_DIR = "models"

def train_probe(model_file):
    
    dist.init_process_group(backend="nccl")  # Initialize distributed training

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    pretrained_model = ClockRegressor(out_dim=OUT_DIM, input_dim=128).to('cuda')
    model_path = os.path.join(MODELS_DIR, model_file)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_model.eval()

    # Get some data points by running the model on a batch
    dataset = ClockDataset(img_dir=IMG_DIR, supervised=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to('cuda')
            latents.append(pretrained_model.get_latent(batch).cpu())

    latents = torch.cat(latents, dim=0)
        
    # Create a dataset for the extracted features
    dataset = FeatureDataset(latents.cpu())  # Move to CPU for DataLoader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
        
    # Define a new model for the probe task
    probe_model = ClockRegressorAE(hidden_units=2).to(device)
    probe_model = nn.parallel.DistributedDataParallel(probe_model, device_ids=[rank])

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(probe_model.parameters(), lr=0.0005, weight_decay=0.001)

    # Training loop
    num_epochs = 100
    t = trange(num_epochs, desc=f"Loss: N/A", leave=True)
    for epoch in t:
        sampler.set_epoch(epoch)  # Ensures different shuffling each epoch
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            reconstructed, _ = probe_model(batch)
            loss = criterion(reconstructed, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss for each epoch
        if rank == 0:
            t.set_description(f"Loss: {loss.item():.4f}", refresh=True)

    # Save model (only on rank 0)
    if rank == 0:
        torch.save(probe_model.module.state_dict(), f"models/ae_{model_file}")
        print("Autoencoder training complete!")


    dist.destroy_process_group()  # Clean up the process group


if __name__ == "__main__":
    model_file = f"regressor-{OUT_DIM}-e100-p7.pth"
    train_probe(model_file)


"""
torchrun --nproc_per_node=4 train_probe.py
"""
