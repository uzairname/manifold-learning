import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import io
import random

# Define the fixed output size
IMG_SIZE = 128

class ClockGenerator:
    def __init__(self, img_size=IMG_SIZE, device='cuda', minute_hand_len=1):
        """
        Initializes the clock generator with a coordinate grid to optimize clock image generation.
        """
        self.img_size = img_size
        self.device = device
        self.minute_hand_len = minute_hand_len

        # Create a coordinate grid and store it as a tensor (persistent buffer)
        y, x = torch.meshgrid(torch.linspace(1, -1, img_size, device=device),
                              torch.linspace(-1, 1, img_size, device=device),
                              indexing='ij')

        # Store as attributes to avoid recomputation
        self.x, self.y = x, y
        self.radius = torch.hypot(x, y)  # Equivalent to sqrt(x^2 + y^2)

        # Precomputed clock face (remains the same for every image)
        self.clock_face = (self.radius < 0.97)
        self.outside_clock = (self.radius > .99).float()  # 1 outside circle, 0 inside

    def generate_clock_tensor(self, hour: torch.Tensor, minute: torch.Tensor):
        """
        Generates a clock image tensor (grayscale, shape: (1, img_size, img_size)) 
        optimized for fast generation during model training.
        
        Args:
            hour (int): Hour (0-11)
            minute (float): Minute (0-59)
            
        Returns:
            clock_tensor (torch.Tensor): (1, img_size, img_size) tensor (on the same device)
        """

        # Ensure inputs are tensors (to avoid explicit conversions later)
        hour_angle = 2 * np.pi * ((hour % 12) / 12 + minute / (12 * 60))
        minute_angle = 2 * np.pi * (minute / 60)

        # Function to generate a one-directional hand mask
        def draw_hand(angle, length, thickness):
            # Compute distance of each pixel from the hand's centerline

            width_mask = torch.abs(self.y * torch.cos(angle + np.pi / 2) + self.x * torch.sin(angle + np.pi / 2)) < thickness / 2
            length_mask = torch.abs(self.y * torch.cos(angle) + self.x * torch.sin(angle) - length / 2) < length / 2
            # Combine masks to get the hand
            return (width_mask & length_mask).float()

        # Generate hands with correct length
        hour_hand = draw_hand(hour_angle, length=0.5, thickness=0.1)  # Shorter, thicker
        minute_hand = draw_hand(minute_angle, length=self.minute_hand_len, thickness=0.05)  # Longer, thinner

        # Combine clock face and hands
        clock_tensor = self.outside_clock + self.clock_face - (hour_hand + minute_hand)
        clock_tensor = torch.clamp(clock_tensor, 0, 1)  # Ensure values in [0,1]
        return clock_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)


class ClockDataset(Dataset):
    def __init__(self, device='cpu', len=2**12, img_size=128, supervised=True, augment=True, noise_std=0.05, translate_px=1, minute_hand_len=1):
        """
        Args:
            time_samples (list of tuples): List of (hour, minute) tuples.
            img_size (int): Size of output images.
            noise_std (float): Standard deviation of Gaussian noise.
            translate_px (int): Maximum translation (in pixels).
            augment (bool): Whether to apply augmentation.
        """
        self.len = len
        self.img_size = img_size
        self.supervised = supervised
        self.augment = augment
        self.noise_std = noise_std
        self.translate_px = translate_px
        self.clock_generator = ClockGenerator(img_size=img_size, device=device, minute_hand_len=minute_hand_len)
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get time of day as a fraction of 1
        phi = torch.tensor((1 + np.sqrt(5)) / 2, dtype=torch.float16).to(self.device) # Golden ratio
        time_of_day = (phi * idx) % 1

        # Convert to hour and minute
        total_minutes = time_of_day * 12 * 60 # in [0, 12*60)
        hour = torch.floor(total_minutes / 60) # in [0, 12)
        minute = total_minutes % 60 # in [0, 60)
        # total_minutes, hour, and minute are float16 tensors

        clock_tensor = self.clock_generator.generate_clock_tensor(hour, minute).to(self.device)

        if self.augment:
            # Set seed for deterministic augmentation
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)

            # Generate deterministic translation offsets
            max_t = self.translate_px
            tx = random.randint(-max_t, max_t)
            ty = random.randint(-max_t, max_t)

            # Apply translation
            clock_tensor = TF.affine(clock_tensor, angle=0, translate=(tx, ty), scale=1.0, shear=0, fill=1.0)

            # Add Gaussian noise
            clock_tensor = clock_tensor + torch.randn_like(clock_tensor) * self.noise_std
            clock_tensor = torch.clamp(clock_tensor, 0, 1)

        if self.supervised:
            # Convert time to labels in [0, 1)
            hour_label = hour / 12
            minute_label = minute / 60
            return clock_tensor, torch.stack([hour_label, minute_label]), time_of_day
      
        return clock_tensor
