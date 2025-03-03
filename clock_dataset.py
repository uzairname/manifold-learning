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
    def __init__(self, img_size=IMG_SIZE, device='cuda'):
        """
        Initializes the clock generator with a coordinate grid to optimize clock image generation.
        """
        self.img_size = img_size
        self.device = device

        # Create a coordinate grid ONCE and store it as a tensor (persistent buffer)
        y, x = torch.meshgrid(torch.linspace(-1, 1, img_size, device=device),
                              torch.linspace(-1, 1, img_size, device=device),
                              indexing='ij')

        # Store as attributes to avoid recomputation
        self.x, self.y = x, y
        self.radius = torch.hypot(x, y)  # Equivalent to sqrt(x^2 + y^2)

        # Precomputed clock face (remains the same for every image)
        self.clock_face = (self.radius <= 0.95)
        self.outside_clock = (self.radius > 1).float()  # 1 outside circle, 0 inside

    def generate_clock_tensor(self, hour, minute):
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
        hour = torch.tensor(hour, dtype=torch.float16, device=self.device)
        minute = torch.tensor(minute, dtype=torch.float16, device=self.device)

        # Corrected angles for clockwise rotation
        hour_angle = -2 * np.pi * ((hour % 12) / 12 + minute / (12 * 60))
        minute_angle = -2 * np.pi * (minute / 60)

        # Function to generate a one-directional hand mask
        def draw_hand(angle, length, thickness):
            x_hand = torch.cos(angle)
            y_hand = torch.sin(angle)

            # Compute distance of each pixel from the hand's centerline
            distance_mask = torch.abs((self.y * x_hand - self.x * y_hand)) < thickness

            # Ensure the hand **only extends outward up to `length`**
            radial_mask = (self.x * x_hand + self.y * y_hand) > 0  # Ensure pixels are in the correct direction
            length_mask = self.radius <= length  # Ensure pixels are within hand length

            return (distance_mask & radial_mask & length_mask).float()

        # Generate hands with correct length
        hour_hand = draw_hand(hour_angle, length=0.5, thickness=0.06)  # Shorter, thicker
        minute_hand = draw_hand(minute_angle, length=1, thickness=0.03)  # Longer, thinner

        # Combine clock face and hands
        clock_tensor = self.outside_clock + self.clock_face - (hour_hand + minute_hand)
        clock_tensor = torch.clamp(clock_tensor, 0, 1)  # Ensure values in [0,1]
        return clock_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)
    


import torch

def add_procedural_noise(image, min_freq=-8, max_freq=8, noise_scale=0.1):
    """
    Adds procedural noise to an image using multiple frequency bands.

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        min_freq (int): Minimum exponent for frequency range (default -8).
        max_freq (int): Maximum exponent for frequency range (default 8).
        noise_scale (float): Scaling factor for noise strength.

    Returns:
        torch.Tensor: Noisy image of the same shape as `image`.
    """
    # Ensure image is a float tensor
    image = image.float()

    # Generate noise at multiple frequency bands
    noise_frequencies = 2**torch.arange(min_freq, max_freq, device=image.device, dtype=torch.float32)

    # Initialize noise tensor
    noise = torch.zeros_like(image)

    # Loop over frequencies and apply sinusoidal noise
    for freq in noise_frequencies:
        # Generate per-pixel phase shift
        phase_shift = torch.rand(1, 1, *image.shape[1:], device=image.device)

        # Generate sinusoidal noise pattern
        sin_noise = torch.sin(2 * np.pi * freq * phase_shift)

        # Apply noise modulation with random intensity
        noise += noise_scale * sin_noise * torch.randn_like(image)

    # Add noise to the image
    noisy_image = torch.clamp(image + noise, 0, 1)  # Keep values in [0,1]

    return noisy_image


class ClockDataset(Dataset):
    def __init__(self, device='cpu', len=2**12, img_size=128, supervised=True, augment=True, noise_std=0.1, translate_px=1):
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
        self.clock_generator = ClockGenerator(img_size=img_size, device=device)
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get time of day in minutes based on index
        day_minutes = (idx * np.pi) % (12 * 60)

        # Convert to hour and minute
        hour = int(day_minutes // 60)
        minute = day_minutes % 60

        # normalize minute, hour, and total minutes
        hour_label = hour / 12
        minute_label = minute / 60
        time_in_minutes_label = day_minutes / (12 * 60)

        # Generate clock image
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

            # Apply Procedural noise
            clock_tensor = add_procedural_noise(clock_tensor, noise_scale=self.noise_std)

            clock_tensor = torch.clamp(clock_tensor, 0, 1)

        if self.supervised:
            return clock_tensor, torch.tensor([hour_label, minute_label]).to(self.device), torch.tensor([time_in_minutes_label]).to(self.device)
      
        return clock_tensor
