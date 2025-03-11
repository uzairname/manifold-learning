import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import io
import random
from dataclasses import dataclass
import typing

# Define the fixed output size
IMG_SIZE = 128


@dataclass
class ClockConfig:
    minute_hand_len: int = 1
    minute_hand_start: float = 0.5
    minute_hand_width: float = 0.1
    hour_hand_len: int = 0.5
    hour_hand_start: float = 0
    hour_hand_width: float = 0.1
    angle_quantization: int = None


class ClockGenerator:
    def __init__(self, img_size=IMG_SIZE, device='cuda', config: ClockConfig = None):
        """
        Initializes the clock generator with a coordinate grid to optimize clock image generation.
        """
        self.img_size = img_size
        self.device = device
        
        if config is None:
            config = ClockConfig()
        else: 
            self.config = config

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


    def generate_clock_tensor(
      self, 
      hour: torch.Tensor, 
      minute: torch.Tensor,
      quantization=None,
      hand_width=None,
    ):
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

        _quantization = quantization or self.config.angle_quantization or 0

        # Quantize angles if specified
        if _quantization > 0:
            hour_angle = torch.round(hour_angle * _quantization / (2*torch.pi)) * (2*torch.pi) / _quantization
            minute_angle = torch.round(minute_angle * _quantization / (2*torch.pi)) * (2*torch.pi) / _quantization

        # Function to generate a one-directional hand mask
        def draw_hand(angle, start, length, width):
            # Compute distance of each pixel from the hand's centerline

            width_mask = torch.abs(self.y * torch.cos(angle + np.pi / 2) + self.x * torch.sin(angle + np.pi / 2)) < width / 2
            length_mask = torch.abs(self.y * torch.cos(angle) + self.x * torch.sin(angle) - (length + start) / 2) < (length - start) / 2
            # Combine masks to get the hand
            return (width_mask & length_mask).to(torch.float32)

        hour_hand_width = hand_width or self.config.hour_hand_width
        minute_hand_width = hand_width or self.config.minute_hand_width

        # Generate hands with correct length
        hour_hand = draw_hand(hour_angle, start=self.config.hour_hand_start, length=self.config.hour_hand_len, width=hour_hand_width)
        minute_hand = draw_hand(minute_angle, start=self.config.minute_hand_start, length=self.config.minute_hand_len, width=minute_hand_width)

        # Combine clock face and hands
        clock_tensor = self.outside_clock + self.clock_face - (hour_hand + minute_hand)
        clock_tensor = torch.clamp(clock_tensor, 0, 1)  # Ensure values in [0,1]
        return clock_tensor.unsqueeze(0)  # Add channel dimension (1, H, W)


@dataclass
class ClockDatasetConfig:
  device: int='cpu'
  data_size: int=2**12
  img_size: int=128
  augment: dict=None
  quantization_scheduler: typing.Callable[[int], int] = None
  hand_width_scheduler: typing.Callable[[int], float] = None
  initial_time: float= torch.sqrt( torch.tensor(2) ) - 1


class ClockDataset(Dataset):
    def __init__(
      self, 
      dataset_config: ClockDatasetConfig = ClockDatasetConfig(),
      clock_config: ClockConfig = ClockConfig(),
    ):
        """
        Args:
            time_samples (list of tuples): List of (hour, minute) tuples.
            img_size (int): Size of output images.
            noise_std (float): Standard deviation of Gaussian noise.
            translate_px (int): Maximum translation (in pixels).
            augment (bool): Whether to apply augmentation.
        """
        self.config = dataset_config

        if self.config.augment is not None:
            self.noise_std = self.config.augment.get('noise_std', 0.0)
            self.blur = self.config.augment.get('blur', 0.0)

        self.generator = ClockGenerator(img_size=self.config.img_size, device=self.config.device, config=clock_config)

    def __len__(self):
        return self.config.data_size

    def set_quantization(self, quantization):
        self.generator.config.angle_quantization = quantization

    def __getitem__(self, idx):
      # Get time of day as a fraction of 1
      phi = torch.tensor((1 + np.sqrt(5)) / 2, dtype=torch.float32).to(self.config.device) # Golden ratio
      time_of_day = ( phi * (idx) + self.config.initial_time ) % 1

      # Convert to hour and minute
      total_minutes = time_of_day * 12 * 60 # in [0, 12*60)
      hour = torch.floor(total_minutes / 60) # in [0, 12)
      minute = total_minutes % 60 # in [0, 60)
      # total_minutes, hour, and minute are float16 tensors

      # Convert time to labels in [0, 1)
      hour_label = hour / 12
      minute_label = minute / 60

      # Get quantization amount
      quantization = self.config.quantization_scheduler(idx) if self.config.quantization_scheduler else None
      hand_width = self.config.hand_width_scheduler(idx) if self.config.hand_width_scheduler else None
      
      # Generate clock tensor
      clock_tensor = self.generator.generate_clock_tensor(hour, minute, quantization, hand_width).to(self.config.device)

      if self.config.augment is not None:
          # Set seed for deterministic augmentation
          random.seed(idx)
          np.random.seed(idx)
          torch.manual_seed(idx)

          # Add Gaussian noise
          noisy_clock_tensor = clock_tensor + torch.randn_like(clock_tensor) * self.noise_std
          
          # Apply Gaussian blur by sampling from a gaussian kernel and convolving
          if self.blur > 0:
              kernel_size = int(2 * np.ceil(2 * self.blur) + 1)
              kernel = torch.exp(-torch.arange(-kernel_size//2, kernel_size//2, dtype=torch.float32)**2 / (2 * self.blur**2))
              kernel /= kernel.sum()
              noisy_clock_tensor = torch.nn.functional.conv2d(noisy_clock_tensor, kernel[None, None, :, None], padding=kernel_size//2)
          
          noisy_clock_tensor = torch.clamp(noisy_clock_tensor, 0, 1)
      else:
          noisy_clock_tensor = clock_tensor
          
          
      assert not torch.isnan(noisy_clock_tensor).any(), "NaNs in input!"
      assert 0 <= noisy_clock_tensor.min() and noisy_clock_tensor.max() <= 1, f"Input range invalid: {noisy_clock_tensor.min()}, {noisy_clock_tensor.max()}"

      return noisy_clock_tensor, clock_tensor, torch.stack([hour_label, minute_label]), time_of_day




