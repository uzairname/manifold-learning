import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import csv

# Create output directory
IMG_DIR = "data/clock_images"

# Define the fixed output size
IMG_SIZE = 64

LABELS_FILE = "data/clock_labels.csv"

def draw_clock(hour, minute, save_path, img_size=IMG_SIZE):
    """Generates a clock image at a given hour/minute and resizes it."""
    fig, ax = plt.subplots(figsize=(2,2), dpi=100)  # Higher DPI for better quality

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Draw clock face
    circle = plt.Circle((0, 0), 1, edgecolor="black", facecolor="white", lw=3)
    ax.add_patch(circle)

    # Compute hand angles
    hour_angle = np.pi / 2 - (2 * np.pi * (hour % 12) / 12) - (2 * np.pi * (minute / 60) / 12)
    minute_angle = np.pi / 2 - (2 * np.pi * minute / 60)

    # Draw hands
    ax.plot([0, 0.5 * np.cos(hour_angle)], [0, 0.5 * np.sin(hour_angle)], 'k', lw=5)  # Hour hand
    ax.plot([0, 0.8 * np.cos(minute_angle)], [0, 0.8 * np.sin(minute_angle)], 'k', lw=3)  # Minute hand

    # Save image with padding
    plt.axis('off')
    temp_path = "data/temp.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0.1)  # Small padding
    plt.close()

    # Load image and resize
    img = Image.open(temp_path).convert("L")  # Convert to grayscale
    img = img.resize((img_size, img_size), Image.LANCZOS)  # Resize to power of 2
    img.save(save_path)




# Custom dataset class
class ClockDataset(Dataset):
    def __init__(self, img_dir, supervised):
        
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        self.img_dir = img_dir
        self.supervised = supervised
        self.images = os.listdir(img_dir)
        self.labels_df = pd.read_csv("data/clock_labels.csv")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        if self.supervised:
            label_2d, label_1d = self.get_label(img_name)
            return image, label_2d, label_1d
        return image
    

    def get_label(self, img_name):
      row = self.labels_df[self.labels_df["filename"] == img_name].iloc[0]

      return (
          torch.tensor([row["hour"], row["minute"]], dtype=torch.float32),  # 2D label
          torch.tensor(row["time_in_minutes"], dtype=torch.float32)  # 1D label
      )


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    # Open CSV file for writing labels
    with open(LABELS_FILE, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "hour", "minute", "time_in_minutes"])

        # Generate clock images at different times
        for hour in range(12):  # 12-hour format
            for minute in range(0, 60, 1):  # Every 5 minutes
                filename = f"clock_{hour:02d}_{minute:02d}.png"
                save_path = os.path.join(IMG_DIR, filename)
                draw_clock(hour, minute, save_path)

                # Compute single number label (time in minutes past midnight)
                time_in_minutes = hour * 60 + minute

                # normalize minute, hour, and total minutes
                hour_label = hour / 12
                minute_label = minute / 60
                time_in_minutes_label = time_in_minutes / (12 * 60)

                # Write to CSV
                writer.writerow([filename, hour_label, minute_label, time_in_minutes_label])

    print(f"Clock dataset saved in '{IMG_DIR}', resized to {IMG_SIZE}x{IMG_SIZE}")
    print(f"Labels saved in '{LABELS_FILE}'")
