import torch
from torch.utils.data import DataLoader
from datasets.clock import ClockConfig, ClockDataset
import matplotlib.pyplot as plt
import os
from config import MODELS_DIR
import typing
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_size = 4096

def load_data(
  img_size=128,
  batch_size=64,
  data_size=data_size,
  data_config=None
): 
  """
  Load the dataset for inference.
  """
  # Load dataset
  dataset = ClockDataset(device=device, len=data_size, img_size=img_size, augment=False, config=data_config)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type != 'cuda')
  
  return dataloader

def load_model_and_data(
    img_size=128,
    batch_size=64,
    data_size=data_size,
    latent_dim=2,
    postfix='',
    name='model',
    data_config=None,
    checkpoint=None
):
    """
    Load the model and dataset for inference.
    """
    # Load dataset
    dataloader = load_data(
        img_size=img_size,
        batch_size=batch_size,
        data_size=data_size,
        data_config=data_config
    )
    
    # Load model

    model_file = f"{latent_dim}-i{img_size}{postfix}"
    
    if checkpoint is None:
      model_path = os.path.join(MODELS_DIR, name, model_file, f"final.pt")
    else:
      model_path = os.path.join(MODELS_DIR, name, model_file, f"{checkpoint}.pt")
    
    model = torch.jit.load(model_path).to(device)
    model.eval()
    
    return model, dataloader



def print_model_parameters(cls: nn.Module, img_size=128, latent_dim=2):
  
    model = cls(img_size=img_size, latent_dim=latent_dim).to(device)

    print(f"{'Layer':<40}{'Param Count':>15}")
    print("-" * 60)
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"{name:<40}{param_count:>15}")
    
    print("-" * 60)
    print(f"{'Total Trainable Parameters':<40}{total_params:>15,}")


def get_outputs(type_: typing.Literal['encoder', 'autoencoder', 'decoder'], model, dataloader):
  with torch.no_grad():
    for _, clean_imgs, label2d, label1d in dataloader:
      images = clean_imgs.to(device)
      label2d = label2d.to(device)
      
      latents = None
      reconstructeds = None
      if type_ == 'encoder':
        latents = model.forward(images)
      
      if type_ == 'autoencoder':
        latents = model.encoder(images)
        reconstructeds = model.forward(images)

      elif type_ == 'decoder':
        reconstructeds = model.forward(label2d)
      
      for i in range(images.size(0)):
        latent = latents[i] if latents is not None else None
        reconstructed = reconstructeds[i] if reconstructeds is not None else None
        yield images[i], label1d[i], label2d[i], latent, reconstructed


def show_data(dataloader: DataLoader):
  
  # visualize 16 images
  fig, axs = plt.subplots(4, 4, figsize=(10, 10))
  fig.tight_layout()
  imgs, clean_imgs, labels2d, label1d = next(iter(dataloader))
  for i in range(16):
      if i >= 16:
        break
      
      img = imgs[i]
      label2d = labels2d[i]

      label2d_unnormalized = (label2d * torch.tensor([12, 60]).to(device).float())

      hour = label2d_unnormalized[0]
      minute = label2d_unnormalized[1]

      axs[i // 4, i % 4].imshow(img.squeeze().cpu(), cmap='gray')
      axs[i // 4, i % 4].set_title(f"{hour:.0f}h{minute:.0f}m")
      axs[i // 4, i % 4].axis('off')

  plt.show()
  

def visualize_reconstruction(type_, model, dataloader):
  if type_ == 'encoder':
    print("Encoder")
    return

  n=6
  s=3
  fig, axs = plt.subplots(n, 2, figsize=(2*s, n*s))
  
  if type_ == 'decoder':
    fig.suptitle('Decoder Reconstructions')
  elif type_ == 'autoencoder':
    fig.suptitle('Autoencoder Reconstructions')

  for i, (img, label1d, label2d, latent, reconstructed) in enumerate(get_outputs(type_, model, dataloader)):
    if i >= n:
      break
    
    loss = torch.nn.functional.mse_loss(img, reconstructed)
    
    axs[i,0].imshow(img.squeeze().cpu(), cmap='gray')
    axs[i,0].set_title(f"Original: {label1d:.0f}h{label2d[1]:.0f}m")
    axs[i,0].axis('off')
    
    axs[i,1].imshow(reconstructed.squeeze().cpu(), cmap='gray')
    axs[i,1].set_title(f"Loss: {loss:.4f}")
    axs[i,1].axis('off')

  plt.show()



def visualize_predictions(type_, model, dataloader):
  if type_ != 'encoder':
    print("Not encoder")
    return
  
  n=4
  s=3
  fig, axs = plt.subplots(n, n, figsize=(n*s, 1.2*n*s))
  fig.suptitle('Encoder Predictions')
    
  for i, (img, label1d, label2d, latent, reconstructed) in enumerate(get_outputs(type_, model, dataloader)):
    if i >= n**2:
      break
    
    unnormalized_pred = (latent * torch.tensor([12, 60]).to(device).float())
    unnormalized_labels = (label2d * torch.tensor([12, 60]).to(device).float())
    
    pred_hour = unnormalized_pred[0]
    pred_minute = unnormalized_pred[1]
    
    true_hour = unnormalized_labels[0]
    true_minute = unnormalized_labels[1]
    
    loss = torch.nn.functional.mse_loss(img, latent)
    
    axs[i//n, i%n].imshow(img.squeeze().cpu(), cmap='gray')
    axs[i//n, i%n].set_title(f"Pred: {pred_hour:.0f}h{pred_minute:.0f}m\nTrue: {true_hour:.0f}h{true_minute:.0f}m\nLoss: {loss:.4f}")
    axs[i//n, i%n].axis('off')
    
    

def visualize_latent():
  if (LATENT_DIM <= 2):
    # Plot latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents[:, 0], label1d if LATENT_DIM==1 else latents[:,1], c=label1d, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Time in minutes past midnight")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Time in minutes past midnight" if LATENT_DIM==1 else "Latent Dimension 2")
    plt.title("Learned Manifold of Autoencoder (Clock Dataset)")
    plt.show()
  else:
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    # Plot PCA-reduced latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=label1d, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Time in minutes past midnight")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.title("Learned Manifold of Autoencoder (Clock Dataset)")
    plt.show()