import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from config import MODELS_DIR
import typing
from torch import nn
from functools import partial
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(
    img_size=128,
    latent_dim=2,
    postfix='',
    name='model',
    checkpoint=None
):
    """
    Load the trained model and dataset for inference.
    """

    
    # Load model
    model_file = f"{latent_dim}-i{img_size}-{postfix}"
    
    if checkpoint is None:
      model_path = os.path.join(MODELS_DIR, name, model_file, f"final.pt")
    else:
      model_path = os.path.join(MODELS_DIR, name, model_file, f"{checkpoint}.pt")
    
    model = torch.jit.load(model_path).to(device)
    model.eval()
    
    return model


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
  """
  yields:
    - image, label1d, label2d, latent, reconstructed
  """
  with torch.no_grad():
    for _, clean_imgs, label2d, label1d in dataloader:
      images = clean_imgs.to(device)
      label1d = label1d.to(device)
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


def eval_model(
  type_: typing.Literal['encoder', 'autoencoder', 'decoder'],
  model: nn.Module,
  val_data: typing.List,
  device: str,
  criterion: nn.Module=nn.MSELoss(),
  latent_dim: int=2,
):
  
  val_loss = 0
  model.eval()
  for i, batch in enumerate(val_data):
      imgs, clean_imgs, labels2d, labels1d = batch
      labels = labels1d.unsqueeze(1) if latent_dim == 1 else labels2d

      if type_ == "encoder":
          input = imgs.to(device)
          output = labels.to(device)
          
      elif type_ == "decoder":
          input = labels.to(device)
          output = clean_imgs.to(device)
      elif type_ == "autoencoder":
          input = imgs.to(device)
          output = clean_imgs.to(device)

      with torch.no_grad():
          pred = model(input)
          loss = criterion(pred, output)
          val_loss += loss.item()
          
          
  val_loss /= len(val_data)

  return val_loss



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



def visualize_predictions(
  type_,
  model, 
  dataloader, 
  criterion=nn.MSELoss(),
  latent_dim=2
):
  if type_ != 'encoder':
    print("Not encoder")
    return
  
  n=4
  s=2
  fig, axs = plt.subplots(n, n, figsize=(n*s, 1.3*n*s))
  fig.suptitle('Encoder Predictions')
    
  for i, (img, label1d, label2d, latent, reconstructed) in enumerate(get_outputs(type_, model, dataloader)):
    if i >= n**2:
      break
    
    if latent_dim == 1:
      # Latents are in minutes past midnight. Multiply by 12*60 to unnormalize
      unnormalized_latent = (latent * torch.tensor(12*60).to(device).float()).cpu()
      unnormalized_label1d = (label1d * torch.tensor(12*60).to(device).float()).cpu()

      pred_minute = unnormalized_latent.item()
      true_minute = unnormalized_label1d.item()

      desc= f"Pred: {pred_minute:.0f}m\nTrue: {true_minute:.0f}m"
      labels = label1d.unsqueeze(0)
      
    else:
      # Latents are in hours and minutes. Multiply by [12, 60] to unnormalize
      unnormalized_latent = (latent * torch.tensor([12, 60]).to(device).float())
      unnormalized_label2d = (label2d * torch.tensor([12, 60]).to(device).float())
      
      pred_hour = unnormalized_latent[0]
      pred_minute = unnormalized_latent[1]
      true_hour = unnormalized_label2d[0]
      true_minute = unnormalized_label2d[1]

      desc= f"Pred: {pred_hour:.0f}h{pred_minute:.0f}m\nTrue: {true_hour:.0f}h{true_minute:.0f}m"
      labels = label2d

    assert labels.shape == latent.shape
    loss = criterion(labels, latent)
    
    axs[i//n, i%n].imshow(img.squeeze().cpu(), cmap='gray') 
    axs[i//n, i%n].set_title(f"{desc}\nLoss: {loss:.6f}")
    axs[i//n, i%n].axis('off')
    
    

def visualize_latent(type_, model, latent_dim, dataloader):
  
  if type_ == 'decoder':
    print("Decoder")
    return
  
  # Get a batch of model outputs
  latents = []
  labels1d = []
  for _, label1d, _, latent, _ in get_outputs(type_, model, dataloader):
    latents.append(latent.unsqueeze(0).cpu())
    labels1d.append(label1d.unsqueeze(0).cpu())

  latents = torch.cat(latents, dim=0)
  labels1d = torch.cat(labels1d, dim=0)
  
  
  if (latent_dim <= 2):
    # Plot latent space
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(labels1d if latent_dim==1 else latents[:,1], latents[:,0], c=labels1d, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Time in minutes past midnight")
    plt.xlabel("Time in minutes past midnight" if latent_dim==1 else "Latent Dimension 2")
    plt.ylabel("Latent Dimension 1")
    plt.title("Output of encoder")
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
    
    plt.title("PCA of output of encoder")
    plt.show()