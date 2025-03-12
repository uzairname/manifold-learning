import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from utils.config import MODELS_DIR
import typing
from torch import nn
from functools import partial
from sklearn.decomposition import PCA
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_script(
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


def load_model_state_dict(
  model_class: nn.Module,
  img_size=128,
  latent_dim=2,
  model_params:dict=None,
  postfix='',
  name='model',
  checkpoint=None
):
  """
  Loads a clock model by state dict and architecture
  """
  
  model_dir = f"{latent_dim}-i{img_size}-{postfix}"
  
  if checkpoint is None:
    model_path = os.path.join(MODELS_DIR, name, model_dir, f"final.pt")
  else:
    model_path = os.path.join(MODELS_DIR, name, model_dir, f"{checkpoint}.pt")
  
  if not model_params:
    # load model args from json
    with open(os.path.join(MODELS_DIR, name, model_dir, 'model_params.json'), 'r') as f:
      model_params = json.load(f).get('model_params', {})
  
  model = model_class(**model_params).to(device)
  state_dict = torch.load(model_path, map_location=device)
  
  if 'model' in state_dict:
    state_dict = state_dict['model']
  
  model.load_state_dict(state_dict)
  model.eval()
  
  return model



def print_model_parameters(cls: nn.Module, details=False):
  
    model = cls().to(device)
    model.eval()

    print(f"{'Layer':<40}{'Param Count':>15}")
    print("-" * 60)
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            if details:
              print(f"{name:<40}{param_count:>15}")
    
    print("-" * 60)
    print(f"{'Total Trainable Parameters':<40}{total_params:>15,}")


def get_outputs(type_: typing.Literal['encoder', 'autoencoder', 'decoder'], model, dataloader, latent_dim=2, limit=None):
  """
  yields:
    - image, label1d, label2d, latent, reconstructed
  """
  total_generated = 0
  with torch.no_grad():
    for noisy_imgs, clean_imgs, label2d, label1d in dataloader:
      noisy_imgs = noisy_imgs.to(device)
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
        reconstructeds = model.forward(label1d.unsqueeze(1) if latent_dim == 1 else label2d)
      
      
      for i in range(images.size(0)):
        latent = latents[i] if latents is not None else None
        reconstructed = reconstructeds[i] if reconstructeds is not None else None
        yield noisy_imgs[i], images[i], label1d[i], label2d[i], latent, reconstructed
        total_generated += 1
        if limit is not None and total_generated >= limit:
          return



def map_inputs(type_: typing.Literal['encoder', 'autoencoder', 'decoder'], dataloader, forward_fn, latent_dim=2):
  """
  Given a forward function that takes an input according to the type and returns an output of batches,
  this function maps the inputs to the outputs.
  """
  
  with torch.no_grad():
    for _, clean_imgs, label2d, label1d in dataloader:
      images = clean_imgs.to(device)
      label1d = label1d.to(device)
      label2d = label2d.to(device)
      
      outs = None
      if type_ == 'encoder':
        outs = forward_fn(images)
      
      elif type_ == 'autoencoder':
        outs = forward_fn(images)

      elif type_ == 'decoder':
        outs = forward_fn(label1d.unsqueeze(1) if latent_dim == 1 else label2d)
        
      for i in range(images.size(0)):
        yield images[i], label1d[i], label2d[i], outs[i]
  





def show_data(dataloader: DataLoader, device=device):
  
  # visualize 16 images
  fig, axs = plt.subplots(8, 8, figsize=(10, 12))
  # fig.tight_layout()
  # imgs, clean_imgs, labels2d, label1d = next(iter(dataloader.dataset))
  for i, (imgs, clean_imgs, labels2d, label1d) in enumerate(iter(dataloader.dataset)):
      if i >= 64:
        break
      # print(labels2d)
      img = imgs.to(device)
      label2d = labels2d.to(device)

      label2d_unnormalized = (label2d * torch.tensor([12, 60]).to(device).float())

      hour = label2d_unnormalized[0]
      minute = label2d_unnormalized[1]

      axs[i // 8, i % 8].imshow(img.squeeze().cpu(), cmap='gray')
      axs[i // 8, i % 8].set_title(f"{hour:.0f}h{minute:.0f}m")
      axs[i // 8, i % 8].axis('off')

  plt.show()
  

def visualize_reconstruction(type_, model, dataloader, latent_dim=2):
  if type_ == 'encoder':
    print("Encoder")
    return

  n=10
  s=2
  fig, axs = plt.subplots(n, 6, figsize=(6*s, n*s))
  
  if type_ == 'decoder':
    fig.suptitle('Decoder Reconstructions')
  elif type_ == 'autoencoder':
    fig.suptitle('Autoencoder Reconstructions')

  for i, (noisy_img, img, label1d, label2d, latent, reconstructed) in enumerate(get_outputs(type_, model, dataloader, latent_dim=latent_dim)):
    if i >= 2*n:
      break
    
    loss = torch.nn.functional.smooth_l1_loss(img, reconstructed)
    
    unnormalized_label2d = (label2d * torch.tensor([12, 60]).to(device).float())
    
    if i % 2 == 0:
      axs[i//2,0].imshow(noisy_img.squeeze().cpu(), cmap='gray')
      axs[i//2,0].set_title("Noisy Image")
      axs[i//2,0].axis('off')
      
      axs[i//2,1].imshow(img.squeeze().cpu(), cmap='gray')
      axs[i//2,1].set_title(f"Original: {unnormalized_label2d[0]:.0f}h{unnormalized_label2d[1]:.0f}m")
      axs[i//2,1].axis('off')
      
      axs[i//2,2].imshow(reconstructed.squeeze().cpu(), cmap='gray')
      axs[i//2,2].set_title(f"Loss: {loss:.4f}")
      axs[i//2,2].axis('off')

    else:
      axs[i//2,3].imshow(noisy_img.squeeze().cpu(), cmap='gray')
      axs[i//2,3].set_title("Noisy Image")
      axs[i//2,3].axis('off')
      
      axs[i//2,4].imshow(img.squeeze().cpu(), cmap='gray')
      axs[i//2,4].set_title(f"Original: {unnormalized_label2d[0]:.0f}h{unnormalized_label2d[1]:.0f}m")
      axs[i//2,4].axis('off')
      
      axs[i//2,5].imshow(reconstructed.squeeze().cpu(), cmap='gray')
      axs[i//2,5].set_title(f"Loss: {loss:.4f}")
      axs[i//2,5].axis('off')
    
    

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
    
  for i, (_, img, label1d, label2d, latent, reconstructed) in enumerate(get_outputs(type_, model, dataloader)):
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
  for i, (_, _, label1d, _, latent, _) in enumerate(get_outputs(type_, model, dataloader)):
    if i >= 4000:
      break
    latents.append(latent.unsqueeze(0).cpu())
    labels1d.append(label1d.unsqueeze(0).cpu())

  
  latents = torch.cat(latents, dim=0)
  labels1d = torch.cat(labels1d, dim=0)
  
  print("plotting")
  
  
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
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels1d, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Time in minutes past midnight")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.title("PCA of output of encoder")
    plt.show()