import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

class ActivationNormHook:
    """Hook to capture activation norms from network layers"""
    
    def __init__(self):
        self.activations = defaultdict(list)
    
    def __call__(self, name):
        def hook(module, input, output):
            # Get the output tensor
            if isinstance(output, tuple):
                output = output[0]
            
            # Calculate the norm of each example in the batch
            # For convolutional layers, first flatten the spatial dimensions
            if len(output.shape) == 4:  # Conv layer: [B, C, H, W]
                # Reshape to [B, C*H*W]
                batch_size = output.shape[0]
                flattened = output.view(batch_size, -1)
                norms = torch.norm(flattened, dim=1).detach().cpu().numpy()
            else:  # FC layer: [B, F]
                norms = torch.norm(output, dim=1).detach().cpu().numpy()
            
            self.activations[name].extend(norms)
            
        return hook

def register_activation_hooks(model):
    """Register hooks on all modules that have parameters"""
    hooks = []
    hook_handler = ActivationNormHook()
    
    # Register hooks for all suitable layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            hook = module.register_forward_hook(hook_handler(name))
            hooks.append(hook)
    
    return hooks, hook_handler

def plot_activation_histograms(activation_norms, figsize=(15, 10), bins=50):
    """Plot histograms of activation norms for each layer"""
    # Determine number of subplots needed
    num_layers = len(activation_norms)
    nrows = int(np.ceil(np.sqrt(num_layers)))
    ncols = int(np.ceil(num_layers / nrows))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if num_layers > 1 else [axes]
    
    # Set a nice color palette
    colors = sns.color_palette("viridis", num_layers)
    
    for i, (layer_name, norms) in enumerate(activation_norms.items()):
        ax = axes[i]
        ax.hist(norms, bins=bins, alpha=0.7, color=colors[i])
        ax.set_title(f"Layer: {layer_name}")
        ax.set_xlabel("Activation Norm")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def analyze_model_activations(model, dataloader, device='cuda', num_batches=None):
    """Analyze activation norms across specified batches of data"""
    # Register hooks
    hooks, hook_handler = register_activation_hooks(model)
    
    # Set model to eval mode
    model.eval()
    
    # Process batches
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Unpack batch (adjust according to your dataloader format)
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch
                
            # Forward pass
            model(inputs)
            
            # Stop after specified number of batches
            if num_batches is not None and i + 1 >= num_batches:
                break
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    
    return hook_handler.activations