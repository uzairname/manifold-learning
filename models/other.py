import torch
import torch.nn as nn
import numpy as np

from models.decoders import ActivationType, str_to_activation
from models.encoders import ConvResidualEncoderBlock
from models.transformer import Embedding



class CNNClassifier(nn.Module):
    def __init__(self, img_size=28, n_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv = nn.Sequential(
          ConvResidualEncoderBlock(1, 32), # -> 7x7
          nn.Flatten(),
        )
        
        dummy_input = torch.randn(1, 1, img_size, img_size)
        dummy_output = self.conv(dummy_input)
        self.dim_after_flatten = dummy_output.shape[-1]
        
        self.fc = nn.Sequential(
          nn.Linear(self.dim_after_flatten, 512),
          nn.BatchNorm1d(512),
          nn.Tanh(),
          nn.Linear(512, n_classes),
          nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
      
      
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=784, n_classes=10, hidden_dim=512):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
          nn.Linear(input_dim, hidden_dim),
          nn.BatchNorm1d(hidden_dim),
          nn.Tanh(),
          nn.Linear(hidden_dim, n_classes),
          nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.fc(x)
      

      
class MLP(nn.Module):
  def __init__(self, dims=[1,1], activation: ActivationType=ActivationType.tanh):
    super().__init__()

    self.fc = nn.Sequential()

    for i in range(len(dims)-1):
      is_last = i == len(dims)-2

      self.fc.extend([
        nn.Linear(dims[i], dims[i+1]),
        nn.BatchNorm1d(dims[i+1]) if not is_last else nn.Identity(),
        nn.Tanh() if not is_last else str_to_activation(activation)(),
      ])
      
  def forward(self, x):
    x = self.fc(x)
    return x
  
  

class MLPWithEmbedding(nn.Module):
  def __init__(self, n_vocab, embed_dim, sequence_len, hidden_dims=[], activation: ActivationType=ActivationType.relu, last_token=True):
    super().__init__()
    
    self.last_token = last_token

    self.embedding = Embedding(n_vocab=n_vocab, d_model=embed_dim)
    
    self.flatten = nn.Flatten(start_dim=1)  # Shape batch_size, sequence_len * embed_dim
    
    dims = [embed_dim * sequence_len] + hidden_dims + [n_vocab * sequence_len]  # Output shape should be (batch_size, n_vocab * sequence_len)
    self.mlp = MLP(dims=dims, activation=activation)
    
    self.unflatten = nn.Unflatten(dim=1, unflattened_size=(sequence_len, n_vocab))  # Reshape back to (batch_size, sequence_len, n_vocab)
    
    self.sequence_len = sequence_len
    
    
  def forward(self, x):
    assert x.shape[1] == self.sequence_len, f"Input shape {x.shape} does not match expected sequence length {self.sequence_len}"
    x = self.embedding(x)
    x = self.flatten(x)  # Flatten the embedding output
    x = self.mlp(x)
    x = self.unflatten(x)  # Reshape back to (batch_size, sequence_len, n_vocab)
    if self.last_token:
      x = x[:,-1]
    return x
