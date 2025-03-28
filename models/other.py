import torch
import torch.nn as nn

from models.decoders import ActivationType, str_to_activation
from models.encoders import ConvResidualEncoderBlock


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
  def __init__(self, dims=[1,1], activation: ActivationType='tanh'):
    super().__init__()

    self.fc = nn.Sequential()

    for i in range(len(dims)-1):
      is_last = i == len(dims)-2

      self.fc.extend([
        nn.Linear(dims[i], dims[i+1]),
        nn.BatchNorm1d(dims[i+1]),
        nn.Tanh() if not is_last else str_to_activation(activation)(),
      ])
      
  def forward(self, x):
    x = self.fc(x)
    return x
  