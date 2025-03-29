import torch
import torch.nn as nn

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