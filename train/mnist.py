from models.classifiers import MnistClassifier
from mnist1.trainer import train_mnist_model
from mnist1.mnist import TrainRunConfig
import torch

if __name__ == "__main__":
  cls = MnistClassifier
  
  world_size = torch.cuda.device_count()
  
  train_mnist_model(
    c=TrainRunConfig(
      model_class=cls,
      n_epochs=3,
      batch_size=64,
      learning_rate=1e-4,
      weight_decay=1e-2,
      max_gpus=1
    )
  )
  