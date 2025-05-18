from run_configs import baseline
from tasks.arithmetic.trainer import ArithmeticTrainer


if __name__ == "__main__":
  
  
  config = baseline
  
  trainer = ArithmeticTrainer(baseline)
  
  trainer.train()
  
  