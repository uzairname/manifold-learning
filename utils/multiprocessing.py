import torch.distributed as dist
import os
import torch
import logging

from utils.logging import setup_logging

def process_group_setup(rank, world_size):
    """ Initialize the process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    setup_logging(rank)
    logging.info(f"Process group initialized for rank {rank}")

def process_group_cleanup():
    """ Clean up the process group."""
    dist.destroy_process_group()
    logging.info("Process group destroyed")
