import torch.distributed as dist
import os
import torch
import logging

def setup_logging(rank):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)  # Only Rank 0 logs INFO
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler
    handler = logging.StreamHandler()
    
    # Custom log formatter that works without 'extra' dict
    formatter = logging.Formatter(f"%(asctime)s [Rank {rank}] %(message)s")
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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
