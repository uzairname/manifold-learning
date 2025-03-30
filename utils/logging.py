import logging

import numpy as np
import torch
import torch.nn as nn
import neptune

from utils.data_types import TrainRunState


def setup_logging(rank:int=None):
    logger = logging.getLogger()
    is_primary = rank == 0 or rank is None
    logger.setLevel(logging.INFO if is_primary else logging.WARNING)  # Only Rank 0 logs INFO

    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a handler
    handler = logging.StreamHandler()

    # Custom log formatter that works without 'extra' dict
    formatter = logging.Formatter(f"%(asctime)s [Rank {rank}] %(message)s" if rank is not None else f"%(asctime)s %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

