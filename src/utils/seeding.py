"""Seeding utilities for reproducible results."""

import random
import numpy as np
import torch
from typing import Optional


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set environment variable for additional reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
