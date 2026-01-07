"""
Utility functions for the risk forecasting project.
"""

import random
import numpy as np
import torch
from typing import Optional
import logging


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: Optional[str] = None) -> str:
    """
    Get the device to use for PyTorch operations.
    
    Args:
        device: Requested device ('cuda', 'cpu', or None for auto)
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    return device


def format_time(seconds: float) -> str:
    """
    Format seconds into a readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
