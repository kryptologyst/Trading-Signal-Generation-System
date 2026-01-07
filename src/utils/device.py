"""Device management utilities."""

import torch
from typing import Literal


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device preference ("auto", "cpu", "cuda", "mps").
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dictionary with device information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "current_device": str(get_device()),
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name()
    
    return info
