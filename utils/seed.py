"""
Reproducibility helpers: set random seeds for Python, NumPy and PyTorch.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch for reproduciblity.
    
    Args:
        seed: Integer seed value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_device() -> torch.device:
    """
    Return the best available device: CUDA > MPS > CPU.
    
    Returns:
        torch.device instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
    
def detect_precision(device: torch.device) -> dict[str, bool]:
    """
    Detect suitable mixed-precision mode for the given device.
    
    Args:
        device: The target device.
        
    Returns:
        Dict with keys fp16 and bf16.
    """
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability()
        bf16_supported = capability >= (8, 0) # Ampere+
        return {"fp16": not bf16_supported, "bf16": bf16_supported}
        
    # MPS / CPU: no native mixed precision via Trainer
    return {"fp16": False, "bf16": False}