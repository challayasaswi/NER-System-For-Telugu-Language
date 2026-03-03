# utils.py
import random
import numpy as np
import torch

def fix_random_seed(seed: int = 42):
    """Fix random seed across libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed fixed to {seed}")

def set_device():
    """Return the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_banner(title: str):
    """Print a nice banner for logs."""
    print("=" * 50)
    print(title.center(50))
    print("=" * 50)
