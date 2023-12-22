import random
import numpy as np
import torch


def set_random_seed(seed):
    """
    Set the random number seed for Python, NumPy, and PyTorch.

    Parameters:
    seed (int): The seed value to be set for all random number generators.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch's random number generator
    torch.manual_seed(seed)

    # Additionally for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
