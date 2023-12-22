from dataclasses import dataclass
import torch

@dataclass
class MMFNetConfig:
    epochs = 20
    loss_fn = "fringe"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decay_steps = 10
    decay_gamma = 0.99

    batch_size = 32
    lr = 0.001

    # log
    log_dir = "data/log/mmfnet"
