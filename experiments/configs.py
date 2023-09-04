from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size = 32
    epoch_num = 15
    lr = 1e-4

    loss_gamma = 1.5
    decay_steps = 5
    decay_gamma = 0.1

    device = "cuda"