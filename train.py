import argparse

from mmfnet import MMFNet
from configs.train_configs import MMFNetConfig
from configs.dataset_configs import IFRDatasetConfig
from dataset import IFRDataset
from run_fn.train_fn import train_fn
from utils.random_fn import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    dataset_config = IFRDatasetConfig()
    train_dataset = IFRDataset(config=dataset_config, mode="train")

    train_config = MMFNetConfig()
    model = MMFNet()
    model.to(train_config.device)

    train_fn(model, train_config=train_config,
             train_dataset=train_dataset, write_log=True)
