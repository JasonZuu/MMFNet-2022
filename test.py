import argparse

from mmfnet import MMFNet
from configs.train_configs import MMFNetConfig
from configs.dataset_configs import IFRDatasetConfig
from dataset import IFRDataset
from run_fn.test_fn import test_fn
from utils.random_fn import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    dataset_config = IFRDatasetConfig()
    test_dataset = IFRDataset(config=dataset_config, mode="test")

    train_config = MMFNetConfig()
    model = MMFNet()
    model.to(train_config.device)

    test_result = test_fn(model, train_config=train_config,
                          test_dataset=test_dataset)
    print(test_result)
