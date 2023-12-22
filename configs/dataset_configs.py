from dataclasses import dataclass


@dataclass
class IFRDatasetConfig:
    train_metadata_fpath = "data/IF_train.csv"
    test_metadata_fpath = "data/IF_test.csv"
    img_size = 224