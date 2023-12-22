from torch.utils.data import Dataset
import PIL
import torch
import pandas as pd
from torchvision import transforms


class IFRDataset(Dataset):
    def __init__(self, configs, mode='train'):
        assert mode in ['train', 'test'], "mode must be 'train' or 'test'"
        self.mode = mode
        self.configs = configs
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((configs.img_size, configs.img_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        # load data
        self.img_paths, self.structs, self.labels = self._load_data()

    def _load_data(self):
        if self.mode == 'train':
            data = pd.read_csv(self.configs.train_metadata_fpath)
        else:
            data = pd.read_csv(self.configs.train_metadata_fpath)

        img_paths = data['img_path'].values
        structs = data[['frequence', 'fuzzy','fuzzy_norm', 'size', 'size_norm']].values
        labels = data['labels'].values

        return img_paths, structs, labels

    def __getitem__(self, index):
        img = PIL.Image.open(self.img_paths[index])
        img = self.transform(img)
        struct = self.structs[index]
        y = self.labels[index]
        return img, torch.tensor(struct, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.img_paths.shape[0]
