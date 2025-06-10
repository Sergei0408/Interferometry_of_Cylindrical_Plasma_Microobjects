import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.transforms import preprocess


class TrainDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.annotations = []
        self.mode = mode

        self.img_dir = os.path.join(
            config["data_path"],
            self.config['images'] if (mode == 'train') else self.config['val_images']
        )
        csv_path = os.path.join(
            config["annotation_path"],
            self.config['annotations'] if (mode == 'train') else self.config['val_annotations']
        )        
        
        self.target_cols = config["target_cols"]
        self.target_cols_means = config["target_cols_means"]
        self.target_cols_stds = config["target_cols_stds"]

        self._read_csv(csv_path)
        self._init_transforms()

    def _read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            labels_np = row[self.target_cols].values.astype('float32')
            self.annotations.append((row['filename'], labels_np))

    def _init_transforms(self):
        self.preprocess = preprocess(self.config['preprocess'])

    def __len__(self):
        return len(self.annotations)

    def load_sample(self, idx):
        image_path, label = self.annotations[idx]
        if not os.path.exists(os.path.join(self.img_dir, image_path)):
            raise ValueError(f"{os.path.join(self.img_dir, image_path)} doesn't exist")
        image = cv2.imread(os.path.join(self.img_dir, image_path))
        return image, label

    def __getitem__(self, idx):
        image, label = self.load_sample(idx)
        
        image = self.preprocess(image=image)['image']

        label = (label - self.target_cols_means) / self.target_cols_stds
        label = torch.from_numpy(label).float()

        idx = torch.as_tensor(idx).long()
        return {'image': image, 'label': label, 'idx': idx}


def collate_fn(batch):
    image = torch.stack([b['image'] for b in batch], dim=0)
    label = torch.stack([b['label'] for b in batch], dim=0)
    return image, label


def get_train_dl_ds(
        config,
        mode='train'
):
    dataset = TrainDataset(
        config, mode=mode
    )

    dataloader = DataLoader(
        dataset,
        shuffle=(mode == "train"),
        collate_fn=collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset

class TestDataset(TrainDataset):
    def __init__(self, config, mode='test'):
        config['val_images'] = config['images']
        config['val_annotations'] = config['annotations']
        super().__init__(config, mode='test')
        self.mode = mode
        self.config = config
        self.annotations = []
        self.img_dir = os.path.join(config["data_path"], self.config['images'])
        csv_path = os.path.join(config["annotation_path"], self.config['annotations'])
        self._read_csv(csv_path)
        self._init_transforms()

def test_collate_fn(batch):
    return {
        'image': torch.stack([b['image'] for b in batch], dim=0),
        'label': torch.stack([b['label'] for b in batch], dim=0),
        'idx': torch.stack([b['idx'] for b in batch], dim=0),
    }


def get_inference_dl_ds(config):
    dataset = TestDataset(config)

    dataloader = DataLoader(
        dataset, collate_fn=test_collate_fn,
        **config['dataloader']
    )
    return dataloader, dataset
