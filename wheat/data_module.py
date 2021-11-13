import ast
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms

from wheat.dataset import WheatDataset


class WheatDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load annotation data, create train/val split, and create datasets."""
        data_dir = Path(self.config['data_dir'])
        image_dir = data_dir/'train'
        # must get unique image ids from files on disk, not from dataframe,
        # because images without bounding boxes are not in the dataframe
        unique_image_ids = sorted([image_path.stem for image_path in image_dir.glob('*.jpg')])
        anno_dict = {image_id: [] for image_id in unique_image_ids}

        train_csv = data_dir/'train.csv'
        df = pd.read_csv(train_csv)
        for _, row in df.iterrows():
            bbox = np.array(ast.literal_eval(row.bbox))
            anno_dict[row.image_id].append(bbox)

        rng = np.random.default_rng()
        rng.shuffle(unique_image_ids)
        train_fraction = 0.8
        train_samples = int(train_fraction * len(unique_image_ids))
        train_ids = unique_image_ids[:train_samples]
        val_ids = unique_image_ids[train_samples:]

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = WheatDataset(
            image_dir, train_ids, anno_dict, 
            transform=transform, target_transform=target_transform,
        )
        self.val_dataset = WheatDataset(
            image_dir, val_ids, anno_dict, 
            transform=transform, target_transform=target_transform,
        )
#     image_id  width  height                         bbox   source
# 0  b6ab77fd7   1024    1024   [834.0, 222.0, 56.0, 36.0]  usask_1

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['train']['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['eval']['batch_size'])

    # def test_dataloader(self):
    #     transforms = ...
    #     return DataLoader(self.test, batch_size=64)
