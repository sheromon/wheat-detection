"""PyTorch Lightning DataModule for working with the Kaggle wheat dataset"""
import ast
import os
from pathlib import Path
from typing import Optional

import albumentations as A
import albumentations.pytorch
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from wheat.dataset import WheatDataset


class WheatDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for working with the Kaggle wheat dataset"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        pass

    def get_transforms(self):
        """Construct and return transforms for train and val/test/predict stages.

        Training trainsforms are composed albumentations transforms with
        probabilities controlled in the config. Val/test/predict transform is
        just converting the images to torch tensors.

        :return: tuple (train_transform, val_transform)
        """
        train_trans_config = self.config['train']['transforms']
        bbox_params = A.BboxParams(
            format='pascal_voc', min_visibility=0.5, label_fields=['labels'])
        train_transform = A.Compose([
            A.HorizontalFlip(p=train_trans_config['horizontal_flip_prob']),
            A.VerticalFlip(p=train_trans_config['vertical_flip_prob']),
            A.RandomRotate90(p=train_trans_config['random_rotate90_prob']),
            A.ColorJitter(p=train_trans_config['color_jitter_prob']),
            A.Blur(p=train_trans_config['blur_prob']),
            albumentations.pytorch.ToTensorV2(),
        ], bbox_params=bbox_params)
        val_transform = A.Compose([
            albumentations.pytorch.ToTensorV2(),
        ], bbox_params=bbox_params)
        return train_transform, val_transform

    def setup(self, stage: Optional[str] = None):
        """Load annotation data, create train/val split, and create datasets.

        :param stage: one of 'fit', 'validate', 'test', or 'predict'
        """
        data_dir = Path(self.config['data_dir'])
        train_or_test = 'train' if stage in ('fit', 'validate') else 'test'
        image_dir = data_dir/train_or_test
        # must get unique image ids from files on disk, not from train.csv,
        # because images without bounding boxes are not in train.csv
        unique_image_ids = sorted([image_path.stem for image_path in image_dir.glob('*.jpg')])
        anno_dict = {image_id: [] for image_id in unique_image_ids}

        train_transform, val_transform = self.get_transforms()
        if stage in ['fit', 'validate']:
            train_csv = data_dir/'train.csv'
            df = pd.read_csv(train_csv)
            for _, row in df.iterrows():
                bbox = np.array(ast.literal_eval(row.bbox))
                anno_dict[row.image_id].append(bbox)

            rng = np.random.default_rng(seed=self.config['numpy_seed'])
            rng.shuffle(unique_image_ids)
            train_fraction = 0.8
            train_samples = int(train_fraction * len(unique_image_ids))
            train_ids = unique_image_ids[:train_samples]
            val_ids = unique_image_ids[train_samples:]
            # if we want to run overfitting, use the val set as the train set
            if self.config['train']['overfit']:
                train_ids = val_ids

            if stage == 'fit':
                self.train_dataset = WheatDataset(
                    image_dir, train_ids, anno_dict, transform=train_transform)
            self.val_dataset = WheatDataset(
                image_dir, val_ids, anno_dict, transform=val_transform)

        if stage == 'predict':
            self.test_dataset = WheatDataset(
                image_dir, unique_image_ids, anno_dict, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['eval']['batch_size'],
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
        )

def collate_fn(batch):
    """From https://github.com/pytorch/vision/blob/main/references/detection/utils.py"""
    return tuple(zip(*batch))
