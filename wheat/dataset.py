#import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    def __init__(self, image_dir, image_ids, anno_dict, transform=None, target_transform=None):
        """PyTorch Dataset class for loading wheat images and annotations"""
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.anno_dict = {key: val for key, val in anno_dict.items() if key in image_ids}
        self.transform = transform
        self.target_transform = target_transform

        if not all([image_id in self.anno_dict for image_id in self.image_ids]):
            raise RuntimeError("Not all image IDs were found in anno_dict.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ind):
        """Return one dataset sample from given index."""
        if torch.is_tensor(ind):
            ind = ind.tolist()
        image_id = self.image_ids[ind]
        image_path = self.image_dir/(image_id + '.jpg')
        img = Image.open(image_path)
        example = {
            'image': img,
            'bbox': self.anno_dict[image_id]
        }
        if self.transform:
            example['image'] = self.transform(example['image'])
        return example
