"""PyTorch Dataset class for loading wheat images and annotations"""
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    """PyTorch Dataset class for loading wheat images and annotations"""

    def __init__(self, image_dir: Path, image_ids: list, anno_dict: dict, transform=None):
        """Store image and annotation data and corresponding transforms.

        :param image_dir: path to directory containing image files
        :param image_ids: list of image IDs (image file name stems)
        :param anno_dict: dictionary with image ID as key and list of bounding
            boxes [xmin, ymin, width, height] as values
        :param transform: PyTorch transform to be applied to images
        """
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.anno_dict = {key: val for key, val in anno_dict.items() if key in image_ids}
        self.transform = transform

        if not all((image_id in self.anno_dict for image_id in self.image_ids)):
            raise RuntimeError('Not all image IDs were found in anno_dict.')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ind):
        """Return one dataset sample from given index."""
        if torch.is_tensor(ind):
            ind = ind.tolist()
        image_id = self.image_ids[ind]
        image_path = self.image_dir/(image_id + '.jpg')
        # torchvision models accept images as PIL images or numpy arrays
        image = Image.open(image_path)

        target_boxes = []
        for bbox in self.anno_dict[image_id]:
            # bbox is [xmin, ymin, width, height]
            left = bbox[0]
            top = bbox[1]
            right = bbox[0] + bbox[2]
            bottom = bbox[1] + bbox[3]
            target_boxes.append([left, top, right, bottom])

        # if no annos then boxes tensor will be wrong shape, so fix the shape
        boxes_tensor = torch.as_tensor(target_boxes, dtype=torch.float32) \
            if target_boxes else torch.zeros((0, 4), dtype=torch.float32)
        n_boxes = boxes_tensor.shape[0]
        target = {
            'boxes': boxes_tensor,
            # there's only one class, so class labels are all ones
            'labels': torch.ones(n_boxes, dtype=torch.int64),
        }

        if self.transform is not None:
            image = self.transform(image)

        return image, target
