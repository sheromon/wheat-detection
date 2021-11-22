"""PyTorch Dataset class for loading wheat images and annotations"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class WheatDataset(Dataset):
    """PyTorch Dataset class for loading wheat images and annotations"""

    def __init__(self, image_dir: Path, image_ids: list, anno_dict: dict, transform=None):
        """Store image and annotation data and corresponding transforms.

        :param image_dir: path to directory containing image files
        :param image_ids: list of image IDs (image file name stems)
        :param anno_dict: dictionary with image ID as key and list of bounding
            boxes [xmin, ymin, width, height] as values
        :param transform: albumentations PyTorch transform to be applied to
            images and bounding boxes
        """
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.anno_dict = {key: val for key, val in anno_dict.items() if key in image_ids}
        # TODO enforce that transform is an albumentations transform
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
        image = np.asarray(Image.open(image_path))

        target_boxes = []
        for bbox in self.anno_dict[image_id]:
            # input bbox is [xmin, ymin, width, height]
            left = bbox[0]
            top = bbox[1]
            right = bbox[0] + bbox[2]
            bottom = bbox[1] + bbox[3]
            # output bbox is [xmin, ymin, xmax, ymax]
            target_boxes.append([left, top, right, bottom])

        # if no annos then boxes tensor will be wrong shape, so fix the shape
        boxes_tensor = np.array(target_boxes, dtype=np.float32) \
            if target_boxes else np.zeros((0, 4), dtype=np.float32)
        n_boxes = boxes_tensor.shape[0]
        target = {
            'boxes': boxes_tensor,
            # there's only one class, so class labels are all ones
            'labels': np.ones(n_boxes, dtype=np.int64),
        }

        if self.transform is not None:
            # apply albumentations transformations
            transformed = self.transform(
                image=image,
                bboxes=target['boxes'],
                labels=target['labels'],
            )
            # albumentations returns the transformed image in uint8 format, but
            # torchvision wants it as a float, so convert it
            image_int_to_float_transform = T.ConvertImageDtype(torch.float)
            image = image_int_to_float_transform(transformed['image'])
            # the albumentations bounding box transforms only convert the images
            # to tensors, so we have to convert the boxes and labels to tensors
            target = {
                'boxes': torch.tensor(np.array(transformed['bboxes'], dtype=np.float32)) \
                    if transformed['bboxes'] else torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.tensor(np.array(transformed['labels'], dtype=np.int64)),
            }

        return image, target
