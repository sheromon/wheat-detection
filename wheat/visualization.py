"""A couple of functions for visualizing the dataset in Jupyter notebooks"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


# this conversion is needed because albumentations transforms return
# images in uint8, but pytorch expects them to be floats in [0, 1]
image_float_to_int_transform = T.ConvertImageDtype(torch.uint8)

def show(imgs):
    """Display a single or a list of torch tensor images.

    from https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
