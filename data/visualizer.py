from __future__ import annotations

import torch
import h5py
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder


ImageFile.LOAD_TRUNCATED_IMAGES = True

# DataLoader

from torch.utils.data import DataLoader

from torchvision.utils import save_image

from datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset

import numpy as np
import random

from matplotlib import pyplot as plt

# Visualizer

# from torchvision.utils import make_grid
# from torchvision.io import read_image
# from pathlib import Path

# dog1_int = read_image(str(Path('../assets') / 'dog1.jpg'))
# dog2_int = read_image(str(Path('../assets') / 'dog2.jpg'))
# dog_list = [dog1_int, dog2_int]

# grid = make_grid(dog_list)
# show(grid)

print('check')

data = CIFAR10Dataset(img_size=64, train=False)

# i = random.randint(0, 10_000)
# print(i)
# print(data.input_size)
# data.__getitem__(i)

def inverse_transform(img):
    transform = [
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [1/0.2023, 1/0.1994, 1/0.2010]),
        transforms.Normalize(mean = [-0.4914, -0.4822, -0.4465], std = [ 1., 1., 1. ]),
        transforms.Resize([64] * 2),
    ]
    return transforms.Compose(transform)(img)
# %%
for i in range(10):
    image, label = data.__getitem__(random.randint(0, 10_000))
    # label = datapoint[0]
    # image = datapoint[1]
    # image.reshape(3,32,32).permute(1, 2, 0)
    # image_path = "output/resnet18/image" + str(i) + "_" + str(label.item()) + ".png"
    # tensor  = image.cpu()
    # save_image(tensor, image_path)
    # image = mpimg.imread(image_path)
    # mpimg.title = image_path
    untransformed_image = inverse_transform(image)
    untransformed_image = np.array(untransformed_image)

    plt.imshow(untransformed_image.transpose((1, 2, 0)))
    plt.show()

print('done')
