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

from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset

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

# for i in range(number of things to visualize):
#     datapoint = dataset_train.__getitem__(random index)
#     label = datapoint[0]
#     image = datapoint[1]
#     image.reshape(3,32,32).permute(1, 2, 0)
#     image_path = "output/resnet18/image" + str(i) + "_" + str(label.item()) + ".png"
#     tensor  = image.cpu()
#     save_image(tensor, image_path)
#     image = mpimg.imread(image_path)
#     mpimg.title = image_path
#     mpimg.imshow(image)
#     mpimg.show()
