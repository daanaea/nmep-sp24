from __future__ import annotations

import h5py
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# DataLoader

from torch.utils.data import DataLoader

from data.datasets import CIFAR10Dataset, MediumImagenetHDF5Dataset

# Visualizer

from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path

dog1_int = read_image(str(Path('../data/medium-imagenet/data')))
dog_list = [dog1_int]

grid = make_grid(dog_list)
show(grid)
