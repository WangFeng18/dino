import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from imagenet_lmdb import ImageNetLMDB as lmdb
from PIL import Image
from PIL import ImageFile
import random
import os
import glob
import torchvision
from torchvision.datasets.folder import default_loader
from collections import defaultdict
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageNetLMDB(lmdb):
    def __init__(self, root, list_file, aug):
        super(ImageNetLMDB, self).__init__(root, list_file, ignore_label=False)
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(ImageNetLMDB, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target
