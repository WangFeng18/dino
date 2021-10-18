# modify from https://github.com/facebookresearch/deit/blob/main/datasets.py
import os
import json
import torch
import random
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import ImageFilter
from imagenet_lmdb import ImageNetLMDB

class ImageNetInstance(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(ImageNetInstance, self).__init__()
        self.dataset = datasets.ImageFolder(root, transform)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)

class ImageNetInstanceLMDB(ImageNetLMDB):
    def __init__(self, root, list_file, transform):
        super(ImageNetInstanceLMDB, self).__init__(root, list_file, ignore_label=False)
        self.transform = transform

    def __getitem__(self, index):
        img, tgt = super(ImageNetInstanceLMDB, self).__getitem__(index)
        img = self.transform(img)
        return index, img, tgt



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = ImageNetInstance(root, transform=transform)
    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if args.self_transforms:
            print("Using Self Transforms")
            return transform

        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
