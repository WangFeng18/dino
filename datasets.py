import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from imagenet_lmdb import ImageNetLMDB as lmdb
from coco_lmdb import CocoLMDB as cocolmdb
from PIL import Image
from PIL import ImageFile
import random
import os
import glob
import torchvision
from torchvision.datasets.folder import default_loader
from collections import defaultdict
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CocoLMDB(cocolmdb):
    def __init__(self, root, list_file, aug):
        super(CocoLMDB, self).__init__(root, list_file, ignore_label=True)
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(CocoLMDB, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, 0, index

class ImageNetLMDB(lmdb):
    def __init__(self, root, list_file, aug):
        super(ImageNetLMDB, self).__init__(root, list_file, ignore_label=False)
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(ImageNetLMDB, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target, index

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, aug, train=True):
        super(ImageNet, self).__init__(os.path.join(root, 'train' if train else 'val'))
        self.aug = aug

    def __getitem__(self, index):
        img, target = super(ImageNet, self).__getitem__(index)
        imgs = self.aug(img)
        return imgs, target, index

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, (split in ['train', 'val']), transform, target_transform, download)
        indices = list(range(50000))
        random.seed(0)
        random.shuffle(indices)
        if split == 'train':
            self.data = self.data[indices[:45000]]
            self.targets = np.array(self.targets)[indices[:45000]]
        elif split == 'val':
            self.data = self.data[indices[45000:]]
            self.targets = np.array(self.targets)[indices[45000:]]

class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, (split in ['train', 'val']), transform, target_transform, download)
        indices = list(range(50000))
        random.seed(0)
        random.shuffle(indices)
        if split == 'train':
            self.data = self.data[indices[:44933]]
            self.targets = np.array(self.targets)[indices[:44933]]
        elif split == 'val':
            self.data = self.data[indices[44933:]]
            self.targets = np.array(self.targets)[indices[44933:]]

class ListData(data.Dataset):
    def __init__(self, root, list_file, transform):
        self.transform = transform
        self.data = self.parse(list_file)
        self.samples = [self.assemble(root, sub[0]) for sub in self.data]
        self.labels = [sub[1] for sub in self.data]
        self.transform = transform
    
    def __getitem__(self, idx):
        pth = self.samples[idx]
        lbl = self.labels[idx]
        img = default_loader(pth)
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl
    
    def __len__(self):
        return len(self.samples)

    def assemble(self, root, sample_id):
        return os.path.join(root, sample_id)

    def parse(self, listfile):
        if type(listfile) not in [tuple, list]:
            listfile = [listfile]
        
        results = []
        for lf in listfile:
            with open(lf, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pth, idx = line.split(' ')
                results.append([pth, int(idx)])
        return results

class AircraftData(ListData):
    def assemble(self, root, sample_id):
        return os.path.join(root, sample_id+'.jpg')
    def parse(self, listfile):
        with open(listfile, 'r') as f:
            lines = f.readlines()
        names = []
        labels = []
        for line in lines:
            line_parse = line.strip().split(' ')
            pth = line_parse[0]
            lbl = '_'.join(line_parse[1:])
            names.append(pth)
            labels.append(lbl)
        meta_label = sorted(list(set(labels)))
        print('Having {} classes'.format(len(meta_label)))
        label2id = {lbl:idx for idx, lbl in enumerate(meta_label)}
        labels = [label2id[lbl] for lbl in labels]

        return list(zip(names, labels))

class ListDataSun(data.Dataset):
    def __init__(self, root, list_file, transform):
        self.transform = transform
        self.data = self.parse(list_file)
        self.samples = [os.path.join(root,sub[0][1:]) for sub in self.data]
        self.labels = [sub[1] for sub in self.data]
        meta_labels = {lbl: idx for idx, lbl in enumerate(sorted(list(set(self.labels))))}
        self.labels = [meta_labels[lbl] for lbl in self.labels]
        self.transform = transform
    
    def __getitem__(self, idx):
        pth = self.samples[idx]
        lbl = self.labels[idx]
        img = default_loader(pth)
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl
    
    def __len__(self):
        return len(self.samples)

    def parse(self, listfile):
        if type(listfile) not in [tuple, list]:
            listfile = [listfile]

        results = []
        for lf in listfile:
            with open(lf, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pth = line.strip()
                label = pth.split('/')[2]
                results.append([pth, label])
        return results

class ListDataFood(ListDataSun):
    def parse(self, listfile):
        if type(listfile) not in [tuple, list]:
            listfile = [listfile]

        results = []
        for lf in listfile:
            with open(lf, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pth = line.strip()+'.jpg'
                label = pth.split('/')[0]
                results.append([pth, label])
        return results

class ListDataDTD(ListDataSun):
    def __init__(self, root, list_file, transform):
        super(ListDataDTD, self).__init__(root, list_file, transform)
        self.samples = [os.path.join(root,sub[0][1:]) for sub in self.data]

    def parse(self, listfile):
        if type(listfile) not in [tuple, list]:
            listfile = [listfile]

        results = []
        for lf in listfile:
            with open(lf, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pth = line.strip()
                label = pth.split('/')[0]
                results.append([pth, label])
        return results

class Flowers102Data(data.Dataset):
    def __init__(self, root, label, setid, split='train', transform=None):
        import scipy.io
        label_mat = scipy.io.loadmat(label)
        setid_mat = scipy.io.loadmat(setid)
        self.split_ids = setid_mat[{'train':'trnid', 'val':'valid', 'test':'tstid'}[split]][0]
        self.labels = label_mat['labels'][0]
        self.transform = transform
        self.root = root

    def __getitem__(self, idx):
        current_id = self.split_ids[idx]
        img_pth = os.path.join(self.root, 'image_'+str(current_id).zfill(5)+'.jpg')
        img = default_loader(img_pth)
        if self.transform is not None:
            img = self.transform(img)
        lbl = self.labels[current_id-1] - 1
        return img, lbl

    def __len__(self):
        return len(self.split_ids)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class VOC2007_dataset(torch.utils.data.Dataset):
    def __init__(self, voc_dir, split='train', transform=None):
        # Find the image sets
        image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + split + '.txt'))
        assert len(image_sets) == 20
        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda: -np.ones(self.n_labels, dtype=np.uint8))
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                        lbl = 0
                elif lbl == 0:
                        lbl = 255
                images[os.path.join(voc_dir, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        np.random.shuffle(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        #img = Image.open(self.images[i][0])
        #img = img.convert('RGB')
        #img = accimage.Image(self.images[i][0])
        img = pil_loader(self.images[i][0])
        if self.transform is not None:
                img = self.transform(img)
        return img, self.images[i][1]
