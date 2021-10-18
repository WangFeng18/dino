import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
from datasets import ImageNetInstance, ImageNetInstanceLMDB
from torchvision import transforms
import argparse
from BaseTaskModel.task_network import get_moco_network, get_swav_network, get_selfboost_network, get_minmaxent_network, get_simclr_network, get_sup_network, get_dino_network
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
import torch.distributed as dist
from lars import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.filterwarnings('ignore')

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def main():
    parser = argparse.ArgumentParser("The first stage of BoostrapSelfSup")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed parallel')
    parser.add_argument("--task", type=str, default="moco", help="the pretraining models")
    parser.add_argument("--pretrained_path", type=str, default="", help="the pretraining models")
    parser.add_argument("--save_path", type=str, default="", help="where to save the memory_bank")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--data_path", type=str, default="~/ILSVRC2012/", help="the data path")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--feat_dim", type=int, default=128, help="feat dimension")
    parser.add_argument("--feature_layer", type=str, default='lowdim', help="feature layer")
    parser.add_argument('--use-lmdb', action='store_true')
    args = parser.parse_args()
    
    pretrained_path = os.path.expanduser(args.pretrained_path)
    save_path = os.path.expanduser(args.save_path)
    data_path = os.path.expanduser(args.data_path)
    batch_size = args.batch_size
    feat_dim = args.feat_dim

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # network = ResNet(50, frozen_stages=4)
    if args.task == 'moco':
        network = get_moco_network(pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'swav':
        network = get_swav_network(pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'selfboost':
        network = get_selfboost_network(pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'minmaxent':
        network = get_minmaxent_network(args.backbone, pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'dino':
        network = get_dino_network(args.backbone, pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'simclr':
        network = get_simclr_network(args.backbone, pretrained_path, feature_layer=args.feature_layer)
    elif args.task == 'sup':
        network = get_sup_network(args.backbone, pretrained_path, feature_layer=args.feature_layer)
    else:
        raise NotImplementedError
    
    network.cuda(args.local_rank)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[args.local_rank])

    cudnn.benchmark = True


    augmentation = transforms.Compose([
            transforms.Resize(int(256*args.img_size/224), interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if args.use_lmdb:
        train_dataset = ImageNetInstanceLMDB(root=data_path, list_file='train.lmdb', transform=augmentation)
        val_dataset = ImageNetInstanceLMDB(root=data_path, list_file='val.lmdb', transform=augmentation)
    else:
        train_dataset = ImageNetInstance(root=os.path.join(data_path, 'train'), transform=augmentation)
        val_dataset = ImageNetInstance(root=os.path.join(data_path, 'val'), transform=augmentation)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False, rank=args.local_rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, rank=args.local_rank)

    n_train_points = len(train_dataset)
    n_val_points = len(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=4)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, sampler=val_sampler, pin_memory=True, num_workers=4)

    print("Initializing train memory bank: {} points.".format(n_train_points))
    train_memory_bank = torch.zeros(n_train_points, feat_dim).to("cpu").detach()

    print("Initializing val memory bank: {} points.".format(n_val_points))
    val_memory_bank = torch.zeros(n_val_points, feat_dim).to("cpu").detach()

    network.eval()
    train_sampler.set_epoch(0)
    val_sampler.set_epoch(0)
    for data in tqdm(train_dataloader):
        idx, img, _ = data
        idx = idx.cuda(args.local_rank, non_blocking=True)
        img = img.cuda(args.local_rank, non_blocking=True)
        if True: #args.backbone.startswith('resnet'):
            feature = network(img)
        else:
            feature = network.module.get_intermediate_layers(img, 4)
            feature = [x[:, 0] for x in feature]
            feature = torch.cat(feature, dim=-1)

        feature = concat_all_gather(feature.contiguous())
        idx = concat_all_gather(idx)

        with torch.no_grad():
            train_memory_bank[idx,:] = feature.detach().cpu()

    for data in tqdm(val_dataloader):
        idx, img, _ = data
        idx = idx.cuda(args.local_rank, non_blocking=True)
        img = img.cuda(args.local_rank, non_blocking=True)
        if True: #args.backbone.startswith('resnet'):
            feature = network(img)
        else:
            feature = network.module.get_intermediate_layers(img, 4)
            feature = [x[:, 0] for x in feature]
            feature = torch.cat(feature, dim=-1)

        feature = concat_all_gather(feature.contiguous())
        idx = concat_all_gather(idx)

        with torch.no_grad():
            val_memory_bank[idx,:] = feature.detach().cpu()

    if args.local_rank == 0:
        torch.save(
            {'train_memory_bank': train_memory_bank,
            'val_memory_bank':  val_memory_bank
            },
            args.save_path
        )

if __name__ == '__main__':
    main()
