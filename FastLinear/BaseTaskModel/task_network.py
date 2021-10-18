# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torchvision.models import resnet50
from BaseTaskModel.resnet_mm import resnet50 as resnet50_mm
from BaseTaskModel.resnet_swav import resnet50 as resnet50_swav
from timm.models import resnet50 as resnet50_selfboost
import vision_transformer as vit

def get_moco_network(pretrained_path, feature_layer='lowdim'):
    network = resnet50(num_classes=128)
    dim_mlp = network.fc.weight.shape[1]
    network.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                network.fc)
    state_dict = torch.load(pretrained_path)['state_dict']
    new_state_dict = {k[len('module.encoder_q.'):]:v for k, v in state_dict.items()}
    network.load_state_dict(new_state_dict)
    if feature_layer != 'lowdim':
        network.fc = nn.Identity()
    return network

def get_swav_network(pretrained_path, feature_layer='lowdim'):
    network = resnet50_swav(hidden_mlp=2048, output_dim=128, nmb_prototypes=3000)
    state_dict = torch.load(pretrained_path)
    new_state_dict = {k[len('module.'):]:v for k, v in state_dict.items()}
    network.load_state_dict(new_state_dict)
    network.prototypes = None
    if feature_layer != 'lowdim':
        network.projection_head = None
    return network

def get_selfboost_network(pretrained_path, feature_layer='lowdim'):
    backbone = resnet50_selfboost(num_classes=0, drop_rate=0.0, drop_path_rate=0.2)
    projection_head = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
    )
    network = nn.Sequential(
        backbone,
        projection_head,
    )

    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))['model']
    network.load_state_dict(state_dict)
    if feature_layer != 'lowdim':
        network[1] = nn.Identity()
    return network

def get_minmaxent_network(backbone, pretrained_path, feature_layer='lowdim'):
    if backbone == 'resnet50':
        network = resnet50_mm(num_classes=0)
    elif backbone == 'vit_b':
        network = vit.vit_base()
    elif backbone == 'vit_s':
        network = vit.vit_small()
    #projection_head = nn.Sequential(
    #    nn.Linear(2048, 2048),
    #    nn.BatchNorm1d(2048),
    #    nn.ReLU(),
    #    nn.Linear(2048, 2048),
    #    nn.BatchNorm1d(2048),
    #    nn.ReLU(),
    #    #nn.Linear(2048, dim),
    #    #nn.BatchNorm1d(dim),
    #)
    
    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))['model']
    tsd = {}
    for k, v in state_dict.items():
        if k.startswith('backbone'):
            tsd[k[len('backbone.'):]] = v

    rt = network.load_state_dict(tsd, strict=False)
    print(rt)
    return network

def get_dino_network(backbone, pretrained_path, feature_layer='lowdim'):
    if backbone == 'vit_b':
        network = vit.vit_base()
    elif backbone == 'vit_s':
        network = vit.vit_small()

    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))['student']
    tsd = {}
    for k, v in state_dict.items():
        if k.startswith('module.backbone'):
            tsd[k[len('module.backbone.'):]] = v

    rt = network.load_state_dict(tsd, strict=False)
    print(rt)
    return network


def get_simclr_network(backbone, pretrained_path, feature_layer='lowdim'):
    if backbone == 'resnet50':
        network = resnet50_mm(num_classes=0)
    elif backbone == 'vit_b':
        network = vit.vit_base()
    elif backbone == 'vit_s':
        network = vit.vit_small()
    #projection_head = nn.Sequential(
    #    nn.Linear(2048, 2048),
    #    nn.BatchNorm1d(2048),
    #    nn.ReLU(),
    #    nn.Linear(2048, 2048),
    #    nn.BatchNorm1d(2048),
    #    nn.ReLU(),
    #    #nn.Linear(2048, dim),
    #    #nn.BatchNorm1d(dim),
    #)
    
    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
    tsd = {}
    for k, v in state_dict.items():
        if k.startswith('_feature_blocks'):
            tsd[k[len('_feature_blocks.'):]] = v

    rt = network.load_state_dict(tsd, strict=False)
    print(rt)
    return network

def get_sup_network(backbone, pretrained_path, feature_layer='lowdim'):
    if backbone == 'resnet50':
        network = resnet50_mm(num_classes=0)
    elif backbone == 'vit_b':
        network = vit.vit_base()
    elif backbone == 'vit_s':
        network = vit.vit_small()
    
    state_dict = torch.load(pretrained_path, map_location=torch.device("cpu"))
    rt = network.load_state_dict(state_dict, strict=False)
    print(rt)
    return network
