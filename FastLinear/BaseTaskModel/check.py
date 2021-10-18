import random
import torch
import numpy as np
from resnet_timm import resnet50


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    net = resnet50(num_classes=1000)
    dummy_input = torch.rand(3,3,224,224)
    b = net(dummy_input, type="block")
    for block in b:
        print(block.size())
    
