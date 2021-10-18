import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

class MultiHead(nn.Module):
    def __init__(self, dims, num_classes=1000):
        super(MultiHead, self).__init__()
        self.linear = nn.ModuleList()
        for d in dims:
            self.linear.append(
                nn.Sequential(
                    Reduce("b c h w -> b c", reduction="mean"),
                    nn.Linear(d, num_classes),
                )
            )
        self.init()    

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m)
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()
    
    def forward(self, inputs):
        return [net(input) for net, input in zip(self.linear, inputs)]


def resnet50_multihead_classifier(num_classes=1000):
    dims = [64, 256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]
    return MultiHead(dims, num_classes=num_classes)

