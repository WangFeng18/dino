import utils
import random
import torch
import torch.nn as nn
import torch.distributed as dist
EPS = 1e-5

def KL(probs1, probs2):
    kl = (probs1 * (probs1 + EPS).log() - probs1 * (probs2 + EPS).log()).sum(dim=1)
    kl = kl.mean()
    torch.distributed.all_reduce(kl)
    return kl

def HE(probs): 
    mean = probs.mean(dim=0)
    torch.distributed.all_reduce(mean)
    ent  = - (mean * (mean + utils.get_world_size() * EPS).log()).sum()
    return ent

def EH(probs):
    ent = - (probs * (probs + EPS).log()).sum(dim=1)
    mean = ent.mean()
    torch.distributed.all_reduce(mean)
    return mean
