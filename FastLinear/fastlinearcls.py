"""
"""
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
from tensorboardX import SummaryWriter
import json

def adjust_learning_rate(lr, lr_decay_steps, optimizer, epoch, lr_decay_rate=0.1):
    """Decay the learning rate based on schedule"""
    steps = list(map(int, lr_decay_steps.split(',')))
    for milestone in steps:
        lr *= lr_decay_rate if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser("Fixed feature linear classification")
    parser.add_argument("--save_path", default="./.fixlin_ckp/", type=str)
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--normalize", default=1, type=int)

    parser.add_argument("--memorybank_path", default="./moco_v2_800ep_feat.pth", type=str)

    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--lr", default=120, type=float)
    parser.add_argument("--lr_decay_rate", default=0.1, type=float)
    parser.add_argument("--lr_decay_steps", default="60,80", type=str)
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    device = "cuda"

    args.memorybank_path = os.path.expanduser(args.memorybank_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_ordered_labels = np.load('train_ordered_labels.npy')
    val_ordered_labels = np.load('val_ordered_labels.npy')
    
    memory_bank = torch.load(args.memorybank_path)
    train_memory_bank = memory_bank['train_memory_bank'].to("cpu")
    val_memory_bank = memory_bank['val_memory_bank'].to("cpu")

    feat_dim = train_memory_bank.size(1)
    network = nn.Linear(feat_dim, 1000)
    network.weight.data.normal_(mean=0.0, std=0.01)
    network.bias.data.zero_()
    network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0,
    )
    local_runs = os.path.join(args.output_dir, 'runs')
    writer = SummaryWriter(logdir=local_runs)
    best_acc = 0.0
    for i_epoch in range(args.max_epochs):
        adjust_learning_rate(args.lr, args.lr_decay_steps, optimizer, i_epoch, args.lr_decay_rate)
        n_total_samples = train_memory_bank.size(0)
        perm_indices = [i for i in range(n_total_samples)]
        random.shuffle(perm_indices)
        network.train()

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            n_total_samples//args.batch_size,
            [batch_time, losses, top1, top5],
            prefix='Epoch {}: '.format(i_epoch))
        end = time.time()
        for iter in range(n_total_samples//args.batch_size + 1):
            train_data = train_memory_bank[perm_indices[iter*args.batch_size:(iter+1)*args.batch_size]].to(device)
            target = torch.from_numpy(train_ordered_labels[perm_indices[iter*args.batch_size:(iter+1)*args.batch_size]]).long().to(device)
            if args.normalize:
                train_data = torch.nn.functional.normalize(train_data, dim=-1)
            output = network(train_data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("loss: {:.4f}".format(loss.item()))
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), train_data.size(0))
            top1.update(acc1[0], train_data.size(0))
            top5.update(acc5[0], train_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_freq == 0:
                progress.display(iter)
        
        acc = validate(val_memory_bank, network, criterion, args, val_ordered_labels, device)
        if acc > best_acc:
            best_acc = acc
        print("Best Acc1: {:.4f}".format(best_acc))
        with open(os.path.join(args.output_dir, 'norm{}_linear_log.txt'.format(args.normalize)), 'a') as f:
            recorded_dict = {
                    "lc_epoch": i_epoch,
                    "lc_acc": acc,
                    "lc_best_acc":best_acc,
                    "lc_loss": losses.avg,
            }
            #f.write(json.dumps(recorded_dict) + "\n")
            f.write('acc:{} bacc:{}'.format(acc, best_acc) + "\n")
            for k, v in recorded_dict.items():
                if 'epoch' not in k:
                    writer.add_scalar(k, v, i_epoch)
        torch.save(network.state_dict(), os.path.join(args.save_path, 'ckpt.pth'))
        hdfs_runs = os.path.join(os.environ['ARNOLD_OUTPUT'], os.path.basename(args.output_dir))
        cmd1 = "hadoop fs -mkdir {}".format(hdfs_runs)
        cmd2 = "hadoop fs -put -f {} {}".format(local_runs, hdfs_runs)
        print(cmd1)
        os.system(cmd1)
        print(cmd2)
        os.system(cmd2)

 
    print('finish')
        


def validate(val_memory_bank, model, criterion, args, val_ordered_labels, device):
    n_total_samples = val_memory_bank.size(0)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        n_total_samples//args.batch_size + 1,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for iter in range(n_total_samples//args.batch_size+1):
            images = val_memory_bank[iter*args.batch_size:(iter+1)*args.batch_size].to(device)
            target = torch.from_numpy(val_ordered_labels[iter*args.batch_size:(iter+1)*args.batch_size]).long().to(device)

            if args.normalize:
                images = torch.nn.functional.normalize(images, dim=-1)
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iter % args.print_freq == 0:
                progress.display(iter)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
