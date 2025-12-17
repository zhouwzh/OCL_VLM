import argparse
import glob
import os, sys
import random
import shutil
import time
import warnings
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pytorch_lightning as pl

from models import TextEncoder, Multimodalmodel, SCModel_Wrapper
from utils import set_seed, mkdir, setup_logger, load_config_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Linear decoding with headcam data')
# parser.add_argument('--train_dir', metavar='DIR', help='path to train dataset')
# parser.add_argument('--test_dir', metavar='DIR', help='path to test dataset')
parser.add_argument('--data_dir',metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1024), this is the total batch size of all GPUs on the current node '
                         'when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--num_classes', default=22, type=int, help='number of classes in downstream classification task')
parser.add_argument('--subset', default=1.0, type=float, choices=[1.0, 0.1, 0.01],
                    help="proportion of training data to use for linear probe")
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--checkpoint', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--exp_name', default='None')


def set_parameter_requires_grad(model, feature_extracting=True):
    '''Helper function for setting body to non-trainable'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def load_split_train_test(data_dir, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    exclude_classes = ['carseat','couch','greenery','plushanimal']

    if args.slot:
        dataset = datasets.ImageFolder(
            data_dir,
            transform=transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(), 
                normalize
            ])
        )
    else:
        dataset = datasets.ImageFolder(
            data_dir,
            transform=transforms.Compose([transforms.ToTensor(), normalize])
        )

    keep_samples = [(path,label) for path, label in dataset.samples
                    if dataset.classes[label] not in exclude_classes]   
    new_classes = [c for c in dataset.classes if c not in exclude_classes]  
    new_class_to_idx = {cls_name: i for i, cls_name in enumerate(new_classes)} # cls_name -> new idx
    new_samples = [(path, new_class_to_idx[dataset.classes[label]])
                   for path, label in keep_samples]
    # import pdb; pdb.set_trace()
    
    dataset.samples = new_samples
    dataset.targets = [label for _,label in new_samples]
    dataset.classes = new_classes
    dataset.class_to_idx = new_class_to_idx

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers, 
        pin_memory=True
    )
    
    num_train = len(train_loader)
    num_test = len(test_loader)
    print('Total train data size is', num_train * args.batch_size)
    print('Total test data size is', num_test * args.batch_size)

    return train_loader, test_loader


class linear_probe(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model
        self.fc = nn.Identity()

    def forward(self, x):
        x = self.fc(self.backbone(x))
        return x
    
def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    num_classes = args.num_classes

    # set random seed
    pl.seed_everything(args.seed)
    
    # load model
    model_config = load_config_file(args.model_config)
    text_encoder = TextEncoder(model_config.text_encoder)
    sc_args = {"config": model_config.slotcontrast.config,"continue_from": model_config.slotcontrast.continue_from,"config_overrides_file": None,"config_overrides": None,"pooling": "mean","ret_type": "feat"}
    vision_encoder = SCModel_Wrapper(sc_args)
    set_parameter_requires_grad(vision_encoder)
    vision_encoder.feat_fc = torch.nn.Linear(in_features=768, out_features=512, bias=True)  # B,K,512
    vision_encoder.slot_fc = torch.nn.Linear(in_features=64, out_features=512, bias=True)  # B,K,512
    model = Multimodalmodel(vision_encoder, text_encoder)
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict)

    model = linear_probe(model.vision_encoder).to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # Data loading code
    # savefile_name = f'/home/wz3008/steve/probe_results/self_supervised_dino_sfp_resnext50_labeled_s_linear_probe_slot_{args.slot}_seed_{args.seed}_pool_{args.slot_pooling}.tar'
    # savefile_name = f"/home/wz3008/steve/probe_results/slotcontrast_probe_{args2['pooling']}_{os.path.basename(args2['continue_from'])}.tar"
    # savefile_name = f"/home/wz3008/steve/probe_results/slotcontrast_probe_dino_s_vitb14.tar"
    savefile_name = f"/home/wz3008/steve/probe_results/{args.model}.tar"
    print(f"Save file name: {savefile_name}")

    args.data_dir = "/scratch/yy2694/data/saycam_labeled/"
    train_loader, test_loader = load_split_train_test(args.data_dir, args)
    acc1_list = []
    val_acc1_list = []

    if args.dev:
        args.epochs = args.start_epoch + 1

    print("<=== training start ===>")
    # import pdb; pdb.set_trace()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch, args)
        acc1_list.append(acc1)

        if not args.dev and (epoch ==15 or epoch == 25 or epoch == 50 or epoch == 75):
            val_acc1, preds, target, images = validate(test_loader, model, args)
            val_acc1_list.append(val_acc1)

            
            torch.save({'acc1_list': acc1_list,
                        'val_acc1_list': val_acc1_list,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'preds': preds,
                        'target': target,
                        'images': images
                        }, savefile_name)

    # validate at end of epoch
    val_acc1, preds, target, images = validate(test_loader, model, args)
    val_acc1_list.append(val_acc1)

    # if not args.dev:
    torch.save({'acc1_list': acc1_list,
                'val_acc1_list': val_acc1_list,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'preds': preds,
                'target': target,
                'images': images
                }, savefile_name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)   #[64,3,224,224]
        target = target.to(device)   # [64]

        # compute output
        if args.sc or args.model=='sc':
            inputs = {
                'video': images.unsqueeze(1),
                'batch_padding_mask': torch.tensor(False).repeat(32).to(device)
            }
            output = model(inputs)
        else:
            # import pdb; pdb.set_trace()
            output = model(images) # [64,22]

        loss = criterion(output, target)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.dev:
            print("DEV mode, break after 1 epoch")
            break

    return top1.avg.cpu().numpy()


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.sc or args.model=='sc':
                inputs = {
                    'video': images.unsqueeze(1),
                    'batch_padding_mask': torch.tensor(False).repeat(32).to(device)
                }
                output = model(inputs)
            else:
                output = model(images) # [64,22]

            # loss = criterion(output, target)

            # compute output
            # output = model(images)   # [64,22]

            preds = np.argmax(output.cpu().numpy(), axis=1) # [64]

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            top1.update(acc1[0].cpu().numpy()[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.dev:
                print("DEV mode, break after 1 epoch")
                break

        print('* Acc@1 {top1.avg:.3f} '.format(top1=top1))

    return top1.avg, preds, target.cpu().numpy(), images.cpu().numpy()


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
    # Displays progress of training or validation
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

        _, pred = output.topk(maxk, 1, True, True)  # B,k
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))  # [64] -> [1, 64] -> [2, 64]

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # inplace operation
        return res


if __name__ == '__main__':
    main()
