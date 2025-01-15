# Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
from __future__ import print_function
from mpi4py import MPI
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
import time
import os
from enum import Enum

import torchvision.models as models

# 1. Import necessary packages.
import os
import socket
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


log = logging.getLogger('ResNet50')
log.setLevel(logging.DEBUG)

class MyImageFolder(datasets.ImageFolder):
    def preprocess(self, sample, target):
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            return sample, target
        def read_data(self, index):
            path, target = self.samples[index]
        return self.loader(path), target
    def __getitem__(self, index):
        sample, target = self.read_data(index)
        sample, target = self.preprocess(sample, target)
        return sample, target



# 2. Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    #local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size() 
    rank = MPI.COMM_WORLD.Get_rank()


    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)

    # MPI.COMM_WORLD.Barrier()

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)

def train_no_dataloader(train_loader, model, criterion, optimizer, epoch, device, args):
    images, target = train_loader.next()    
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

    if rank==0:
        print("start training")
    end = time.time()

    for i in range(args.steps):
        # measure data loading time
        data_time.update(time.time() - end)
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        if i==0:
            first_batch_time = time.time() - end
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0 and rank==0:
        #     progress.display(i + 1)
        if i >= args.steps and args.steps > 0:
            if rank==0:
                print('Throughput: {:.3f} images/s,'.format((args.steps-1) * args.batch_size * size / (batch_time.sum-first_batch_time)), 
                    'Batch size: {},'.format(args.batch_size),
                    'Num of GPUs: {},'.format(size),
                    'Total time: {:.3f} s,'.format(batch_time.sum),
                    'Average batch time: {:.3f} s,'.format(batch_time.avg),
                    'First batch time: {:.3f} s'.format(first_batch_time))
            return

def train(train_loader, model, criterion, optimizer, epoch, device, args):
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

    if rank==0:
        print("start training")
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        if i==0:
            first_batch_time = time.time() - end
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0 and rank==0:
        #     progress.display(i + 1)
        if i >= args.steps and args.steps > 0:
            if rank==0:
                print('Throughput: {:.3f} images/s,'.format((args.steps-1) * args.batch_size * size / (batch_time.sum-first_batch_time)), 
                    'Batch size: {},'.format(args.batch_size),
                    'Num of GPUs: {},'.format(size),
                    'Total time: {:.3f} s,'.format(batch_time.sum),
                    'Average batch time: {:.3f} s,'.format(batch_time.avg),
                    'First batch time: {:.3f} s'.format(first_batch_time))
            return


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if rank==0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('data', metavar='DIR', nargs='?', default='/eagle/datasets/ImageNet/ILSVRC/Data/CLS-LOC/',
                    help='path to dataset (default: imagenet)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--steps', default=100, type=int,
                    metavar='N', help='number of iterations to measure throughput, -1 for disable')
    parser.add_argument("--shuffle", action='store_true', help="shuffle the dataset")
    parser.add_argument("--dont_pin_memory", action='store_true')    
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--output-folder", default='outputs', type=str)
    parser.add_argument("--no-dataloader", action='store_true')
    parser.add_argument("--profile", action='store_true')
    import os
    
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    try:
        import intel_extension_for_pytorch as ipex
        import oneccl_bindings_for_pytorch        
    except:
        ipex = None

    args.xpu = not args.cpu and ipex and torch.xpu.is_available()
    args.cuda = not args.cpu and torch.cuda.is_available() and not args.xpu
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # 3.DDP: Initialize library.
    if with_ddp:
        if args.xpu:
            backend = "ccl"
        else:
            backend = "nccl"
        import datetime
        torch.distributed.init_process_group(backend = backend, init_method = 'env://', world_size = size, rank = rank, timeout = datetime.timedelta(seconds=120))

    torch.manual_seed(args.seed)

    # 4.DDP: Pin GPU to local rank.
    if args.xpu:
        local_rank = os.getenv("LOCAL_RANK", rank%torch.xpu.device_count())                
        torch.xpu.set_device(int(local_rank)) if torch.xpu.device_count() > 1 else 0
        torch.xpu.manual_seed(args.seed)

    elif args.cuda:
        local_rank = os.getenv("LOCAL_RANK", rank%torch.cuda.device_count())
        torch.cuda.set_device(int(local_rank))
    print("DDP: I am worker %s of %s. My local rank is %s" %(rank, size, local_rank))
    if args.cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    elif args.xpu:
        device = torch.device(f"xpu:{local_rank}")
    else:
        device = torch.device("cpu")
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.test_batch_size}
    pin_memory = not args.dont_pin_memory    
    if args.xpu or args.cuda:
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': args.shuffle}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    # Data loading code
    if args.dummy:
        if rank==0:
            print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = MyImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=size, rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, **val_kwargs)

    model = models.resnet50()
    if args.cuda:
        model = model.cuda()
    # 6.Wrap the model in DDP:
    if args.xpu:
        model = model.xpu()
    if with_ddp:
        model = DDP(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr*size,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    def run():
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            # train(args, model, criterion, device, train_loader, optimizer, epoch)
            # train for one epoch
            if (args.no_dataloader):
                train_no_dataloader(train_loader, model, criterion, optimizer, epoch, device, args)
            else:
                train(train_loader, model, criterion, optimizer, epoch, device, args)                
            # test(model, device, val_loader)
            scheduler.step()
        if rank==0:
            print(f"Total time: {time.time()-t0}")

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
        
        #test(model, device, val_loader)

    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities) as prof:
            run()
        prof.export_chrome_trace(
            f"{args.output_folder}/torch-trace-{rank}-of-{size}.json"
        )
    else:
        run()

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif args.xpu:
            device = torch.device(f"xpu:{local_rank}")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
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
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

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
