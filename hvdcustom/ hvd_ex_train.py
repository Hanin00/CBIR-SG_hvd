import argparse
import os
from packaging import version

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms

import horovod
import horovod.torch as hvd

import time
import pandas as pd
import sys

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
import torch.optim as optim
import torch.nn as nn
import torch
from subgraph_matching.test import validation
from common import utils
from common import models



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=254, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-mixed-precision', action='store_true', default=False,
                    help='use mixed precision for training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--data-dir',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')

# Arguments when not run through horovodrun
parser.add_argument('--num-proc', type=int)
parser.add_argument('--hosts', help='hosts to run on in notation: hostname:slots[,host2:slots[,...]]')
parser.add_argument('--communication', help='collaborative communication to use: gloo, mpi')






def main(args):
    def train_mixed_precision(epoch, scaler):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = F.nll_loss(output, target)

            scaler.scale(loss).backward()
            # Make sure all async allreduces are done
            optimizer.synchronize()
            # In-place unscaling of all gradients before weights update
            scaler.unscale_(optimizer)
            with optimizer.skip_synchronize():
                scaler.step(optimizer)
            # Update scaler in case of overflow/underflow
            scaler.update()

            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss Scale: {}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item(), scaler.get_scale()))

    def train_epoch(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_loader), loss.item()))

    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))

        return test_loss, 100.*test_accuracy

    # Horovod: initialize library.
    hvd.init() #horovod 초기화
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.  / GPU를 로컬 rank로 고정
        torch.cuda.set_device(hvd.local_rank()) # torch 모듈에 생성되는 프로세스에 대해 device를 할당
        torch.cuda.manual_seed(args.seed)
    else:
        if args.use_mixed_precision:
            raise ValueError("Mixed precision is only supported with cuda enabled.")

    if (args.use_mixed_precision and version.parse(torch.__version__)
            < version.parse('1.6.0')):
        raise ValueError("""Mixed precision is using torch.cuda.amp.autocast(),
                            which requires torch >= 1.6.0""")

    # Horovod: limit # of CPU threads to be used per worker. / 작업자당 사용할 CPU 스레드 수 제한
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_dir = args.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))


    # print(type(train_dataset[3]))
    # print(train_dataset[:10])
    # print(len(train_dataset))
    # print(len(train_dataset[1]))
    # sys.exit()

    # Horovod: use DistributedSampler to partition the training data. / train data로 분할
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # print(len(train_dataset))
    # print((args.test_batch_size))


    test_dataset = \
        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)
    # print(len(test_dataset))
    # print((args.test_batch_size))


    model = models.GnnEmbedder(1, args.hidden_dim, args)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        # GPU Adasum allreduce를 사용하는 경우, local_size를 기준으로 학습 속도를 조정함
        # Adasum allreduce
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()


    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=args.lr * lr_scaler,
                          momentum=args.momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average,
                                         gradient_predivide_factor=args.gradient_predivide_factor)

    if args.use_mixed_precision:
        # Initialize scaler in global scale
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        if args.use_mixed_precision: 
            train_mixed_precision(epoch, scaler)
        else:
            train_epoch(epoch)
        # Keep test in full precision since computation is relatively light.
        testAcc, test_loss = test()

    end = time.time()
    timeList.append(end-start)
    accList.append(testAcc)
    lossList.append(test_loss)


if __name__ == '__main__':
    timeList = []
    accList = []
    lossList = []
    for mm in range(10) : 
        start = time.time()
        args = parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        if args.num_proc:
            # run training through horovod.run
            # python pytorch_mnist.py --epochs 2 --num-proc 2
            # 32.9776sec
            print('Running training through horovod.run')
            horovod.run(main,
                        args=(args,),
                        np=args.num_proc,
                        hosts=args.hosts,
                        use_gloo=args.communication == 'gloo',
                        use_mpi=args.communication == 'mpi')

            end = time.time()
            timeList.append(end-start)
            accList.append(testAcc)
            lossList.append(test_loss)

        else:
            # horovodrun -np 2 -H localhost:2 python pytorch_mnist.py --epochs 2
            # 36.5366sec
            # this is running via horovodrun
            main(args)

            end = time.time()
            print("time2 : ", end-start)

        df = pd.DataFrame({"time" : timeList, "acc" : accList, "loss" : lossList})
        df.to_csv("result/hvd_gloo_epoch10.csv")



# horovodrun -np 2 -H localhost:2 python pytorch_mnist.py --epochs 10
# horovodrun -np 2 -H localhost:2 python pytorch_mnist.py --epochs 10 --communication gloo
# horovodrun -np 2 -H localhost:2 python pytorch_mnist.py --epochs 10 --communication mpi