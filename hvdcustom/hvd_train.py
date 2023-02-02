import argparse

from hvdcustom.hvd_config import parse_encoder
from subgraph_matching.test import validation
from common import utils
from common import models
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import os


import tqdm

import horovod
import horovod.torch as hvd

import torchvision

import torch.multiprocessing as mp
import time, sys
import pandas as pd
import pickle
import random
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader, DynamicBatchSampler

from filelock import FileLock
# from torch_geometric.sampler import B

from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch

# class CustomDataset(torch.utils.data.Dataset): 
#   def __init__(self):
#   데이터셋의 전처리를 해주는 부분

#   def __len__(self):
#   데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분

#   def __getitem__(self, idx): 
#   데이터셋에서 특정 1개의 샘플을 가져오는 함수

def build_model(args):
    if args.method_type == "gnn":
        model = models.GnnEmbedder(1, args.hidden_dim, args)
    # elif args.method_type == "mlp":
    #     model = models.BaselineMLP(1, args.hidden_dim, args)
    model.to(utils.get_device())
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path,
                                         map_location=utils.get_device()))
    return model

def make_data_source(args):
    if args.dataset == "scene":
        # data_source = SceneDataSource("scene")
        data_source = SceneDataSet("scene")
    return data_source

def load_dataset(name):
    task = "graph"
    if name == "scene":
        # dataset = []
        dataset = [[], [], []]
        with open("common/data/v3_x1003/0_64.pickle", "rb") as fr:
            tmp = pickle.load(fr)
            #todo dataloader Sampler 를 적용하기 위해 Tuple 형태의 데이터가 필요함 해당 형식으로 변경 -- 23.01.31
            # for i in range(0, len(tmp[0])):
                # dataset.append((tmp[0][i], tmp[1][i], tmp[2][i]))
            for i in range(0, len(tmp[0]), 64):
                dataset[0].append(tmp[0][i])
                dataset[1].append(tmp[1][i])
                dataset[2].append(tmp[2][i])
        return dataset
    
    if task == "graph":
        train_len = int(0.8 * len(dataset))
        train, test = [], []
        dataset = list(dataset)
        random.shuffle(dataset)
        has_name = hasattr(dataset[0], "name")
        for i, graph in tqdm(enumerate(dataset)):
            if not type(graph) == nx.Graph:
                if has_name:
                    del graph.name
                x_f = graph.x
                graph = pyg_utils.to_networkx(graph).to_undirected()
                if name != "scene":
                    for j in range(3):
                        nx.set_node_attributes(
                            graph, {idx: f.item() for idx, f in enumerate(x_f[:, j])}, "f"+str(j))
            if i < train_len:
                train.append(graph)
            else:
                test.append(graph)
    return train, test, task

class DataSource:
    def gen_batch(batch_target, batch_neg_target, batch_neg_query, train):
        raise NotImplementedError

# https://wikidocs.net/57165
# class SceneDataSource(DataSource):
class SceneDataSet(DataSource):
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)
    # def __len__(self) : 
    #     return len(self.dataset)
    # 총 데이터의 개수를 리턴
    def __len__(self): 
        print(len(self.dataset))
        print(len(self.dataset[0]))
        print(self.dataset[0])
        sys.exit()

        
        return len(self.dataset[0])


    # # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    # def __getitem__(self, idx): 
    #     x = torch.FloatTensor(self.x_data[idx])
    #     y = torch.FloatTensor(self.y_data[idx])
    #     return x, y

    def gen_data_loaders(self, batch_sizes, train=True):
        n = batch_sizes
        l1, l2, l3 = [], [], []
        for i in range(len(self.dataset[0])//batch_sizes):
            l1.append(self.dataset[0][i:i+batch_sizes])
            l2.append(self.dataset[1][i:i+batch_sizes])
            l3.append(self.dataset[2][i:i+batch_sizes])

        return [[a, b, c] for a, b, c in zip(l1, l2, l3)]

    def gen_batch(self, datas, train):
        pos_d = datas[2]
        pos_a =batch_nx_graphs(datas[0])
        for i in range(len(datas[1])):
            if len(datas[1][i].edges()) == 0:
                datas[1][i] = datas[0][i]
                datas[2][i] = 0.0
        pos_b =batch_nx_graphs(datas[1])
        return pos_a, pos_b, pos_d

# class TestDataSet(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return {"input":torch.tensor([index, 2*index, 3*index], dtype=torch.float32), 
                "label": torch.tensor(index, dtype=torch.float32)}



                

def batch_nx_graphs(graphs, anchors=None):
    # motifs_batch = [pyg_utils.from_networkx(
    #    nx.convert_node_labels_to_integers(graph)) for graph in graphs]
    #loader = DataLoader(motifs_batch, batch_size=len(motifs_batch))
    #for b in loader: batch = b
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    for g in graphs:
        for v in g.nodes:
            g.nodes[v]["node_feature"] = torch.tensor([
                g.nodes[v]["f0"]])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = batch.to(utils.get_device())
    # print(batch)
    return batch


def train(args, model, dataset, data_source):
    """Train the embedding model.
    args: Commandline arguments
    dataset: Dataset of batch size
    data_source: DataSource class
    """
    scheduler, opt = utils.build_optimizer(args, model.parameters())
    if args.method_type == "gnn":
        clf_opt = optim.Adam(model.clf_model.parameters(), lr=args.lr)

    model.train()   # dorpout 및 batchnomalization 활성화
    model.zero_grad()   # 학습하기위한 Grad 저장할 변수 초기화
    pos_a, pos_b, pos_label = data_source.gen_batch(
        dataset, True)

    emb_as, emb_bs = model.emb_model(pos_a), model.emb_model(pos_b)
    labels = torch.tensor(pos_label).to(utils.get_device())

    intersect_embs = None
    pred = model(emb_as, emb_bs)
    loss = model.criterion(pred, intersect_embs, labels)
    print("loss", loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if scheduler:
        scheduler.step()

    # 분류하기 위해서
    if args.method_type == "gnn":
        with torch.no_grad():
            pred = model.predict(pred)  # 해당 부분은 학습에 반영하지 않겠다
        model.clf_model.zero_grad()
        pred = model.clf_model(pred.unsqueeze(1)).view(-1)
        criterion = nn.MSELoss()
        clf_loss = criterion(pred.float(), labels.float())
        clf_loss.backward()
        clf_opt.step()

    # acc = torch.mean((pred == labels).type(torch.float))

    return pred, labels, loss.item()

def train_loop(args):
    # device_cache = torch.device("cuda") if torch.cuda.is_available() \
    #         else torch.device("cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if device_cache == "cuda" else {}
    # # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # # issues with Infiniband implementations that are not fork-safe
    # if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
    #         mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
    #     kwargs['multiprocessing_context'] = 'forkserver'
    
    model = build_model(args)
    model.cuda()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = 1e-4)
        # lr=lr)
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters())
    criterion = torch.nn.CrossEntropyLoss()
    hvd.broadcast_parameters(
        model.state_dict(),
        root_rank=0)
    
    val = []
    batch_n = 10
    epoch = 100

    data_source = make_data_source(args)
    data_source.len()

    loaders = data_source.gen_data_loaders(
        args.batch_size, train=False)
    
    for e in range(epoch):
        for dataset in loaders:
            if args.test:
                    mae = validation(args, model, dataset, data_source)
                    val.append(mae)
            else:
                pred, labels, loss = train(
                    args, model, dataset, data_source)

                if batch_n % 10 == 0:
                    print(pred, pred.shape, sep='\n')
                    print(labels, labels.shape, sep='\n')
                    print("epoch :", e, "batch :", batch_n,
                            "loss :", loss)

                batch_n += 1
        
            if not args.test & e% 10 == 0 :
                print("Saving {}".format(args.model_path[:-5]+"_e"+str(e+1)+".pt"))
                torch.save(model.state_dict(),
                        args.model_path[:-5]+"_e"+str(e+1)+".pt")
            else:
                print(len(loaders))
                print(sum(val)/len(loaders))


'''
    CBIR-SG에 HVD 이용해 학습 적용
    데이터 로드 부분 및 병렬 학습 부분 적용
'''
# def main(force_test=False, args):
def main(args):
    def train_mixed_precision(epoch, scaler):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        # train_sampler.set_epoch(epoch)
        print("111")
        for batch_idx, (data, target) in enumerate(train_loader):
            
            print("222")
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
                    epoch, batch_idx * len(data), 100,
                    # epoch, batch_idx * len(data), len(train_sampler),
                            100. * batch_idx / len(train_loader), loss.item(), scaler.get_scale()))

    def train_epoch(epoch):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        # train_sampler.set_epoch(epoch)
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
                    epoch, batch_idx * len(data), 100,
                    # epoch, batch_idx * len(data), len(train_sampler),
                            100. * batch_idx / len(train_loader), loss.item()))

    def metric_average(val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def test():
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        # for data, target in test_loader:
        for batch_idx, (data, target) in enumerate(train_loader):
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


    def train_loop(args):
        if not os.path.exists(os.path.dirname(args.model_path)):
            os.makedirs(os.path.dirname(args.model_path))
        if not os.path.exists("plots/"):
            os.makedirs("plots/")

        model = build_model(args)

        data_source = make_data_source(args)
        loaders = data_source.gen_data_loaders(
            args.batch_size, train=False)

        val = []
        batch_n = 0
        epoch = 1
        for e in range(epoch):
            for dataset in loaders:
                if args.test:
                    mae = validation(args, model, dataset, data_source)
                    val.append(mae)
                else:
                    pred, labels, loss = train(
                        args, model, dataset, data_source)

                    if batch_n % 100 == 0:
                        print(pred, pred.shape, sep='\n')
                        print(labels, labels.shape, sep='\n')
                        print("epoch :", e, "batch :", batch_n,
                            "loss :", loss)
                    batch_n += 1

            if not args.test:
                print("Saving {}".format(args.model_path[:-5]+"_e"+str(e+1)+".pt"))
                torch.save(model.state_dict(),
                        args.model_path[:-5]+"_e"+str(e+1)+".pt")
            else:
                print(len(loaders))
                print(sum(val)/len(loaders))
        

    # Horovod: initialize library.
    hvd.init() #horovod 초기화   
    
    torch.cuda.set_device(hvd.local_rank())    
    print("local_rank : ",hvd.local_rank())

    device_cache = torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if device_cache == "cuda" else {}
    kwargs = {'num_workers': 1, 'pin_memory': True} if device_cache == "cuda" else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # data_dir = args.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = make_data_source(args)
        # loaders = train_dataset.gen_data_loaders(args.batch_size, train=False)

    # Horovod: use DistributedSampler to partition the training data. / train data로 분할
    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = torch.utils.data.DataLoader(
        # train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
        train_dataset, batch_size=args.batch_size, **kwargs)
    print(train_loader)

    #model 생성 후 hvd에서 gradient 합침
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
    
    timeList = []
    accList = []
    lossList = []
    for epoch in range(1, args.epochs + 1):
        if args.use_mixed_precision: 
            train_mixed_precision(epoch, scaler)
            print("this 1")
        else:
            train_loop(args)
        # Keep test in full precision since computation is relatively light.


    # testAcc, test_loss = test()

    # end = time.time()
    # timeList.append(end-start)
    # accList.append(testAcc)
    # lossList.append(test_loss)

    # df = pd.DataFrame({"time" : timeList, "acc" : accList, "loss" : lossList})
    # df.to_csv("result/hvd_gloo_epoch10.csv")





if __name__ == '__main__':

    # timeList = []
    # accList = []
    # lossList = []
    for mm in range(10) : 
        start = time.time()

        parser = argparse.ArgumentParser(description='Embedding arguments')
        
        utils.parse_optimizer(parser)
        parse_encoder(parser)
        args = parser.parse_args()

        if args.num_proc:
            # run training through horovod.run
            # python pytorch_mnist.py --epochs 2 --num-proc 2
            # 32.9776sec
            print('Running training through horovod.run')
            horovod.run(main,
                        args=(args, ),
                        np=args.num_proc,
                        hosts=args.hosts,
                        use_gloo=args.communication == 'gloo',
                        use_mpi=args.communication == 'mpi')

            end = time.time()
            # timeList.append(end-start)
            # # accList.append(testAcc)
            # # lossList.append(test_loss)

        else:
            # horovodrun -np 2 -H localhost:2 python pytorch_mnist.py --epochs 2
            # 36.5366sec
            # this is running via horovodrun
            main(args)

            # end = time.time()
            # timeList.append(end-start)
            print("time2 : ", end-start)

        # df = pd.DataFrame({"time" : timeList, "acc" : accList, "loss" : lossList})
        # df.to_csv("result/hvd_gloo_epoch10.csv")
