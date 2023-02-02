from subgraph_matching.hvd_config import parse_encoder
from subgraph_matching.test import validation
from common import utils
from common import models
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.manifold import TSNE
import os
import argparse

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
        data_source = SceneDataSource("scene")
    return data_source

def load_dataset(name):
    task = "graph"
    if name == "scene":
        # dataset = [[], [], []]        
        dataset = []
        with open("common/data/v3_x1003/0_64.pickle", "rb") as fr:
            tmp = pickle.load(fr)
            #todo dataloader Sampler 를 적용하기 위해 Tuple 형태의 데이터가 필요함 해당 형식으로 변경 -- 23.01.31
            for i in range(0, len(tmp[0])):
                dataset.append((tmp[0][i], tmp[1][i], tmp[2][i]))
                # print(dataset)
                # print(dataset[0])
                # print(type(dataset[0]))
                # sys.exit()
            # for i in range(0, len(tmp[0]), 64):
            #     dataset[0].append(tmp[0][i])
            #     dataset[1].append(tmp[1][i])
            #     dataset[2].append(tmp[2][i])
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

class SceneDataSource(DataSource):
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)

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
        pos_a = utils.batch_nx_graphs(datas[0])
        for i in range(len(datas[1])):
            if len(datas[1][i].edges()) == 0:
                datas[1][i] = datas[0][i]
                datas[2][i] = 0.0
        pos_b = utils.batch_nx_graphs(datas[1])
        return pos_a, pos_b, pos_d
    

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


# import torch.distributed as dist
# from . import Sampler


# class DistributedSampler(Sampler):
    """
        http://man.hubwiz.com/docset/PyTorch.docset/Contents/Resources/Documents/_modules/torch/utils/data/distributed.html
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



def train_loop(args):
    data_source = load_dataset(args.dataset)
    data_source = tuple(data_source)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        data_source,
        num_replicas=hvd.size(),
        rank=hvd.rank())
    
    train_loader = torch.utils.data.DataLoader(
        data_source,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        sampler=train_sampler)
    
    device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device_cache == "cuda" else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
        
    train_loader = torch.utils.data.DataLoader(
        data_source, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

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
    batch_n = 0
    epoch = 100
    for e in range(epoch):
        for dataset in train_loader:
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
            if e % 10 == 0 : 
                print("Saving {}".format(args.model_path[:-5]+"_e"+str(e+1)+".pt"))
                torch.save(model.state_dict(),
                        args.model_path[:-5]+"_e"+str(e+1)+".pt")
        else:
            print(len(dataset))
            print(sum(val)/len(dataset))
    
    
    
    # for epoch in range(100):
    #     acc = 0
    #     loss = 0
    #     model.train()
    #     for data, target in tqdm.tqdm(train_loader):
    #         data = data.cuda()
    #         target = target.cuda()
    
    #         optimizer.zero_grad()
    #         logits = model(data)
    #         step_loss = criterion(logits, target)
    #         step_loss.backward()
    #         optimizer.step()
    
    #         pred = torch.argmax(logits, axis=1)
    #         pred = pred.eq(target).sum().item() / data.shape[0]

    #         loss += step_loss.item()
    #         acc += pred
    
    #     print(f'loss : {loss / len(train_loader)}')
    #     print(f'acc : {acc / len(train_loader)}')


def main(force_test=False):
      # Horovod: initialize library.
    hvd.init() #horovod 초기화   
        
    # hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    
    print(hvd.local_rank())
    
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)

    parse_encoder(parser)
    args = parser.parse_args()
    

    if force_test:
        args.test = True
        
    train_loop(args)


if __name__ == '__main__':
    main()
