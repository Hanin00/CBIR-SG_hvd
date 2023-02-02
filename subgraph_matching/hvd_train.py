from subgraph_matching.hvd_config import parse_encoder
from subgraph_matching.test import validation
from common import utils
from common import models
from common import data
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


import time, sys
import pandas as pd
import pickle
import random
import torch_geometric.utils as pyg_utils


def build_model(args):
    if args.method_type == "gnn":
        model = models.GnnEmbedder(1, args.hidden_dim, args)
    # elif args.method_type == "mlp":
    #     model = models.BaselineMLP(1, args.hidden_dim, args)
    model.to(utils.get_device())
    
    print("여기")

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path,
                                         map_location=utils.get_device()))
    return model

def make_data_source(args):
    if args.dataset == "scene":
        data_source = data.SceneDataSource("scene")

        
    return data_source

def load_dataset(name):
    if name == "scene":
        dataset = [[], [], []]        
        with open("common/data/v3_x1003/0_64.pickle", "rb") as fr:
            tmp = pickle.load(fr)
            for i in range(0, len(tmp[0]), 64):
                dataset[0].append(tmp[0][i])
                dataset[1].append(tmp[1][i])
                dataset[2].append(tmp[2][i])
                print(dataset)
                sys.exit()
        # for foldername in os.listdir('common/data/'):
        #     for filename in os.listdir('common/data/'+foldername):
        #         with open("common/data/"+foldername+"/"+filename, "rb") as fr:
        #             tmp = pickle.load(fr)
        #             for i in range(0, len(tmp[0]), 64):
        #                 dataset[0].append(tmp[0][i])
        #                 dataset[1].append(tmp[1][i])
        #                 dataset[2].append(tmp[2][i])
        #                 print(dataset)
        #                 sys.exit()
        return dataset
    
    task = "graph"
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




def train_loop(args):
    train_dataset = make_data_source(args)
    # loaders = train_dataset.gen_data_loaders(
    #     args.batch_size, train=False)
    
    
    # train_dataset = torchvision.datasets.MNIST(
    # root='data',
    # train=True,
    # transform=torchvision.transforms.ToTensor())

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset,
    #     num_replicas=hvd.size(),
    #     rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=8,
        sampler=train_sampler)


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
    
    for epoch in range(100):

        acc = 0
        loss = 0
        model.train()
        for data, target in tqdm.tqdm(train_loader):
    
            data = data.cuda()
            target = target.cuda()
    
            optimizer.zero_grad()
            logits = model(data)
            step_loss = criterion(logits, target)
            step_loss.backward()
            optimizer.step()
    
            pred = torch.argmax(logits, axis=1)
            pred = pred.eq(target).sum().item() / data.shape[0]

            loss += step_loss.item()
            acc += pred
    
        print(f'loss : {loss / len(train_loader)}')
        print(f'acc : {acc / len(train_loader)}')


def main(force_test=False):
    
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    
    parser = argparse.ArgumentParser(description='Embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    if force_test:
        args.test = True
        
    train_loop(args)


if __name__ == '__main__':
    main()
