from collections import Counter

import os
import torch
import argparse
import pickle
import time
from collections import defaultdict, Counter

#common.utils.py
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
import torch
import torch.optim as optim
import torch_geometric.utils as pyg_utils
from torch_geometric.data import DataLoader
import networkx as nx
import numpy as np
import random
import scipy.stats as stats
from tqdm import tqdm


#common.models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

#connom.subgraph.py
from itertools import combinations
import matplotlib.pyplot as plt
import sys



def feature_extract(args):
    ''' Extract feature from subgraphs
    It extracts all subgraphs feature using a trained model.
    and then, it compares DB subgraphs and query subgraphs and finds
    5 candidate DB subgraphs with similar query subgraphs.
    Finally, it counts all candidate DB subgraphs and finds The highest counted image.
    '''
    max_node = 3
    R_BFS = True
    ver = 2
    dataset, db_idx, querys, query_idx = load_dataset(max_node, R_BFS)
    db_data = utils.batch_nx_graphs(dataset, None)
    db_data = db_data.to(utils.get_device())

    # model load
    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))
    model = models.GnnEmbedder(1, args.hidden_dim, args)
    model.to(utils.get_device())
    if args.model_path:
        model.load_state_dict(torch.load(
            args.model_path, map_location=utils.get_device()))
    else:
        return print("model does not exist")

    db_check = [{i[1] for i in d.nodes(data="name")}for d in dataset]
    temp = []
    results = []
    candidate_imgs = []
    model.eval()
    with torch.no_grad():
        emb_db_data = model.emb_model(db_data)
        for i in querys:
            query = temp.copy()
            query.append(i)
            query = utils.batch_nx_graphs(query, None)
            query = query.to(utils.get_device())
            emb_query_data = model.emb_model(query)
            print(emb_db_data.shape)
            retreival_start_time = time.time()
            e = torch.sum(torch.abs(emb_query_data - emb_db_data), dim=1)
            rank = [(i, d) for i, d in enumerate(e)]
            rank.sort(key=lambda x: x[1])
            q_check = {n[1] for n in i.nodes(data="name")}
            print("Q graph nodes :", q_check)
            print("number of DB subgraph", e.shape)
            # result = [(query_idx+1, i)]
            result = []
            for n, d in rank[:5]:
                print("DB img id :", db_idx[n]+1)
                print("similarity : {:.5f}".format(d.item()))
                print("DB graph nodes :", db_check[n])
                result.append((db_idx[n]+1, dataset[n]))

                candidate_imgs.append(db_idx[n]+1)

            results.append(result)
            retreival_time = time.time() - retreival_start_time
            print("@@@@@@@@@@@@@@@@@retreival_time@@@@@@@@@@@@@@@@@ :", retreival_time)

            # Check similar/same class count with subgraph in DB
            checking_in_db = [len(q_check) - len(q_check - i)
                              for i in db_check]
            checking_result = Counter(checking_in_db)
            print(checking_result)

            # Check similar/same class with subgraph in DB
            value_checking_in_db = [
                str(q_check - (q_check - i)) for i in db_check]
            value_checking_result = Counter(value_checking_in_db)
            print(value_checking_result)
            print("==="*20)
    # Final image rank
    imgs = Counter(candidate_imgs)
    print(imgs)

    # Store result
    # if R_BFS:
    #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_RBFS.pickle", "wb") as fr:
    #         pickle.dump(results, fr)
    # else:
    #     with open("plots/data/"+"ver"+str(ver)+"_"+str(query_idx+1)+"_"+str(max_node)+"_dense.pickle", "wb") as fr:
    #         pickle.dump(results, fr)


def load_dataset(max_node, R_BFS):
    ''' Load subgraphs
    Load Scene Graph and then, Creat subgraphs from Scene Graphs.
    First, it reads scene graphs of Visual Genome and then, it makes subgraphs
    Second, it selects query image and then, it makes subgraphs
    ps) It can use user-defined query images

    max_node: When subgraphs create, It configures subgraph size.
    R_BFS: When subgraphs create, Whether it`s R_BFS mothod or not.

    Return
    db: Subgraphs in database
    db_idx: Index image of subgraphs
    query: Query subgraphs/subgraph
    query_number: Query subgraph number
    '''
    with open("data/networkx_ver3_100000/v3_x1000.pickle", "rb") as fr:
        # with open("data/networkx_ver2_10000.pickle", "rb") as fr:
        datas = pickle.load(fr)

    db = []
    db_idx = []

    # Make subgraph from scene graph of Visual Genome
    query_number = 5002
    for i in range(len(datas)):
        if query_number == i:
            continue
        subs = subgraph.make_subgraph(datas[i], max_node, False, R_BFS)
        db.extend(subs)
        db_idx.extend([i]*len(subs))

    # Select query image
    # query = subgraph.make_subgraph(datas[query_number], max_node, False, True)

    # user-defined query images
    with open("data/query_road_0819.pickle", "rb") as q:
        querys = pickle.load(q)
        query = subgraph.make_subgraph(querys[0], max_node, False, False)
        query_number = 1
    return db, db_idx, query, query_number


def main():
    parser = argparse.ArgumentParser(description='embedding arguments')

    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    feature_extract(args)









#common.utils


def sample_neigh(graphs, size):
    ps = np.array([len(g) for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))
    while True:
        idx = dist.rvs()
        #graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])
        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            #new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        if len(neigh) == size:
            return graph, neigh


cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]
    #v = [hash(tuple(v)) ^ mask for mask in cached_masks]
    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    #v = [np.sum(v) for mask in cached_masks]
    return v


def wl_hash(g, dim=64, node_anchored=False):
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=np.int)
    if node_anchored:
        for v in g.nodes:
            if g.nodes[v]["anchor"] == 1:
                vecs[v] = 1
                break
    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=np.int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]],
                                         axis=0))
        vecs = newvecs
    return tuple(np.sum(vecs, axis=0))


def gen_baseline_queries_rand_esu(queries, targets, node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    max_size = max(sizes.keys())
    all_subgraphs = defaultdict(lambda: defaultdict(list))
    total_n_max_subgraphs, total_n_subgraphs = 0, 0
    for target in tqdm(targets):
        subgraphs = enumerate_subgraph(target, k=max_size,
                                       progress_bar=len(targets) < 10, node_anchored=node_anchored)
        for (size, k), v in subgraphs.items():
            all_subgraphs[size][k] += v
            if size == max_size:
                total_n_max_subgraphs += len(v)
            total_n_subgraphs += len(v)
    print(total_n_subgraphs, "subgraphs explored")
    print(total_n_max_subgraphs, "max-size subgraphs explored")
    out = []
    for size, count in sizes.items():
        counts = all_subgraphs[size]
        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


def enumerate_subgraph(G, k=3, progress_bar=False, node_anchored=False):
    ps = np.arange(1.0, 0.0, -1.0/(k+1)) ** 1.5
    #ps = [1.0]*(k+1)
    motif_counts = defaultdict(list)
    for node in tqdm(G.nodes) if progress_bar else G.nodes:
        sg = set()
        sg.add(node)
        v_ext = set()
        neighbors = [nbr for nbr in list(G[node].keys()) if nbr > node]
        n_frac = len(neighbors) * ps[1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            v_ext.add(nbr)
        extend_subgraph(G, k, sg, v_ext, node, motif_counts, ps, node_anchored)
    return motif_counts


def extend_subgraph(G, k, sg, v_ext, node_id, motif_counts, ps, node_anchored):
    # Base case
    sg_G = G.subgraph(sg)
    if node_anchored:
        sg_G = sg_G.copy()
        nx.set_node_attributes(sg_G, 0, name="anchor")
        sg_G.nodes[node_id]["anchor"] = 1

    motif_counts[len(sg), wl_hash(sg_G,
                                  node_anchored=node_anchored)].append(sg_G)
    if len(sg) == k:
        return
    # Recursive step:
    old_v_ext = v_ext.copy()
    while len(v_ext) > 0:
        w = v_ext.pop()
        new_v_ext = v_ext.copy()
        neighbors = [nbr for nbr in list(G[w].keys()) if nbr > node_id and nbr
                     not in sg and nbr not in old_v_ext]
        n_frac = len(neighbors) * ps[len(sg) + 1]
        n_samples = int(n_frac) + (1 if random.random() < n_frac - int(n_frac)
                                   else 0)
        neighbors = random.sample(neighbors, n_samples)
        for nbr in neighbors:
            # if nbr > node_id and nbr not in sg and nbr not in old_v_ext:
            new_v_ext.add(nbr)
        sg.add(w)
        extend_subgraph(G, k, sg, new_v_ext, node_id, motif_counts, ps,
                        node_anchored)
        sg.remove(w)


def gen_baseline_queries_mfinder(queries, targets, n_samples=10000,
                                 node_anchored=False):
    sizes = Counter([len(g) for g in queries])
    #sizes = {}
    # for i in range(5, 17):
    #    sizes[i] = 10
    out = []
    for size, count in tqdm(sizes.items()):
        print(size)
        counts = defaultdict(list)
        for i in tqdm(range(n_samples)):
            graph, neigh = sample_neigh(targets, size)
            v = neigh[0]
            neigh = graph.subgraph(neigh).copy()
            nx.set_node_attributes(neigh, 0, name="anchor")
            neigh.nodes[v]["anchor"] = 1
            neigh.remove_edges_from(nx.selfloop_edges(neigh))
            counts[wl_hash(neigh, node_anchored=node_anchored)].append(neigh)
        #bads, t = 0, 0
        # for ka, nas in counts.items():
        #    for kb, nbs in counts.items():
        #        if ka != kb:
        #            for a in nas:
        #                for b in nbs:
        #                    if nx.is_isomorphic(a, b):
        #                        bads += 1
        #                        print("bad", bads, t)
        #                    t += 1

        for _, neighs in list(sorted(counts.items(), key=lambda x: len(x[1]),
                                     reverse=True))[:count]:
            print(len(neighs))
            out.append(random.choice(neighs))
    return out


device_cache = None


def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache


def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
                            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
                            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
                            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
                            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
                            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
                            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
                            help='Gradient clipping.')
    opt_parser.add_argument('--weight_decay', type=float,
                            help='Optimizer weight decay.')


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95,
                              weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


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
    batch = batch.to(get_device())
    # print(batch)
    return batch



#common.models.py

class BaselineMLP(nn.Module):
    # GNN -> concat -> MLP graph classification baseline
    def __init__(self, input_dim, hidden_dim, args):
        super(BaselineMLP, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.mlp = nn.Sequential(nn.Linear(2 * hidden_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 2))

    def forward(self, emb_motif, emb_motif_mod):
        pred = self.mlp(torch.cat((emb_motif, emb_motif_mod), dim=1))
        pred = F.log_softmax(pred, dim=1)
        return pred

    def predict(self, pred):
        return pred  # .argmax(dim=1)

    def criterion(self, pred, _, label):
        return F.nll_loss(pred, label)


class GnnEmbedder(nn.Module):
    # Gnn embedder model -- contains a graph embedding model `emb_model`
    def __init__(self, input_dim, hidden_dim, args):
        super(GnnEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.clf_model = nn.Sequential(nn.Linear(1, 1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        """Predict graph edit distance(ged) of graph pairs, where emb_as, emb_bs = pred.

        pred: list (emb_as, emb_bs) of embeddings of graph pairs.

        Returns: list of ged of graph pairs.
        """
        emb_as, emb_bs = pred

        e = torch.sum(torch.abs(emb_bs - emb_as), dim=1)

        return e

    def criterion(self, pred, intersect_embs, labels):
        """Loss function for emb.
        The e term is the predicted ged of graph pairs.

        pred: lists of embeddings outputted by forward
        intersect_embs: not used
        labels: labels for each entry in pred
        """
        emb_as, emb_bs = pred
        e = torch.sum(torch.abs(emb_bs - emb_as), dim=1)
        relation_loss = torch.sum(torch.abs(labels-e))

        return relation_loss


class SkipLastGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):  # 1, 64, 64
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        '''
        pre MLP
        '''
        # Linear(1, 64)
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, 3*hidden_dim if
                                              args.conv_type == "PNA" else hidden_dim))

        '''
        GCN
        '''
        conv_model = self.build_conv_model(args.conv_type, 1)  # SAGE
        if args.conv_type == "PNA":
            self.convs_sum = nn.ModuleList()
            self.convs_mean = nn.ModuleList()
            self.convs_max = nn.ModuleList()
        else:
            self.convs = nn.ModuleList()  # nn.Module을 리스트로 정리하는 방법, 파라미터는 리스트

        # learnable_skip = ones(8,8)
        # nn.Parameter : 모듈의 파라미터들을 iterator로 반환
        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(self.n_layers,
                                                          self.n_layers))

        # hidden_input_dim = 64 * (0~7 + 1)
        '''       
        (0): SAGEConv(64, 64)
        (1): SAGEConv(128, 64)
        (2): SAGEConv(192, 64)
        (3): SAGEConv(256, 64)
        (4): SAGEConv(320, 64)
        (5): SAGEConv(384, 64)
        (6): SAGEConv(448, 64)
        (7): SAGEConv(512, 64)
        '''
        for l in range(args.n_layers):
            if args.skip == 'all' or args.skip == 'learnable':
                hidden_input_dim = hidden_dim * (l + 1)
            else:
                hidden_input_dim = hidden_dim
            if args.conv_type == "PNA":
                self.convs_sum.append(conv_model(
                    3*hidden_input_dim, hidden_dim))
                self.convs_mean.append(conv_model(
                    3*hidden_input_dim, hidden_dim))
                self.convs_max.append(conv_model(
                    3*hidden_input_dim, hidden_dim))
            else:
                self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        '''
        post MLP
        '''
        post_input_dim = hidden_dim * (args.n_layers + 1)  # 64 * 9
        if args.conv_type == "PNA":
            post_input_dim *= 3
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),  # 64 * 9, 64
            nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),      # 64, 64
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),  # 64 256
            nn.Linear(256, hidden_dim))             # 265 64
        #self.batch_norm = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1)
        self.skip = args.skip   # True
        self.conv_type = args.conv_type     # order

    def build_conv_model(self, model_type, n_inner_layers):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            # return lambda i, h: pyg_nn.GINConv(nn.Sequential(
            #    nn.Linear(i, h), nn.ReLU()))
            return lambda i, h: GINConv(nn.Sequential(
                nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)
            ))
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "graph":
            return pyg_nn.GraphConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "gated":
            return lambda i, h: pyg_nn.GatedGraphConv(h, n_inner_layers)
        elif model_type == "PNA":
            return SAGEConv
        else:
            print("unrecognized model type")

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)  # torch.Size([538, 64])

        all_emb = x.unsqueeze(1)    # torch.Size([539, 1, 64])
        emb = x                     # torch.Size([539, 64])
        for i in range(len(self.convs_sum) if self.conv_type == "PNA" else    # i -> 0 ~ 7
                       len(self.convs)):
            if self.skip == 'learnable':
                skip_vals = self.learnable_skip[i,
                                                :i+1].unsqueeze(0).unsqueeze(-1)
                # print(skip_vals.shape)    # torch.Size([1, 1~8, 1])
                # -1 x 1 x 64 * 1 x 1 x 1 // 모든 원소에 sigmoid 값 곱하기
                curr_emb = all_emb * torch.sigmoid(skip_vals)
                curr_emb = curr_emb.view(x.size(0), -1)         # 539 x 64
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](curr_emb, edge_index),
                                   self.convs_mean[i](curr_emb, edge_index),
                                   self.convs_max[i](curr_emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](curr_emb, edge_index)
            elif self.skip == 'all':
                if self.conv_type == "PNA":
                    x = torch.cat((self.convs_sum[i](emb, edge_index),
                                   self.convs_mean[i](emb, edge_index),
                                   self.convs_max[i](emb, edge_index)), dim=-1)
                else:
                    x = self.convs[i](emb, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)    # torch.Size([539, 128])
            if self.skip == 'learnable':
                # torch.Size([539, 2, 64])
                all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        # x = pyg_nn.global_mean_pool(x, batch)

        # torch.Size([32, 576])
        emb = pyg_nn.global_add_pool(emb, batch)
        # torch.Size([32, 64])
        emb = self.post_mp(emb)

        # emb = self.batch_norm(emb)   # TODO: test
        #out = F.log_softmax(emb, dim=1)
        return emb

    def loss(self, pred, label):
        # return F.nll_loss(pred, label)
        return F.MSELoss(pred, label)


class SAGEConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add"):
        super(SAGEConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(out_channels + in_channels,
                                    out_channels)

    def forward(self, x, edge_index, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        # edge_index, edge_weight = add_remaining_self_loops(
        #    edge_index, edge_weight, 1, x.size(self.node_dim))
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_weight):
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        return self.lin(x_j)

    def update(self, aggr_out, x, res_n_id):
        aggr_out = torch.cat([aggr_out, x], dim=-1)

        aggr_out = self.lin_update(aggr_out)
        #aggr_out = torch.matmul(aggr_out, self.weight)

        # if self.bias is not None:
        #    aggr_out = aggr_out + self.bias

        # if self.normalize:
        #    aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GINConv(pyg_nn.MessagePassing):
    # pytorch geom GINConv + weighted edges
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        # reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
                                                              edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                          edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


def img_Show(nexG):
    nx.draw(nexG, with_labels=True)
    plt.show()


def make_subgraph(graph, max_node, train, R_BFS):

    def split(node, subs, max, train, R_BFS, sub=None):
        if train:
            s = max//2
            max = random.randrange(s, max)

        if sub == None:
            sub = [node]

        cur = node
        while True:
            neig = list(graph.neighbors(cur))
            neig = list(set(neig)-set(sub))
            space = max-len(sub)
            if len(neig) == 0:
                # 더이상 갈 곳이 없는 경우
                sub.sort()
                subs.add(tuple(sub))
                break
            elif len(neig) <= space:
                # 여러 곳으로 갈 수 있을 경우
                if len(neig) == 1:
                    sub.extend(neig)
                    cur = neig[0]
                else:
                    sub.extend(neig)
                    if len(neig) == space:
                        sub.sort()
                        subs.add(tuple(sub))
                        break
                    if not R_BFS:
                        # 모든 상황 고려
                        for i in neig:
                            cur = i
                            tmp = sub.copy()
                            split(cur, subs, max, False, R_BFS, tmp)
                        break
                    else:
                        cur = random.choice(neig)
            else:
                # 갈 곳이 많지만 subgraph 노드 개수를 넘을 경우
                if not R_BFS:
                    for c in combinations(list(neig), space):
                        tmp = sub.copy()
                        tmp.extend(list(c))
                        tmp.sort()
                        subs.add(tuple(tmp))
                    break
                else:
                    # 교집합 부분으로 수정해야함
                    sub.extend(
                        list(random.choice(list(combinations(neig, space)))))
                    sub.sort()
                    subs.add(tuple(sub))
                    break

    subgraphs = []
    class_set = set()
    total_subs = set()
    for i in graph.nodes():
        split(i, total_subs, max_node, train, R_BFS)
    pre = [graph.subgraph(i) for i in total_subs]
    # 노드 클래스가 중복으로 가지는 subgraph filtering
    for j in pre:
        class_sub = tuple([f['name'] for _, f in list(j.nodes.data())])
        if len(set(class_sub)) == 1:
            continue
        elif class_sub not in class_set:
            subgraphs.append(j)
            class_set.add(class_sub)
            class_set.add(tuple(reversed(class_sub)))

    return subgraphs




def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    # utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                            help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                            help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                            help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                            help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                            help='Training hidden size')
    enc_parser.add_argument('--skip', type=str,
                            help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                            help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                            help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                            help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                            help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                            help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                            help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                            help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                            help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                            help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                            help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
                            help='tag to identify the run')

    enc_parser.set_defaults(conv_type='SAGE',
                            method_type='gnn',
                            dataset='scene',     # syn
                            n_layers=8,
                            batch_size=64,  # 64, batch 개수
                            hidden_dim=64,
                            skip="learnable",
                            dropout=0.0,
                            n_batches=10,  # 1000000, total 반복
                            opt='adam',     # opt_enc_parser
                            opt_scheduler='none',
                            opt_restart=10,
                            weight_decay=0.0,
                            lr=1e-4,
                            margin=0.1,
                            test_set='',
                            eval_interval=10,   # 1000, batch 반복횟수
                            n_workers=1,        # 4
                            model_path="ckpt/final/scene_model_ver3_10000_e1.pt",
                            tag='',
                            val_size=64,         # 4096,
                            node_anchored=False)    # True

    # return enc_parser.parse_args(arg_str)





if __name__ == "__main__":
    main()
