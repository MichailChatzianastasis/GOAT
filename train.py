from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from graphviz import Digraph
import scipy.sparse as sp
import dgl
from sklearn.metrics import f1_score
import networkx as nx
from utils import load_data, accuracy
from models import GAT, GIN, SpGAT, GATorderedDeep, MLP, GATv2ConvOrdered,GATV2, GATordered, GATorderedIMP4 ,GATordered_shared_LSTM, PNA, GCN, GraphSAGE, GATRandomOrdered,GATIMP4, GATorderedIMP4_node_batching, GATorderedGraphClassification_LSTM_graph_pooling
from utils2 import load_data2,load_extra_data, load_ogbn_arxiv,load_lastfm_asia,load_disease,load_email_eu,load_amazon, load_data_adsf
from torch_geometric.data import DataLoader
import sys

from torch_geometric.utils import degree


print("start")

def print_shape(tensor_list):
    for i in tensor_list:
        print(i.shape)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
#parser.add_argument('--lr', type=float, default=1, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=250, help='Patience')
parser.add_argument('--goat_imp4', default=False)
parser.add_argument('--goat', default=False)
parser.add_argument('--rnn_agg', default="lstm", help='For GOAT --goat True')

parser.add_argument('--v2', default=False)
parser.add_argument('--v3', default=False)

parser.add_argument('--gcn', default=False)
parser.add_argument('--sage', default=False)
parser.add_argument('--adsf', default=False)

parser.add_argument('--mlp', default=False)
parser.add_argument('--gin', default=False)
parser.add_argument('--pna', default=False)


parser.add_argument('--edge_index', default=False)

parser.add_argument('--random', default= False) #for random ordering model
parser.add_argument('--final_outd', type=int, default = 8)
parser.add_argument('--hidden_2', type=int, default = 8)

parser.add_argument('--outd_1', type=int, default = 8)
parser.add_argument('--outd_2', type=int, default = 8)
parser.add_argument('--lstm_h1', type=int, default = 8)
parser.add_argument('--lstm_h2', type=int, default = 8)
parser.add_argument('--pooling_1' , default="cat")
parser.add_argument('--pooling_2' , default="cat")
parser.add_argument('--nb_heads_2', type=int, default=1, help='Number of head attentions in second layer.')
parser.add_argument('--aggregator_type', default="lstm", help='For GRAPH SAGE ONLY')

parser.add_argument('--dataset' , default="disease_nc")
parser.add_argument('--final_mlp', default= False)
parser.add_argument('--impl4', default= False)
parser.add_argument('--shared', default= False)
parser.add_argument('--central',default=False) #for largest two neighbors if phi will use central node features.

parser.add_argument('--random_idx',type=int,default = 0)
parser.add_argument('--node_batch', default= False)
parser.add_argument("--number_of_nodes",type=int,default=1000)
parser.add_argument('--number_of_classes', type=int, default= 2) # for sbm
parser.add_argument('--edge_prob',type=float,default=0.01)





args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

impl4 = args.impl4
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def find_edge_index(adj):
    print("find edge index")
    edge_index = [[],[]]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if(adj[i,j]>0):
                edge_index[0].append(i)
                edge_index[1].append(j)
        if(i%50 ==0):
            print(i)
    
    edge_index = np.array(edge_index)
    edge_index = torch.tensor(edge_index)
    with open(f"{args.dataset}_edge_index","wb") as handle:
        pickle.dump(edge_index,handle)
    return edge_index


#if(args.h_sizes_1):
#    args.h_sizes_1 = args.h_sizes_1[1:-1]
#    args.h_sizes_1 = [int(item) for item in args.h_sizes_1.split(',')]

#if(args.h_sizes_2):
#    args.h_sizes_2 = args.h_sizes_2[1:-1]
#    args.h_sizes_2 = [int(item) for item in args.h_sizes_2.split(',')]

 
def spy_sparse2torch_sparse(data):
    """

    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t

# Load data
if(args.dataset == "cora"): 
    if(args.adsf):
        adj, features,  idx_train, idx_val, idx_test, labels, adj_ad = load_data_adsf("cora")
        adj_ad = adj_ad.cuda()
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_data()


elif(args.dataset == "citeseer"):
    #adj, features, labels, idx_train, idx_val, idx_test = load_data("./data/citeseer/","citeseer")
    if(args.adsf):
        adj, features,  idx_train, idx_val, idx_test, labels, adj_ad = load_data_adsf("citeseer")
        adj_ad = adj_ad.cuda()
    
    adj, features, labels, idx_train, idx_val, idx_test = load_data2("citeseer")
    

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    #labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

elif(args.dataset == "pubmed"):
    #adj, features, labels, idx_train, idx_val, idx_test = load_data("./data/citeseer/","citeseer")
    adj, features, labels, idx_train, idx_val, idx_test = load_data2("pubmed")

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    #labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

elif(args.dataset == "cornell" or args.dataset == "texas" or args.dataset == "wisconsin" or args.dataset == "squirrel"):
    adj, features, labels, idx_train, idx_val, idx_test = load_extra_data(args.dataset,int(args.random_idx))
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    #labels = torch.LongTensor(np.where(labels)[1])
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

elif(args.dataset == "ogbn-arxiv"):
    adj, edge_index, features,labels,idx_train,idx_val,idx_test = load_ogbn_arxiv(args.dataset)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = labels.squeeze()

elif(args.dataset == "lastfm_asia"):
    adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_lastfm_asia(args.dataset,random_state=args.random_idx)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

elif(args.dataset == "email_eu"):
    adj, features, labels, idx_train, idx_val, idx_test = load_email_eu(args.dataset)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


elif(args.dataset == "disease_nc"):
    adj, features, labels, idx_train, idx_val, idx_test = load_disease(args.dataset,random_state=args.random_idx)
    adj = torch.FloatTensor(adj)
    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

elif(args.dataset == "Computers" or args.dataset == "Photo"):
    
    adj, edge_index, features, labels, idx_train, idx_val, idx_test = load_amazon(args.dataset,random_state=args.random_idx)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = labels.squeeze()

store_seq_length = False
if(store_seq_length == True):
    seq_length = sp.csr_matrix.getnnz(adj,axis=0)
    with open(f'./{args.dataset}_seq_length', 'wb') as handle:
        pickle.dump(seq_length,handle)

if(args.dataset == "ogbn-arxiv" or args.dataset == "Photo" or args.dataset == "Computers"):
    adj = spy_sparse2torch_sparse(adj)
    if(args.cuda):
        edge_index = edge_index.cuda()

 
create_edge_dict =  False
if(create_edge_dict == True):
    counter = 0
    edge_dict = {}
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if(adj[i,j]>0):
                counter+=1
                if(i in edge_dict):
                    edge_dict[i].append(j)
                else:
                    edge_dict[i] = [j]
        print(i,"/",adj.shape[0])
    with open(f'./{args.dataset}_edge_dict', 'wb') as handle:
        pickle.dump(edge_dict,handle)
    print("found edges",counter)


# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
else:
    if args.goat_imp4:
        model = GATorderedIMP4(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nhid_2 = args.hidden_2,
                pooling_1 = args.pooling_1,
                nclass=int(labels.max()) + 1, 
                lstm_h1 = args.lstm_h1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                dataset= args.dataset,
                rnn_agg=args.rnn_agg,
                final_mlp=args.final_mlp)
    
    elif args.goat:
        model = GATordered(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nhid_2 = args.hidden_2,
                nclass=int(labels.max()) + 1, 
                lstm_h1 = args.lstm_h1,
                lstm_h2 = args.lstm_h2,
                pooling_1=args.pooling_1,
                pooling_2=args.pooling_2,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                nheads_2 = args.nb_heads_2,
                alpha=args.alpha,
                dataset= args.dataset,
                rnn_agg = args.rnn_agg)

    elif args.v3:
        model = GATv2ConvOrdered(nfeat=features.shape[1], 
                nhid=args.hidden, 
                outd_2= args.outd_2,
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                dataset= args.dataset,
                final_mlp=args.final_mlp)
        args.edge_index = True

    elif args.v2:
        model = GATV2(nfeat=features.shape[1], 
                nhid=args.hidden, 
                outd_2= args.outd_2,
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                dataset= args.dataset,
                final_mlp=args.final_mlp)
                
        args.edge_index = True

    elif args.gcn:
        model = GCN(nfeat = features.shape[1],
                outfeat=args.hidden,
                outd_1=args.outd_1,
                outd_2=args.outd_2,
                nclass = int(labels.max()) + 1,
                )
        args.edge_index = True

    elif args.pna:
        if(args.dataset!="Computers" and args.dataset!="Photo"):
            with open(f"{args.dataset}_edge_index","rb") as handle:
                edge_index = pickle.load(handle)
        #edge_index = find_edge_index(adj)
        edge_index = edge_index.cpu()
        max_degree = -1
        d = degree(edge_index[1], num_nodes=features.shape[0], dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        d = degree(edge_index[1], num_nodes=features.shape[0], dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

        edge_index = edge_index.cuda()
        model = PNA(nfeat = features.shape[1],
                outfeat=args.hidden,
                outd_1=args.outd_1,
                outd_2=args.outd_2,
                nclass = int(labels.max()) + 1,
                deg = deg
                )
        args.edge_index = True
        # Compute the maximum in-degree in the training data.
        
    elif args.gin:
        model = GIN(nfeat = features.shape[1],
                outfeat=args.hidden,
                outd_1=args.outd_1,
                outd_2=args.outd_2,
                nclass = int(labels.max()) + 1,
                )
        args.edge_index = True
    
    elif args.sage:
            model = GraphSAGE(nfeat = features.shape[1],
                    outfeat=args.hidden,
                    outd_1 = args.outd_1,
                    outd_2 = args.outd_2,
                    nclass = int(labels.max()) + 1,
                    aggregator_type = args.aggregator_type
                    )
            args.edge_index = True
    elif args.random:
         model = GATRandomOrdered(nfeat = features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                final_outd=args.final_outd,
                alpha=args.alpha,
                dataset= args.dataset,
                final_mlp=args.final_mlp)

    elif args.shared:
        model = GATordered_shared_LSTM(nfeat=features.shape[1], 
        nhid=args.hidden, 
        outd_2=args.outd_2,
        nclass=int(labels.max()) + 1, 
        dropout=args.dropout, 
        nheads=args.nb_heads, 
        alpha=args.alpha,
        dataset= args.dataset,
        final_mlp=args.final_mlp)

 

    elif args.adsf:
        model = ADSF(nfeat=features.shape[1],
            nhid=args.hidden, 
            nclass=int(labels.max()) + 1, 
            dropout=args.dropout, 
            nheads=args.nb_heads, 
            alpha=args.alpha,
            adj_ad=adj_ad.cuda())

    elif args.node_batch:
        model = GATorderedIMP4_node_batching(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                final_outd=args.final_outd,
                alpha=args.alpha,
                dataset= args.dataset,
                final_mlp=args.final_mlp)
    
    elif args.mlp:
        model = MLP(nfeat = features.shape[1],
                outfeat=args.hidden,
                outd_1=args.hidden,
                nclass = int(labels.max()) + 1
                )
        args.edge_index = True

    else:
        impl4 = True
        if(impl4):
            model = GATIMP4(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha,
                    dataset= args.dataset,
                    final_mlp=args.final_mlp)
        else:
            model = GAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha,
                    dataset = args.dataset)



optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)






if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

if (args.goat_imp4 or impl4 or args.edge_index) and args.dataset!="ogbn-arxiv" and args.dataset!="Photo" and args.dataset!="Computers": #for ogbn-arxiv we have already edge_index
    if(args.dataset=='sbm' or args.dataset=="largest_2_neighbors"):
        edge_index = find_edge_index(adj)
    elif(os.path.isfile(f"{args.dataset}_edge_index")):
        with open(f"{args.dataset}_edge_index","rb") as handle:
            edge_index = pickle.load(handle)
    else:
        edge_index = find_edge_index(adj)
    if(args.cuda):
        edge_index = edge_index.cuda()

def train(epoch):

    t = time.time()
    model.train()
    optimizer.zero_grad()
    if(args.goat_imp4 or impl4 or args.edge_index) or args.dataset =="ogbn-arxiv" or args.dataset =="Photo" or args.dataset =="Computers":
        output = model(features, adj, edge_index)
    elif(args.adsf):
        output = model(features,adj,adj_ad)
    else:
        output = model(features,adj)
    #g = make_dot(output, model.state_dict())
    #g.view()
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    #plot_grad_flow2(model.named_parameters())

    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if(args.goat_imp4 or impl4 or args.edge_index or args.dataset=="ogbn-arxiv"):
            output = model(features, adj, edge_index)
        
        elif(args.adsf):
            output = model(features,adj,adj_ad)
        
        else:
            output = model(features,adj)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test(write_results=True):
    model.eval()
    if(args.goat_imp4 or impl4 or args.edge_index or args.dataset=="ogbn-arxiv"):
        output = model(features, adj, edge_index)
    
    elif(args.adsf):
        output = model(features,adj,adj_ad)
    
    else:
        output = model(features,adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


    if(args.dataset == "disease_nc"):
        preds = output[idx_test].max(1)[1].type_as(labels)
        f1 = f1_score(labels[idx_test].cpu().detach().numpy(),preds.cpu().detach().numpy()).item()
    else:
        f1 = -1
    if(write_results):
        with open("./results/results.txt","a") as handle:
            if(args.dataset == "sbm"):
                handle.write(f'{args.dataset} Number_of_nodes: {args.number_of_nodes} heads:{args.nb_heads} {args.number_of_classes} { type(model).__name__} {args.aggregator_type} hidden:{args.hidden}, outd1:{args.outd_1}, random_idx:{args.random_idx}:  {str(loss_test.item())} , {str(acc_test.item())}')
            elif(args.gcn or args.mlp or args.gin or args.pna ):
                handle.write(f'{args.dataset} { type(model).__name__}  outfeat:{args.hidden}, outd1:{args.outd_1}, random_idx:{args.random_idx}: {str(f1)} {str(loss_test.item())} , {str(acc_test.item())}')
                #handle.write("\n")
            elif(args.sage):
                handle.write(f'{args.dataset} {args.aggregator_type} {  type(model).__name__}  outfeat:{args.hidden}, outd1:{args.outd_1}, outd2:{args.outd_2}, random_idx:{args.random_idx}: {str(f1)} {str(loss_test.item())} , {str(acc_test.item())}')
            elif(args.goat):
                handle.write(f'{args.dataset} { type(model).__name__} aggregator:{args.rnn_agg} {type(model.attentions[0].lstm).__name__} nhid:{args.hidden} nhid_2:{args.hidden_2} lstm_h1:{args.lstm_h1} lstm_h2: {args.lstm_h2} pooling_1: {args.pooling_1} pooling_2: {args.pooling_2} nb_heads: {args.nb_heads} nb_heads2: {args.nb_heads_2} "random_idx":{args.random_idx}: {str(f1)} {str(loss_test.item())}, {str(acc_test.item())}')
            elif hasattr(model.attentions, 'lstm'):
                handle.write(f'{args.dataset} { type(model).__name__} {type(model.attentions[0].lstm).__name__} nhid:{args.hidden} outd_2:{args.outd_2} pooling_1: {args.pooling_1} nb_heads: {args.nb_heads} "random_idx":{args.random_idx}: {str(f1)} {str(loss_test.item())}, {str(acc_test.item())}')
            else: 
                handle.write(f'{args.dataset} { type(model).__name__} aggregator:{args.rnn_agg}  nhid: {args.hidden} hidden_2: {args.hidden_2} lstm_h1: {args.lstm_h1} nb_heads: {args.nb_heads} pooling_1: {args.pooling_1} "random_idx":{args.random_idx} {str(f1)} {str(loss_test.item())}, {str(acc_test.item())}')
            if(args.dataset == "largest_2_neighbors"):
                handle.write(f" central:{args.central}")
            handle.write("\n")
            # 4 10 12 nbheads

# Train model


if(args.sage):
    model.dgl_graph = dgl.graph((edge_index[0],edge_index[1]))

t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

    if (epoch % 50 == 0 ):
        print("test resulst epoch:",epoch)
        compute_test(write_results=False)
        
print("evaluation on last epoch")
compute_test(write_results=False)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test(write_results=True)

