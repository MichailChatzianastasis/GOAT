import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, GATIMP4layer, SpGraphAttentionLayer, GoatLayer, GoatLayerIMP4
import os
from time import ctime
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv,JumpingKnowledge, GATConv,GINConv,PNAConv
import pickle
import itertools
from torch_geometric.nn import PairNorm,LayerNorm
from dgl.nn import SAGEConv as SAGEConvDgl
import dgl
 

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, dataset):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.epoch = 0 
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False , dataset=dataset)

    def forward(self, x, adj):
        if(self.training):
            self.epoch +=1

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj,self.epoch) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj,self.epoch))
        return F.log_softmax(x, dim=1)


class GAT_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT_regression, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, outd_2, dropout=dropout, alpha=alpha, concat=False)
        self.linear = nn.Linear(outd_2,1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.linear(x)
        return x


class GATIMP4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATIMP4, self).__init__()
        final_mlp = True
        self.dropout = dropout
        self.final_mlp = final_mlp
        self.attentions = [GATIMP4layer(nfeat, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATIMP4layer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                                 concat=True, activation=None, bias=False, dataset=dataset)

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj,edge_index)
        return F.log_softmax(x, dim=1)


class GATIMP4_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATIMP4_regression, self).__init__()
        final_mlp = True
        self.dropout = dropout
        self.final_mlp = final_mlp
        self.attentions = [GATIMP4layer(nfeat, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATIMP4layer(nhid * nheads, outd_2, dropout=dropout, alpha=alpha,
                                                 concat=True, dataset=dataset)

        self.linear = nn.Linear(outd_2,1)

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        if(self.final_mlp):
            x = F.elu(self.out_att(x, adj,edge_index))
        else:
            x = F.elu(self.out_att(x,adj))
        x = self.linear(x)
        return x


class GOAT(nn.Module):
    def __init__(self, nfeat, nhid, nhid_2, nclass, lstm_h1, lstm_h2, pooling_1, pooling_2, dropout, alpha, nheads=8, nheads_2=1, dataset="cora", rnn_agg="lstm"):
        """Dense version of GAT."""
        super(GOAT, self).__init__()
        self.epoch = 0
        self.nfeat = nfeat
        self.nhid = nhid
        self.nhid_2 = nhid_2
        self.lstm_h1 = lstm_h1
        self.lstm_h2 = lstm_h2
        self.pooling_1 = pooling_1
        self.pooling_2 = pooling_2
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.nheads_2 = nheads_2
        self.dataset = dataset

        print(f"nfeat:{self.nfeat} nhid:{self.nhid} nhid_2:{self.nhid_2} lstm_h1:{self.lstm_h1} lstm_h2:{self.lstm_h2} pooling_1:{self.pooling_1} pooling_2:{self.pooling_2} nheads:{self.nheads} nheads_2:{self.nheads_2}")
        #first layer
        self.attentions = [GoatLayer(in_features=self.nfeat, out_features=self.nhid, lstm_out_features=self.lstm_h1, dropout=self.dropout, alpha=self.alpha, concat=True, dataset=self.dataset, rnn_agg=rnn_agg) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if(self.pooling_1 == "cat"):
            self.output_1 = self.lstm_h1*nheads
        else:
            self.output_1 = self.lstm_h1
        
        #second layer
        #self.out_att = [GoatLayer(in_features=self.output_1, out_features=nhid_2, lstm_out_features=nclass, dropout=dropout, alpha=alpha, concat=False, dataset=dataset) for _ in range(nheads_2)]
        self.out_att = [GraphAttentionLayer(self.output_1, nclass, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads_2)]

        #self.pairnorm = LayerNorm(self.output_1)
        self.pairnorm = PairNorm()

        for i, attention in enumerate(self.out_att):
            self.add_module('attention_2layer_{}'.format(i), attention)

        if(self.pooling_2 == "cat"):
            self.output_2 = self.lstm_h2*nheads_2
        else:
            self.output_2 = self.lstm_h2

        #self.linear = nn.Linear(self.output_2,nclass)
        
    def forward(self, x, adj):
        if(self.training):
            self.epoch +=1
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        #first layer
        if(self.pooling_1 == "cat"):
            x = torch.cat([att(x, adj,epoch=self.epoch) for att in self.attentions], dim=1)
        else: 
            #x = sum([att(x, adj,epoch=self.epoch) for att in self.attentions])
            x = torch.sum(torch.stack([att(x, adj,epoch=self.epoch) for att in self.attentions]),dim=0)
    
        #x = self.pairnorm(x)
        x= F.elu(x)
        x1 = F.dropout(x, self.dropout, training=self.training)

        #second layer
        if(self.pooling_2 == "cat"):
            x = torch.cat([out_att(x1, adj,epoch=self.epoch) for out_att in self.out_att], dim=1)
        else:
            x = torch.sum(torch.stack([out_att(x1, adj,epoch=self.epoch) for out_att in self.out_att]),dim=0)
            
        #x = F.elu(x)
        #x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)

class GATV2(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, nclass, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATV2, self).__init__()
        self.dropout = dropout
        self.epoch = 0

        #self.attentions = [GraphAttentionLayerOrdered(nfeat, nhid, outd_2, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        #for i, attention in enumerate(self.attentions):
        #    self.add_module('attention_{}'.format(i), attention)
        self.attentions=  GATConv(nfeat,nhid,heads=nheads,dropout=dropout)

        #self.out_att = GraphAttentionLayer(outd_2 * nheads, nclass, dropout=dropout, alpha=alpha, concat=False , dataset=dataset)
        self.out_att=  GATConv(nheads*nhid,nclass,heads=1,dropout=dropout)

    def forward(self, x, adj, edge_index):
        if(self.training):
            self.epoch +=1

        x = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat([att(x, adj,self.epoch) for att in self.attentions], dim=1)
        x = F.elu(self.attentions(x,edge_index))
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.log_softmax(x, dim=1)


 
class GOAT_regression(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, outd_2, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATordered_regression, self).__init__()
        self.dropout = dropout

        self.attentions = [GoatLayer(nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GoatLayer(nhid * nheads, outd_2, outd_2, dropout=dropout, alpha=alpha, concat=False, dataset=dataset)
        
        self.linear = nn.Linear(outd_2,1)        

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.linear(x)
        return x

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



##### IMPLEMENTATION 4

class GOAT_IMP4_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATorderedIMP4_regression, self).__init__()
        self.dropout = dropout
        self.attentions = [GoatLayerIMP4(nfeat, nhid, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True,bias=True,add_skip_connection=False, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.out_att = GoatLayerIMP4(nhid * nheads, outd_2, outd_2, dropout=dropout, alpha=alpha,
                                                 concat=True, bias=True, add_skip_connection=False, dataset=dataset)
        self.linear = nn.Linear(outd_2,1)

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj,edge_index)
        x = self.linear(x)
        return x




class GOAT_IMP4(nn.Module):
    def __init__(self, nfeat, nhid, nhid_2, pooling_1, nclass, dropout, alpha,  nheads, final_mlp=False, dataset="cora",rnn_agg="lstm"):
        """Dense version of GAT."""
        super(GOAT_IMP4, self).__init__()
        
        self.dropout = dropout
        self.final_mlp = final_mlp
        self.attentions = [GoatLayerIMP4(nfeat, nhid, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True, bias=True, add_skip_connection=False, dataset=dataset,rnn_agg=rnn_agg) for _ in range(nheads)]
        pooling_1 = "cat"
        self.pooling_1 = pooling_1
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if(self.pooling_1 == "cat"):
            output_1 = nhid * nheads
        else:
            output_1 = nhid

        #self.out_att = GoatLayerIMP4(output_1, nclass, lstm_out_features=nclass, dropout=dropout, alpha=alpha,concat=True, bias=True, add_skip_connection=False, dataset=dataset,rnn_agg=rnn_agg)

        self.out_att =  GATIMP4layer(output_1, nclass, dropout=dropout, alpha=alpha, concat=True, activation=None, bias=False, dataset=dataset)

        #self.out_att = GATOrderedLayerIMPL4arxiv_ogbn(output_1, nclass, lstm_out_features=nclass, dropout=dropout, alpha=alpha,concat=True, bias=True, add_skip_connection=False, dataset=dataset,rnn_agg=rnn_agg)        #self.linear = nn.Linear(nclass,nclass)
        #self.jumpk = JumpingKnowledge("max")

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        
        if(self.pooling_1=="cat"):
            x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        else:
            x = torch.sum(torch.stack([att(x, adj, edge_index) for att in self.attentions]),dim=0)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj,edge_index)

        #x = self.jumpk([x,x2])
        #x = self.linear(x)
        
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)
        self.conv1 = GCNConv(outfeat, outd_1)
        self.conv2 = GCNConv(outd_1, nclass)
        

    def forward(self, x, adj, edge_index):
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        x = self.leakyrelu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class PNA(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,nclass,deg,alpha=0.2):
        super().__init__()


        #aggregators = ['mean', 'min', 'max', 'std']
        #scalers = ['identity', 'amplification', 'attenuation','linear']
        aggregators = ["max"]
        scalers = ["amplification",'identity']

        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)
        self.conv1 = PNAConv(outfeat, outd_1,aggregators,scalers, deg)
        self.conv2 = PNAConv(outd_1, nclass,aggregators,scalers, deg)
        

    def forward(self, x, adj, edge_index):
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        x = self.leakyrelu(self.conv1(x, edge_index))

        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)

        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(nfeat, outfeat),  self.leakyrelu,
                       nn.Linear(outfeat, outd_1), self.leakyrelu))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(outd_1, outd_1),  self.leakyrelu,
                       nn.Linear(outd_1, nclass), self.leakyrelu))
        
    def forward(self, x, adj, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,aggregator_type,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat) 
        self.conv1 = SAGEConvDgl(outfeat,outd_1, aggregator_type=aggregator_type)
        self.conv2 = SAGEConvDgl(outd_1,nclass, aggregator_type=aggregator_type)
        self.dgl_graph = 1
         
    def forward(self,x,adj,edge_index):
        
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        #x = self.leakyrelu(self.conv1(x,edge_index))
        x = self.leakyrelu(self.conv1(self.dgl_graph,x))
        x = F.dropout(x, training=self.training)

        #x = self.conv2(x,edge_index)
        x = self.conv2(self.dgl_graph,x)

        return F.log_softmax(x, dim=1)
        

class GCN_regression(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.linear = nn.Linear(nfeat,outfeat)
        self.conv1 = GCNConv(outfeat, outd_1)
        self.conv2 = GCNConv(outd_1, outd_2)
        self.linear_2 = nn.Linear(outd_2,1)
        
    def forward(self, x, adj, edge_index):
        x = self.leakyrelu(self.linear(x))
        x = F.dropout(x,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear_2(x)
        return x


class PNA_regression(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,deg,alpha=0.2):
        super().__init__()

        #aggregators = ['mean', 'min', 'max', 'std']
        #scalers = ['identity', 'amplification', 'attenuation','linear']
        aggregators = ["mean","max"]
        scalers = ['attenuation',"linear"]

        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.linear = nn.Linear(nfeat,outfeat)
        self.conv1 = PNAConv(outfeat, outd_1,aggregators,scalers, deg)
        self.conv2 = PNAConv(outd_1, outd_2,aggregators,scalers, deg)
        self.linear_2 = nn.Linear(outd_2,1)
        
    def forward(self, x, adj, edge_index):
        x = self.leakyrelu(self.linear(x))
        x = F.dropout(x,training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear_2(x)
        return x

class GIN_regression(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.linear = nn.Linear(nfeat,outfeat)
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(nfeat, outfeat),  self.leakyrelu,
                       nn.Linear(outfeat, outd_1), self.leakyrelu))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(outd_1, outd_1),  self.leakyrelu,
                       nn.Linear(outd_1, outd_2), self.leakyrelu))
        
        self.linear_2 = nn.Linear(outd_2,1)
        
    def forward(self, x, adj, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.linear_2(x)
        return x


class GraphSAGE_regression(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,outd_2,aggregator_type="lstm",alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.linear = nn.Linear(nfeat,outfeat)
        
        #self.conv1 = SAGEConv(outfeat,outd_1)
        #self.conv2 = SAGEConv(outd_1,outd_2)
        self.conv1 = SAGEConvDgl(outfeat,outd_1, aggregator_type=aggregator_type)
        self.conv2 = SAGEConvDgl(outd_1,outd_2, aggregator_type=aggregator_type)
        self.linear_2 = nn.Linear(outd_2,1)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dgl_graph = 1

    def forward(self,x,adj,edge_index):
        x = self.leakyrelu(self.linear(x))
        x = F.dropout(x,training=self.training)
        
        x = F.relu(self.conv1(self.dgl_graph,x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(self.dgl_graph,x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear_2(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self,nfeat,outfeat,outd_1,nclass,alpha=0.2):
        super().__init__()
        self.alpha = alpha  
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.linear = nn.Linear(nfeat,outfeat)
        self.linear_2 = nn.Linear(outfeat,outd_1)
        self.linear_3 = nn.Linear(outd_1,nclass)

    def forward(self,x,adj,edge_index=None):
        x = self.leakyrelu(self.linear(x))

        x = F.dropout(x,training=self.training)
        
        x = self.leakyrelu(self.linear_2(x))
        x = F.dropout(x, training=self.training)

        
        x= self.linear_3(x)
        return F.log_softmax(x, dim=1)
 
