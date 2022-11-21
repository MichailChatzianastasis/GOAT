import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer,LSTMAggregation,GraphAttentionLayerOrdered2,GraphAttentionV2LayerOrdered, GATIMP4layer, GraphAttentionLayerOrdered_Shared_LSTM, GATOrderedLayerIMPL4arxiv_ogbn_node_batching, GraphAttentionLayerRandomOrdered, SpGraphAttentionLayer, Lstm_graph_pooling, GraphAttentionLayerOurs,GraphAttentionLayerOrdered, GATOrderedLayerIMPL4,GATOrderedLayerIMPL4arxiv_ogbn, GATOrderedLayerIMPL4_graph_classification, GraphAttentionLayerGraphClassification
import os
from time import ctime
from torch_geometric.nn import GCNConv,SAGEConv,GATv2Conv,JumpingKnowledge, GATConv,GINConv,PNAConv
import pickle
import itertools
from torch_geometric.nn import PairNorm,LayerNorm
from dgl.nn import SAGEConv as SAGEConvDgl
import dgl
 
from layers import StructuralFingerprintLayer

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


class GATv2ConvOrdered(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, nclass, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATv2ConvOrdered, self).__init__()
        self.dropout = dropout
        self.attentions = GATv2Conv(in_channels=nfeat, out_channels=nhid, heads=nheads, dropout=dropout, concat=True) 
        self.lstm_aggregation = LSTMAggregation(nhid*nheads, outd_2, dataset=dataset)

        self.out_att = GATv2Conv(in_channels=outd_2, out_channels=nclass, heads=1, dropout=dropout, concat=False) 
        self.lstm_aggregation_2 = LSTMAggregation(nclass, nclass, dataset=dataset)

        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 
        
        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
         
        with open(f"./graph_lengths/{dataset}_maximum_neighbors.pkl",'rb') as handle:
            self.maximum_neighbors = pickle.load(handle)

    def forward(self, x, adj, edge_index):

        #first layer     
        x,out = self.attentions(x,edge_index,return_attention_weights=True)
        edge_index,att = out
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lstm_aggregation(x,att,edge_index,x,adj.shape[0])
        
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        #second layer
        x,out = self.out_att(x, edge_index,return_attention_weights=True)
        edge_index,att = out
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lstm_aggregation_2(x,att,edge_index,x,adj.shape[0])
        x = F.dropout(x, self.dropout, training=self.training)

        return F.log_softmax(x, dim=1)


class GATordered(nn.Module):
    def __init__(self, nfeat, nhid, nhid_2, nclass, lstm_h1, lstm_h2, pooling_1, pooling_2, dropout, alpha, nheads=8, nheads_2=1, dataset="cora", rnn_agg="lstm"):
        """Dense version of GAT."""
        super(GATordered, self).__init__()
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
        self.attentions = [GraphAttentionLayerOrdered(in_features=self.nfeat, out_features=self.nhid, lstm_out_features=self.lstm_h1, dropout=self.dropout, alpha=self.alpha, concat=True, dataset=self.dataset, rnn_agg=rnn_agg) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if(self.pooling_1 == "cat"):
            self.output_1 = self.lstm_h1*nheads
        else:
            self.output_1 = self.lstm_h1
        
        #second layer
        #self.out_att = [GraphAttentionLayerOrdered(in_features=self.output_1, out_features=nhid_2, lstm_out_features=nclass, dropout=dropout, alpha=alpha, concat=False, dataset=dataset) for _ in range(nheads_2)]
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


class GATordered_shared_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, nclass, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATordered_shared_LSTM, self).__init__()
        self.dropout = dropout
        self.shared_lstm = nn.GRU(nhid,nhid,1,bidirectional=True,batch_first=True)
       
        self.attentions = [GraphAttentionLayerOrdered_Shared_LSTM(self.shared_lstm, nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerOrdered(nhid * nheads, outd_2, outd_2=nclass, dropout=dropout, alpha=alpha, concat=False, dataset=dataset)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
        


class GATordered_shared_LSTM_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATordered_shared_LSTM_regression, self).__init__()
        self.dropout = dropout
        self.shared_lstm = nn.LSTM(nhid,nhid,1,bidirectional=True,batch_first=True)
       
        self.attentions = [GraphAttentionLayerOrdered_Shared_LSTM(self.shared_lstm, nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerOrdered(nhid * nheads, outd_2, outd_2=outd_2, dropout=dropout, alpha=alpha, concat=False, dataset=dataset)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GATordered_regression(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, outd_2, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATordered_regression, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayerOrdered(nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerOrdered(nhid * nheads, outd_2, outd_2, dropout=dropout, alpha=alpha, concat=False, dataset=dataset)
        
        self.linear = nn.Linear(outd_2,1)        

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.linear(x)
        return x


class GATRandomOrdered(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATRandomOrdered, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayerRandomOrdered(nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerRandomOrdered(nhid * nheads, outd_2, final_outd= nclass, dropout=dropout, alpha=alpha, dataset=dataset, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



class GATorderedDeep(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, final_outd, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATorderedDeep, self).__init__()
        self.dropout = dropout
        nheads_2 = 4
        self.attentions = [GraphAttentionLayerOrdered(nfeat, nhid, dropout=dropout, alpha=alpha, final_outd=final_outd, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.attentions2 = [GraphAttentionLayerOrdered(final_outd * nheads, nhid, dropout=dropout, alpha=alpha, final_outd=final_outd, concat=True, dataset=dataset) for _ in range(nheads_2)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


        self.out_att = GraphAttentionLayer(final_outd *nheads_2, nclass, dropout=dropout, alpha=alpha, concat=False)
            #self.out_att = GraphAttentionLayerOrderedFinal(final_outd * 2*nheads_2, final_outd * 2*nheads, dropout=dropout, alpha=alpha, final_outd=nclass, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,adj) for att in self.attentions2], dim=1)
        x = F.dropout(x,self.dropout,training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)



class GATRandomOrdered_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATRandomOrdered_regression, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayerRandomOrdered(nfeat, nhid, nhid, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerRandomOrdered(nhid * nheads, outd_2, dropout=dropout, alpha=alpha, dataset=dataset, concat=False)
        self.linear = nn.Linear(outd_2,1)  

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = F.dropout(F.elu(self.linear(x)))
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

class GATorderedIMP4_regression(nn.Module):
    def __init__(self, nfeat, nhid, outd_2, dropout, alpha, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATorderedIMP4_regression, self).__init__()
        self.dropout = dropout
        self.attentions = [GATOrderedLayerIMPL4arxiv_ogbn(nfeat, nhid, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True,bias=True,add_skip_connection=False, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        self.out_att = GATOrderedLayerIMPL4arxiv_ogbn(nhid * nheads, outd_2, outd_2, dropout=dropout, alpha=alpha,
                                                 concat=True, bias=True, add_skip_connection=False, dataset=dataset)
        self.linear = nn.Linear(outd_2,1)

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj,edge_index)
        x = self.linear(x)
        return x




class GATorderedIMP4(nn.Module):
    def __init__(self, nfeat, nhid, nhid_2, pooling_1, nclass, lstm_h1, dropout, alpha,  nheads, final_mlp=False, dataset="cora",rnn_agg="lstm"):
        """Dense version of GAT."""
        super(GATorderedIMP4, self).__init__()
        
        self.dropout = dropout
        self.final_mlp = final_mlp
        self.attentions = [GATOrderedLayerIMPL4arxiv_ogbn(nfeat, nhid, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True, bias=True, add_skip_connection=False, dataset=dataset,rnn_agg=rnn_agg) for _ in range(nheads)]
        pooling_1 = "cat"
        self.pooling_1 = pooling_1
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if(self.pooling_1 == "cat"):
            output_1 = nhid * nheads
        else:
            output_1 = nhid

        #self.out_att = GATOrderedLayerIMPL4arxiv_ogbn(output_1, nclass, lstm_out_features=nclass, dropout=dropout, alpha=alpha,concat=True, bias=True, add_skip_connection=False, dataset=dataset,rnn_agg=rnn_agg)

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


class GATorderedIMP4_node_batching(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, final_outd, nheads, final_mlp=False, dataset="cora"):
        """Dense version of GAT."""
        super(GATorderedIMP4_node_batching, self).__init__()
        final_mlp = True
        self.dropout = dropout
        self.final_mlp = final_mlp
        self.attentions = [GATOrderedLayerIMPL4arxiv_ogbn_node_batching(nfeat, nhid, dropout=dropout, alpha=alpha,
                                                 concat=True, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #if(final_mlp == False):
        #    self.out_att = GraphAttentionLayer(final_outd * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #else:
            #self.out_att  = MlpFinalLayer(nclass=nclass,number_layers=2,in_dimension=final_outd*nheads,dataset=dataset)
        self.out_att = GATOrderedLayerIMPL4arxiv_ogbn_node_batching(nhid * nheads, final_outd, dropout=dropout, alpha=alpha,
                                                 concat=True, dataset=dataset)

        self.linear = nn.Linear(final_outd,nclass)

    def forward(self, x, adj , edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        if(self.final_mlp):
            x = F.elu(self.out_att(x, adj,edge_index))
        else:
            x = F.elu(self.out_att(x,adj))
        x = self.linear(x)
        return F.log_softmax(x, dim=1)






class GATorderedGraphClassification(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, final_outd, linear_dim, nheads, final_mlp=False, readout="sum", dataset="MUTAG"):
        """Dense version of GAT."""
        super(GATorderedGraphClassification, self).__init__()
        self.dropout = dropout

        #
        #self.attentions = [GATOrderedLayerIMPL4_graph_classification(nfeat, nhid, dropout=dropout, alpha=alpha,
        #                                        concat=True, dataset=dataset) for _ in range(nheads)]
        self.attentions = [GraphAttentionLayerOrdered(in_features=nfeat, out_features=nhid, lstm_out_features=8, dropout=self.dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


        #self.out_att = GATOrderedLayerIMPL4_graph_classification(final_outd * nheads, linear_dim, dropout=dropout, alpha=alpha,
        #                                        concat=True, dataset=dataset)
        self.out_att = [GraphAttentionLayerOrdered(in_features=final_outd * nheads, out_features=nhid, lstm_out_features=linear_dim, dropout=dropout, alpha=alpha, concat=True, dataset=dataset) for _ in range(nheads)]

        
        if(readout=="sum"):
            self.readout= torch.sum
        elif(readout == "max"):   
            self.readout  = torch.max
        elif(readout == "mean"):
            self.readout = torch.mean
        
        self.linear_1 = nn.Linear(linear_dim,2*linear_dim)
        self.linear_2 = nn.Linear(2*linear_dim,nclass)

    def forward(self, graph):
        if(type(graph) is list): #for minibatch =1 
            graph = graph[0]
        x = F.dropout(graph.node_features, self.dropout, training=self.training)
        x = torch.cat([att(x,graph) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x,graph))

        x = F.elu(F.dropout(self.linear_1(x)))
        x = self.readout(x,dim=0)
        x = self.linear_2(x)
        x = x.unsqueeze(dim=0)
        return F.log_softmax(x, dim=1)

class GATorderedGraphClassification_LSTM_graph_pooling(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, final_outd, linear_dim, nheads, final_mlp=False, bidirectional=False, max_mean_lstm=False,dataset="MUTAG"):
        """Dense version of GAT."""
        super(GATorderedGraphClassification_LSTM_graph_pooling, self).__init__()
        self.dropout = dropout

        self.attentions = [GATOrderedLayerIMPL4_graph_classification(nfeat, nhid, dropout=dropout, alpha=alpha,
                                                concat=True, dataset=dataset) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        #if(final_mlp == False):
        #    self.out_att = GraphAttentionLayer(final_outd * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        #else:
            #self.out_att  = MlpFinalLayer(nclass=nclass,number_layers=2,in_dimension=final_outd*nheads,dataset=dataset)
        self.out_att = GATOrderedLayerIMPL4_graph_classification(final_outd * nheads, linear_dim, dropout=dropout, alpha=alpha,
                                                concat=True, dataset=dataset)
 
        self.lstm_pooling = Lstm_graph_pooling(input_dim=linear_dim, output_dim=linear_dim, dropout=dropout, bidirectional=bidirectional, max_mean_lstm=max_mean_lstm, dataset=dataset) 
        
        self.linear_1 = nn.Linear(linear_dim,nclass)

    def forward(self, graph):
        if(type(graph) is list): #for minibatch =1 
            graph = graph[0]
        x = F.dropout(graph.node_features, self.dropout, training=self.training)
        x = torch.cat([att(x,graph) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x,graph))

        #x = F.elu(F.dropout(self.linear_1(x)))
        x = self.lstm_pooling(x,graph)
        x = self.linear_1(x)
        x = x.unsqueeze(dim=0)
        return F.log_softmax(x, dim=1)



class GATGraphClassification(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, final_outd, linear_dim, nheads, final_mlp=True, readout="sum", dataset="MUTAG"):
        """Dense version of GAT."""
        super(GATGraphClassification, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayerGraphClassification(nfeat, nhid, dropout=dropout, alpha=alpha,
                                                concat=True) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayerGraphClassification(final_outd * nheads, linear_dim, dropout=dropout, alpha=alpha, concat=False)

        if(readout=="sum"):
            self.readout= torch.sum
        elif(readout == "max"):   
            self.readout  = torch.max
        elif(readout == "mean"):
            self.readout = torch.mean
        
        self.linear = nn.Linear(linear_dim,nclass)

    def forward(self, graph):
        if(type(graph) is list): #for minibatch =1 
            graph = graph[0]
        x = F.dropout(graph.node_features, self.dropout, training=self.training)
        x = torch.cat([att(x,graph) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x,graph))
        x = self.readout(x,dim=0)
        x = self.linear(x)
        x = x.unsqueeze(dim=0)
        return F.log_softmax(x, dim=1)


#baselines
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
 
