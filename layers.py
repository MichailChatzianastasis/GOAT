import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import itertools
import os 
from random import shuffle

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, dataset="cora", store_att_weights=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dataset = dataset
        self.store_att_weights = store_att_weights

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj,epoch=-1):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)

        if(self.training == True and self.store_att_weights == True):
            if(self.concat == True):
                layer = 1
            else:
                layer = 2
            with open(f"./attentions/{self.dataset}/gat_attention_{epoch}_layer_{layer}","wb") as handle:
                pickle.dump(attention[:100].detach().to("cpu").numpy(),handle)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class LSTMAggregation(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dataset):
        super(LSTMAggregation, self).__init__()
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

        self.lstm = nn.LSTM(in_features,
                            out_features,
                            1,
                            bidirectional=True,
                            batch_first=True) 

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        self.nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
        self.src_nodes_dim = 0  # position of source nodes in edge index
        self.trg_nodes_dim = 1  # position of target nodes in edge index
        
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def forward(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes):
        
        self.src_nodes_dim = 0  # position of source nodes in edge index
        self.trg_nodes_dim = 1  # position of target nodes in edge index
        self.nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
        src_nodes_index = edge_index[self.src_nodes_dim]

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted_weighted.index_select(self.nodes_dim, src_nodes_index)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)
        
        indices = [[0],[int(self.seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(self.seq_length[:-1]):
            ind_scatter.append(list(range(index*self.maximum_neighbors,index*self.maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(self.seq_length[index+1]+current))

        k = zip(indices[0], indices[1])

        ind_scatter.append(list(range((len(self.seq_length)-1)*self.maximum_neighbors,(len(self.seq_length)-1)*self.maximum_neighbors+self.seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to(in_nodes_features.device)

        new_attentions = torch.full((num_of_nodes*self.maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*self.maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,self.maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   

        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])
        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])
        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)
        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn


class GraphAttentionLayerGraphClassification(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device="cuda"):
        super(GraphAttentionLayerGraphClassification, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device = device

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, graph):
        adj = graph.adj.to(self.device)
        h = h.to(self.device)
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GraphAttentionLayerOrdered(nn.Module):

    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Instead of aggregating the neighbors representations, we pass them to an LSTM based on the ordering of their attention scores.
    we may use another attention mechanism to output ordering scores.
    """
    def __init__(self, in_features, out_features, lstm_out_features, dropout, alpha, concat=True, dataset="cora", epoch=0, max_mean = False, store_att_weights=False, rnn_agg ="lstm"):
        super(GraphAttentionLayerOrdered, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        #self.out_features = out_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.max_mean = max_mean
        self.dataset = dataset
        self.store_att_weights = store_att_weights
        self.lstm_out_features = lstm_out_features

        self.W = nn.Parameter(torch.empty(size=(in_features, self.out_features)))
        self.b = nn.Parameter(torch.empty(size=(1,self.out_features)))

        self.W2 = nn.Parameter(torch.empty(size=(self.out_features, self.out_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        #self.W3 = nn.Parameter(torch.empty(size=(256, out_features)))
        #nn.init.xavier_uniform_(self.W3.data, gain=1.414)

        #self.mlp = MLP([in_features]+h_sizes)

        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
   
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bidirectional = True
        if(rnn_agg =="lstm"):
            self.lstm = nn.LSTM(self.out_features,self.lstm_out_features,1,bidirectional=self.bidirectional,batch_first=True)
        elif(rnn_agg =="gru"):
            self.lstm = nn.GRU(self.out_features,self.lstm_out_features,1,bidirectional=self.bidirectional,batch_first=True)
        elif(rnn_agg =="rnn"):
            self.lstm = nn.RNN(self.out_features,self.lstm_out_features,1,bidirectional=self.bidirectional,batch_first=True)
        
        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 

            with open(f'{dataset}_seq_length',"wb") as handle:
                pickle.dump(self.seq_length,handle)
        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
        if(False):
            with open(f'./{dataset}_attentions',"rb") as handle:
                self.attention = pickle.load(handle)
       
    def forward(self, h, adj, epoch=0):
        
        #Wh = self.leakyrelu(torch.mm(h, self.W2)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h, self.W)

        #Wh = torch.mm(Wh,self.W2)
        #Wh = self.mlp(h)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        #print(self.edge_dict[0])
        #print(e[0][0].item(),e[0][8].item(),e[0][14].item(),e[0][258].item(),e[0][435].item(),e[0][544].item())
        zero_vec = -9e15*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #print(attention)
        #store attention weights
        if(self.training == True and self.store_att_weights == True):
            nodes_to_save = 15
            if(self.concat == True):
                layer = 1
            else:
                layer = 2
            with open(f"./attentions/{self.dataset}/attention_{epoch}_layer_{layer}","wb") as handle:
            #with open(f"./{self.dataset}_attentions","wb") as handle:

                pickle.dump(attention[:nodes_to_save].detach().to("cpu").numpy(),handle)
                #pickle.dump(attention,handle)

        #h_prime = torch.matmul(attention, Wh)
        sorted_att,sorted_indices = torch.sort(attention,dim=1, descending=True)
        lstm_input = Wh[sorted_indices] 

        #print("indices",sorted_indices.shape) #[2708,2708]
        #print("Wh",Wh.shape) # [2708, 8]
        #print("lstm",lstm_input.shape) # [2708, 2708, 8]

        #lstm_input_final = torch.matmul(sorted_att,lstm_input)
        #lstm_input = lstm_input * sorted_att[:,:,None]
        # pack it
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.lstm(pack)  #for lstm
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.lstm(pack)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.lstm(pack)  # for gru
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.add(hn_mean,hn_max)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.lstm(pack)
        hn = torch.squeeze(hn)
        return hn
        #if self.concat:
            #return F.elu(hn)
        #else:
        #    return hn


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayerOrdered2(nn.Module):

    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Instead of aggregating the neighbors representations, we pass them to an LSTM based on the ordering of their attention scores.
    we may use another attention mechanism to output ordering scores.
    """
    def __init__(self, in_features, out_features, outd_2, dropout, alpha, h_sizes=[8], concat=True, dataset="cora", max_mean = False):
        super(GraphAttentionLayerOrdered2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        #self.out_features = out_features
        self.out_features = h_sizes[-1]
        self.outd_2 = outd_2
        self.alpha = alpha
        self.concat = concat
        self.max_mean = max_mean
        
        self.W = nn.Parameter(torch.empty(size=(in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        #self.W3 = nn.Parameter(torch.empty(size=(256, out_features)))
        #nn.init.xavier_uniform_(self.W3.data, gain=1.414)

        #self.mlp = MLP([in_features]+h_sizes)

        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
   
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bidirectional = True
        self.lstm = nn.LSTM(self.out_features,self.outd_2,1,bidirectional=self.bidirectional,batch_first=True)

        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 

        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
        
       

       
    def forward(self, h, adj):
        
        #Wh = self.leakyrelu(torch.mm(h, self.W2)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(h,self.W)
        #Wh = self.mlp(h)
        #Wh = h
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, Wh)
        sorted_att,sorted_indices = torch.sort(attention,dim=1, descending=True)
        lstm_input = Wh[sorted_indices] 

        #print("indices",sorted_indices.shape) #[2708,2708]
        #print("Wh",Wh.shape) # [2708, 8]
        #print("lstm",lstm_input.shape) # [2708, 2708, 8]

        #lstm_input_final = torch.matmul(sorted_att,lstm_input)
        #lstm_input = lstm_input * sorted_att[:,:,None]
        # pack it
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.lstm(pack)  #for lstm
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.lstm(pack)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.lstm(pack)  # for gru
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.add(hn_mean,hn_max)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.lstm(pack)
        hn = torch.squeeze(hn)
        if self.concat:
            return F.elu(hn)
        else:
            return hn


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionV2LayerOrdered(nn.Module):

    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Instead of aggregating the neighbors representations, we pass them to an LSTM based on the ordering of their attention scores.
    we may use another attention mechanism to output ordering scores.
    """
    def __init__(self, in_features, out_features, outd_2, dropout, alpha, concat=True, dataset="cora", max_mean = False):
        super(GraphAttentionV2LayerOrdered, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.max_mean = max_mean
        #self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(out_features*2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
   
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bidirectional = True
        self.lstm = nn.GRU(out_features,outd_2,1,bidirectional=self.bidirectional,batch_first=True)

        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 

        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
        
       

       
    def forward(self, h, adj,epoch):
        
        Wh = torch.mm(h, self.W2) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        #Wh = self.leakyrelu(torch.mm(Wh, self.W3))

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = torch.matmul(self.leakyrelu(a_input), self.a).squeeze(2)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, Wh)
        sorted_att,sorted_indices = torch.sort(attention,dim=1, descending=True)
        lstm_input = Wh[sorted_indices] 

        #print("indices",sorted_indices.shape) #[2708,2708]
        #print("Wh",Wh.shape) # [2708, 8]
        #print("lstm",lstm_input.shape) # [2708, 2708, 8]

        #lstm_input_final = torch.matmul(sorted_att,lstm_input)
        #lstm_input = lstm_input * sorted_att[:,:,None]
        # pack it
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.lstm(pack)  #for lstm
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.lstm(pack)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.lstm(pack)  # for gru
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.lstm(pack)
        hn = torch.squeeze(hn)
        if self.concat:
            return F.elu(hn)
        else:
            return hn


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GraphAttentionLayerOrdered_Shared_LSTM(nn.Module):

    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Instead of aggregating the neighbors representations, we pass them to an LSTM based on the ordering of their attention scores.
    we may use another attention mechanism to output ordering scores.
    """
    def __init__(self, shared_lstm, in_features, out_features, outd_2, dropout, alpha, concat=True, dataset="cora", max_mean = False):
        super(GraphAttentionLayerOrdered_Shared_LSTM, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.max_mean = max_mean
        #self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.shared_lstm = shared_lstm


        self.W2 = nn.Parameter(torch.empty(size=(in_features, 256)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.W3 = nn.Parameter(torch.empty(size=(256, out_features)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
   
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bidirectional = True

        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 

        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
        
       

       
    def forward(self, h, adj):
        
        Wh = self.leakyrelu(torch.mm(h, self.W2)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.mm(Wh, self.W3)

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, Wh)
        
        sorted_att,sorted_indices = torch.sort(attention,dim=1, descending=True)
        lstm_input = Wh[sorted_indices] 

        #print("indices",sorted_indices.shape) #[2708,2708]
        #print("Wh",Wh.shape) # [2708, 8]
        #print("lstm",lstm_input.shape) # [2708, 2708, 8]

        #lstm_input_final = torch.matmul(sorted_att,lstm_input)
        #lstm_input = lstm_input * sorted_att[:,:,None]
  
        # pack it
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.shared_lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.shared_lstm(pack)  #for lstm
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.shared_lstm(pack)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.shared_lstm(pack)  # for gru
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.shared_lstm(pack)
        hn = torch.squeeze(hn)
        if self.concat:
            return F.elu(hn)
        else:
            return hn


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class MLP(nn.Module):
    def __init__(self, h_sizes):
        super(MLP, self).__init__()
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = 0.5

        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
        
    
    def forward(self,x):
        for layer in self.hidden:
            x = layer(x)
            x = F.dropout(x, self.dropout, training=self.training)

        return x

class GraphAttentionLayerRandomOrdered(nn.Module):

    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Instead of aggregating the neighbors representations, we pass them to an LSTM based on the ordering of their attention scores.
    we may use another attention mechanism to output ordering scores.
    """
    def __init__(self, in_features, out_features, outd_2, dropout, alpha, concat=True, dataset="cora", max_mean = False):
        super(GraphAttentionLayerRandomOrdered, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.max_mean = max_mean
        self.indices = []
        #self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        #nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.empty(size=(in_features, 128)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.W4 = nn.Parameter(torch.empty(size=(128, out_features)))
        nn.init.xavier_uniform_(self.W4.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bidirectional = True
        self.lstm = nn.LSTM(out_features,outd_2,1,bidirectional=self.bidirectional,batch_first=True)
        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 

        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
               
    def forward(self, h, adj):
        Wh = self.leakyrelu(torch.mm(h, self.W2)) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = self.leakyrelu(torch.mm(Wh, self.W4)) # h.shape: (N, in_features), Wh.shape: (N, out_features)

        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, Wh)
        
        sorted_att,sorted_indices = torch.sort(attention,dim=1, descending=True)
        if(self.indices == []):
            self.indices = self.random_permutation(sorted_indices)

        lstm_input = Wh[self.indices] 

        #print("indices",sorted_indices.shape) #[2708,2708]
        #print("Wh",Wh.shape) # [2708, 8]
        #print("lstm",lstm_input.shape) # [2708, 2708, 8]

        #lstm_input_final = torch.matmul(sorted_att,lstm_input)
        #lstm_input = lstm_input * self.indices[:,:,None]

        # pack it
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.lstm(pack)  #for lstm
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.lstm(pack)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.lstm(pack)  # for gru
                if(self.max_mean):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.lstm(pack)
        hn = torch.squeeze(hn)
        if self.concat:
            return F.elu(hn)
        else:
            return hn


    def random_permutation(self,sorted_indices):
        indices = sorted_indices.detach().clone()
        for i,element in enumerate(self.seq_length):
            rnd_idx = torch.randperm(element)
            indices[i,:element] = indices[i,:element].view(-1)[rnd_idx].view(indices[i,:element].size())
        return indices

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayerOurs(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayerOurs, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        #self.W_hidden = nn.Parameter(torch.empty(size=(2*out_features,out_features))) #ours
        self.lstm = nn.LSTM(out_features,out_features,batch_first=True)

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        #self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        self.a = nn.Parameter(torch.empty(size=(3*out_features, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
            
        Wh_reapeated = torch.unsqueeze(Wh,0)
        Wh_reapeated = Wh.expand(Wh_reapeated.shape[1],-1,-1) 

        zero_like_vec = torch.zeros_like(Wh_reapeated)

        Wh_reapeated = torch.where(torch.unsqueeze(adj,-1) > 0, Wh_reapeated, zero_like_vec)
        Wh_final
        for i in range(Wh_reapeated.shape[0]): #for every node 
        #construct dictionary with neighbors of every node (indices)
        #select only these indexes and padd the rest with 0
            Wh_reapeated[i,self.indices[i]]
        
        


        #hidden_states,last_hidden_state = self.lstm(Wh_reapeated)




        a_input_modified = torch.cat((a_input,hidden_states),-1)
       
        e = self.leakyrelu(torch.matmul(a_input_modified, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2       
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1   
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



##### IMPLEMENTATION 4

class GATOrderedLayerIMPL4(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, dropout, alpha,
                concat=True, dataset="cora", activation=nn.ELU(),
                add_skip_connection=True, bias=True, log_attention_weights=False, adj=None):

        super().__init__()

        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.dropout = dropout
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        
        self.bidirectional = True
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        self.lstm = nn.LSTM(num_out_features,num_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)

        if(os.path.isfile(f"{dataset}_edge_dict") == True):
            with open(f'{dataset}_edge_dict',"rb") as handle:
                print(dataset)
                self.edge_dict = pickle.load(handle)
            
            self.seq_length = []   
            for index,values in self.edge_dict.items():
                self.seq_length.append(len(values)) 
        
        else:
            with open(f'./{dataset}_seq_length',"rb") as handle:
                self.seq_length = pickle.load(handle)
         
        with open(f"./graph_lengths/{dataset}_maximum_neighbors.pkl",'rb') as handle:
            self.maximum_neighbors = pickle.load(handle)
        
        self.init_params()
        
    def forward(self, in_nodes_features, adj, edge_index):
        #
        # Step 1: Linear Projection + regularization
        #

        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #GATV2
        nodes_features_proj = self.leakyReLU(nodes_features_proj)

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)


        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        #GATV1
        #scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        #GATV2
        scores_per_edge = scores_source_lifted + scores_target_lifted

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
      
        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        #nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        
        #CAREFULL!!! THIS IS THE MAIN PART THAT WE CHANGE IN OUR MODEL
        # Instead of aggregating the features from the neighbors using max,mean pooling, We pass them in an Lstm
        # using the ordering from the attention mechanism.
        out_nodes_features = self.aggregate_ordering_neighbors(nodes_features_proj_lifted_weighted, attentions_per_edge, edge_index, in_nodes_features, num_of_nodes)
        #out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    
    def aggregate_ordering_neighbors(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes):
        
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)

        indices = [[0],[int(self.seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(self.seq_length[:-1]):
            ind_scatter.append(list(range(index*self.maximum_neighbors,index*self.maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(self.seq_length[index+1]+current))

        k = zip(indices[0], indices[1])
        
        ind_scatter.append(list(range((len(self.seq_length)-1)*self.maximum_neighbors,(len(self.seq_length)-1)*self.maximum_neighbors+self.seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to(in_nodes_features.device)

        ###  
        ''' THIS WORKS but we must find something more efficient

        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        for att_head in range(self.num_of_heads):
            for counter,(i,j) in enumerate(k):                       #for every node
                #sort the edges of every node based on attention
                sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j,att_head],dim=0)
                lstm_input = rearranged_node_features[i:j,att_head][idx]
                lstm_input = lstm_input.squeeze().unsqueeze(dim=0)
                output, (hn,cn) = self.lstm(lstm_input)
                hn = hn[-1].squeeze()
                out_nodes_features[counter,att_head,:] = hn

        return out_nodes_features
        '''
        ###

        '''
        THIS WORKS BUT WE NEED SOMTHING MORE QUICK
        out_nodes_features = torch.zeros((num_of_nodes,rearranged_node_features.shape[-1]), dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        lstm_input_features = torch.zeros((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        for counter,(i,j) in enumerate(k):                       #for every node
            #sort the edges of every node based on attention
            sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j],dim=0)
            lstm_input = rearranged_node_features[i:j][idx]
            lstm_input = lstm_input.squeeze().unsqueeze(dim=0) #[1,neighbors_of_node,features]
            #pad the lstm to shape [1,maximum_neighbors,features] 
            lstm_input_features[counter,:lstm_input.shape[1]] = lstm_input
        
        lstm_input_features = lstm_input_features.to(in_nodes_features.device)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn

        '''
        new_attentions = torch.full((num_of_nodes*self.maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*self.maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,self.maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   


 
        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])

        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))

        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])

        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn



    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).reshape(-1, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class GATOrderedLayerIMPL4arxiv_ogbn(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, lstm_out_features, dropout, alpha,
                concat=True, dataset="cora", activation=nn.LeakyReLU(),
                add_skip_connection=True, bias=True, log_attention_weights=False, adj=None,rnn_agg="lstm"):

        super().__init__()

        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.dropout = dropout
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        
        self.bidirectional = True
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        if(rnn_agg=="lstm"):
            self.lstm = nn.LSTM(num_out_features,lstm_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)
        elif(rnn_agg=="gru"):
            self.lstm = nn.GRU(num_out_features,lstm_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)
        elif(rnn_agg=="rnn"):
            self.lstm = nn.RNN(num_out_features,lstm_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)

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
        
        self.init_params()
        
    def forward(self, in_nodes_features, adj, edge_index):
        #
        # Step 1: Linear Projection + regularization
        #
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        
        #GATV2
        #nodes_features_proj = self.leakyReLU(nodes_features_proj)
        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)


        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        #scores_per_edge = scores_source_lifted + scores_target_lifted

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
        #
        # Step 3: Neighborhood aggregation
        #
        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        #nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        
        #CAREFULL!!! THIS IS THE MAIN PART THAT WE CHANGE IN OUR MODEL
        # Instead of aggregating the features from the neighbors using max,mean pooling, We pass them in an Lstm
        # using the ordering from the attention mechanism.
        out_nodes_features = self.aggregate_ordering_neighbors(nodes_features_proj_lifted_weighted, attentions_per_edge, edge_index, in_nodes_features, num_of_nodes)
        #out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    
    def aggregate_ordering_neighbors(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes):
        

        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)

        indices = [[0],[int(self.seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(self.seq_length[:-1]):
            ind_scatter.append(list(range(index*self.maximum_neighbors,index*self.maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(self.seq_length[index+1]+current))

        k = zip(indices[0], indices[1])

        ind_scatter.append(list(range((len(self.seq_length)-1)*self.maximum_neighbors,(len(self.seq_length)-1)*self.maximum_neighbors+self.seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to('cuda')
        ###  
        ''' THIS WORKS but we must find something more efficient

        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        for att_head in range(self.num_of_heads):
            for counter,(i,j) in enumerate(k):                       #for every node
                #sort the edges of every node based on attention
                sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j,att_head],dim=0)
                lstm_input = rearranged_node_features[i:j,att_head][idx]
                lstm_input = lstm_input.squeeze().unsqueeze(dim=0)
                output, (hn,cn) = self.lstm(lstm_input)
                hn = hn[-1].squeeze()
                out_nodes_features[counter,att_head,:] = hn


        return out_nodes_features
        '''
        ###

        '''
        THIS WORKS BUT WE NEED SOMTHING MORE QUICK
        out_nodes_features = torch.zeros((num_of_nodes,rearranged_node_features.shape[-1]), dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        lstm_input_features = torch.zeros((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        for counter,(i,j) in enumerate(k):                       #for every node
            #sort the edges of every node based on attention
            sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j],dim=0)
            lstm_input = rearranged_node_features[i:j][idx]
            lstm_input = lstm_input.squeeze().unsqueeze(dim=0) #[1,neighbors_of_node,features]
            #pad the lstm to shape [1,maximum_neighbors,features] 
            lstm_input_features[counter,:lstm_input.shape[1]] = lstm_input
        
        lstm_input_features = lstm_input_features.to(in_nodes_features.device)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn

        '''
        new_attentions = torch.full((num_of_nodes*self.maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*self.maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,self.maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   

        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])
        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])
        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)
        if( type(self.lstm).__name__ == "LSTM"):
         output, (hn,cn) = self.lstm(pack)
        else:
            output,hn = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn



    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).reshape(-1, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features = out_nodes_features + self.bias
            
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class GATOrderedLayerIMPL4_graph_classification(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, dropout, alpha,
                concat=True, dataset="cora", activation=nn.ELU(),
                add_skip_connection=False, bias=True, log_attention_weights=False, adj=None ,device ="cuda"):

        super().__init__()

        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.dropout = dropout
        self.device = device
        self.multiply_attention = True
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        
        self.bidirectional = True
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        self.lstm = nn.LSTM(num_out_features,num_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)
        self.init_params()
        
    def forward(self, x, graph):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features = x.to(self.device)
        adj = graph.adj.to(self.device)
        edge_index = graph.edge_mat.to(self.device)
        seq_length = [ len(i) for i in graph.neighbors]
        maximum_neighbors = graph.max_neighbor


        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)


        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
      
        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        
        if(self.multiply_attention):
            nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        else:
            nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        
        #CAREFULL!!! THIS IS THE MAIN PART THAT WE CHANGE IN OUR MODEL
        # Instead of aggregating the features from the neighbors using max,mean pooling, We pass them in an Lstm
        # using the ordering from the attention mechanism.
        out_nodes_features = self.aggregate_ordering_neighbors(nodes_features_proj_lifted_weighted, attentions_per_edge, edge_index, in_nodes_features, num_of_nodes,seq_length,maximum_neighbors)
        #out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    
    def aggregate_ordering_neighbors(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes,seq_length,maximum_neighbors):
        
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)

        indices = [[0],[int(seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(seq_length[:-1]):
            ind_scatter.append(list(range(index*maximum_neighbors,index*maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(seq_length[index+1]+current))

        k = zip(indices[0], indices[1])
        
        ind_scatter.append(list(range((len(seq_length)-1)*maximum_neighbors,(len(seq_length)-1)*maximum_neighbors+seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to(in_nodes_features.device)

        ###  
        ''' THIS WORKS but we must find something more efficient

        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        for att_head in range(self.num_of_heads):
            for counter,(i,j) in enumerate(k):                       #for every node
                #sort the edges of every node based on attention
                sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j,att_head],dim=0)
                lstm_input = rearranged_node_features[i:j,att_head][idx]
                lstm_input = lstm_input.squeeze().unsqueeze(dim=0)
                output, (hn,cn) = self.lstm(lstm_input)
                hn = hn[-1].squeeze()
                out_nodes_features[counter,att_head,:] = hn

        return out_nodes_features
        '''
        ###

        '''
        THIS WORKS BUT WE NEED SOMTHING MORE QUICK
        out_nodes_features = torch.zeros((num_of_nodes,rearranged_node_features.shape[-1]), dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        lstm_input_features = torch.zeros((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        for counter,(i,j) in enumerate(k):                       #for every node
            #sort the edges of every node based on attention
            sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j],dim=0)
            lstm_input = rearranged_node_features[i:j][idx]
            lstm_input = lstm_input.squeeze().unsqueeze(dim=0) #[1,neighbors_of_node,features]
            #pad the lstm to shape [1,maximum_neighbors,features] 
            lstm_input_features[counter,:lstm_input.shape[1]] = lstm_input
        
        lstm_input_features = lstm_input_features.to(in_nodes_features.device)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn

        '''
        new_attentions = torch.full((num_of_nodes*maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   


 
        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])

        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,maximum_neighbors,rearranged_node_features.shape[-1]))

        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])

        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn



    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).reshape(-1, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)



class GATIMP4layer(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, dropout, alpha,
                concat=True, dataset="cora", activation=nn.ELU(),
                add_skip_connection=True, bias=True, log_attention_weights=False, adj=None):

        super().__init__()

        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.dropout = dropout
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        
        self.bidirectional = True
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        self.lstm = nn.LSTM(num_out_features,num_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)

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
        
        self.init_params()
        
    def forward(self, in_nodes_features, adj, edge_index):
        #
        # Step 1: Linear Projection + regularization
        #

       
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)


        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        #nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        
        #CAREFULL!!! THIS IS THE MAIN PART THAT WE CHANGE IN OUR MODEL
        # Instead of aggregating the features from the neighbors using max,mean pooling, We pass them in an Lstm
        # using the ordering from the attention mechanism.
        #out_nodes_features = self.aggregate_ordering_neighbors(nodes_features_proj_lifted_weighted, attentions_per_edge, edge_index, in_nodes_features, num_of_nodes)
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    
    def aggregate_ordering_neighbors(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes):
        
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)

        indices = [[0],[int(self.seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(self.seq_length[:-1]):
            ind_scatter.append(list(range(index*self.maximum_neighbors,index*self.maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(self.seq_length[index+1]+current))

        k = zip(indices[0], indices[1])
        
        ind_scatter.append(list(range((len(self.seq_length)-1)*self.maximum_neighbors,(len(self.seq_length)-1)*self.maximum_neighbors+self.seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to(in_nodes_features.device)
        ###  
        ''' THIS WORKS but we must find something more efficient

        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        for att_head in range(self.num_of_heads):
            for counter,(i,j) in enumerate(k):                       #for every node
                #sort the edges of every node based on attention
                sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j,att_head],dim=0)
                lstm_input = rearranged_node_features[i:j,att_head][idx]
                lstm_input = lstm_input.squeeze().unsqueeze(dim=0)
                output, (hn,cn) = self.lstm(lstm_input)
                hn = hn[-1].squeeze()
                out_nodes_features[counter,att_head,:] = hn


        return out_nodes_features
        '''
        ###

        '''
        THIS WORKS BUT WE NEED SOMTHING MORE QUICK
        out_nodes_features = torch.zeros((num_of_nodes,rearranged_node_features.shape[-1]), dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        lstm_input_features = torch.zeros((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        for counter,(i,j) in enumerate(k):                       #for every node
            #sort the edges of every node based on attention
            sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j],dim=0)
            lstm_input = rearranged_node_features[i:j][idx]
            lstm_input = lstm_input.squeeze().unsqueeze(dim=0) #[1,neighbors_of_node,features]
            #pad the lstm to shape [1,maximum_neighbors,features] 
            lstm_input_features[counter,:lstm_input.shape[1]] = lstm_input
        
        lstm_input_features = lstm_input_features.to(in_nodes_features.device)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn

        '''
        new_attentions = torch.full((num_of_nodes*self.maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*self.maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,self.maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   

        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])
        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])
        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)
        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn



    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).reshape(-1, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)





class GATOrderedLayerIMPL4arxiv_ogbn_node_batching(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
    # We'll use these constants in many functions so just extracting them here as member fields
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, dropout, alpha,
                concat=True, dataset="cora", activation=nn.ELU(),
                add_skip_connection=True, bias=True, log_attention_weights=False, adj=None):

        super().__init__()

        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection
        self.dropout = dropout
        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1,num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1,num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #
        
        self.bidirectional = True
        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here
        
        self.lstm = nn.LSTM(num_out_features,num_out_features,1,
                            bidirectional=self.bidirectional,batch_first=True)

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
        
        self.init_params()
        
    def forward(self, in_nodes_features, adj, edge_index):
        #
        # Step 1: Linear Projection + regularization
        #

       
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        #
        # Step 2: Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)


        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (E, NH, 1)
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)
        #
        # Step 3: Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        #nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted
        
        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        
        #CAREFULL!!! THIS IS THE MAIN PART THAT WE CHANGE IN OUR MODEL
        # Instead of aggregating the features from the neighbors using max,mean pooling, We pass them in an Lstm
        # using the ordering from the attention mechanism.
        out_nodes_features = self.aggregate_ordering_neighbors(nodes_features_proj_lifted_weighted, attentions_per_edge, edge_index, in_nodes_features, num_of_nodes)
        #out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Step 4: Residual/skip connections, concat and bias
        #

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return out_nodes_features

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    
    def aggregate_ordering_neighbors(self, nodes_features_proj_lifted_weighted,attentions_per_edge, 
                                     edge_index, in_nodes_features, num_of_nodes):
        
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted)
                                 
        trg_sorted, inds = torch.sort(trg_index_broadcasted[:,0],dim = 0) 
        rearranged_node_features = nodes_features_proj_lifted_weighted[inds]   
        rearranged_attentions_per_edge = attentions_per_edge[inds]        
        
    
        #print("nodes",rearranged_node_features.shape)
        #print("edges",rearranged_attentions_per_edge.shape)

        indices = [[0],[int(self.seq_length[0])]]
        ind_scatter = []
        current = 0
        for index,element in enumerate(self.seq_length[:-1]):
            ind_scatter.append(list(range(index*self.maximum_neighbors,index*self.maximum_neighbors+element)))
            indices[0].append(int(element+current))
            current+=element
            indices[1].append(int(self.seq_length[index+1]+current))

        k = zip(indices[0], indices[1])
        
        ind_scatter.append(list(range((len(self.seq_length)-1)*self.maximum_neighbors,(len(self.seq_length)-1)*self.maximum_neighbors+self.seq_length[-1]))) #because last element was not added
        ind_scatter = torch.tensor(list(itertools.chain.from_iterable(ind_scatter))).unsqueeze(-1).to(in_nodes_features.device)
        ###  
        ''' THIS WORKS but we must find something more efficient

        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        for att_head in range(self.num_of_heads):
            for counter,(i,j) in enumerate(k):                       #for every node
                #sort the edges of every node based on attention
                sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j,att_head],dim=0)
                lstm_input = rearranged_node_features[i:j,att_head][idx]
                lstm_input = lstm_input.squeeze().unsqueeze(dim=0)
                output, (hn,cn) = self.lstm(lstm_input)
                hn = hn[-1].squeeze()
                out_nodes_features[counter,att_head,:] = hn


        return out_nodes_features
        '''
        ###

        
        #THIS WORKS BUT WE NEED SOMTHING MORE QUICK
        out_nodes_features = torch.zeros((num_of_nodes,rearranged_node_features.shape[-1]), dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        lstm_input_features = torch.zeros((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        for counter,(i,j) in enumerate(k):                       #for every node
            #sort the edges of every node based on attention
            sorted_att, idx = torch.sort(rearranged_attentions_per_edge[i:j],dim=0)
            lstm_input = rearranged_node_features[i:j][idx]
            lstm_input = lstm_input.squeeze().unsqueeze(dim=0) #[1,neighbors_of_node,features]
            #pad the lstm to shape [1,maximum_neighbors,features] 
            lstm_input_features[counter,:lstm_input.shape[1]] = lstm_input
            lstm_input_features = lstm_input_features.to(in_nodes_features.device)
            output, (hn,cn) = self.lstm(lstm_input_features)

        lstm_input_features = lstm_input_features.to(in_nodes_features.device)
        #pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)

        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn

        
        new_attentions = torch.full((num_of_nodes*self.maximum_neighbors,1),-1,dtype=rearranged_attentions_per_edge.dtype).to(in_nodes_features.device)

        lstm_input_features = torch.zeros((num_of_nodes*self.maximum_neighbors,rearranged_node_features.shape[-1]),
                                            device =in_nodes_features.device, dtype=in_nodes_features.dtype)
    


        new_attentions = new_attentions.scatter_(0,ind_scatter,rearranged_attentions_per_edge)
        new_attentions = new_attentions.reshape((num_of_nodes,self.maximum_neighbors))
        sorted_att, idx = torch.sort(new_attentions,dim=1,descending=True)   

        ind_scatter = ind_scatter.expand(-1,rearranged_node_features.shape[-1])
        lstm_input_features = lstm_input_features.scatter_(0,ind_scatter,rearranged_node_features)
        lstm_input_features = lstm_input_features.reshape((num_of_nodes,self.maximum_neighbors,rearranged_node_features.shape[-1]))
        idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1,rearranged_node_features.shape[-1])
        lstm_input_features = torch.zeros(lstm_input_features.shape,device=lstm_input_features.device,dtype=lstm_input_features.dtype).scatter_(1,idx,lstm_input_features)
        pack = torch.nn.utils.rnn.pack_padded_sequence(lstm_input_features, self.seq_length, batch_first=True, enforce_sorted=False)
        output, (hn,cn) = self.lstm(pack)
        hn = hn[-1].squeeze()
        return hn



    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).reshape(-1, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)




class Lstm_graph_pooling(torch.nn.Module):
    """
    Implementation #4 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric

    But, it's hopefully much more readable! (and of similar performance)

    """
    
   

    def __init__(self, input_dim, output_dim, dropout, bidirectional, max_mean_lstm, dataset, device="cuda"):

        super().__init__()

        self.bidirectional = bidirectional
        self.num_out_features = output_dim
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim,output_dim,1,bidirectional=self.bidirectional,batch_first=True)
        self.device = device

    def forward(self, x, graph):
        #
        # Step 1: Linear Projection + regularization
        #
        in_nodes_features = x.to(self.device)
        adj = graph.adj.to(self.device)
        edge_index = graph.edge_mat.to(self.device)

        num_of_nodes = in_nodes_features.shape[0]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps

        in_nodes_features = F.dropout(in_nodes_features, self.dropout, training=self.training).unsqueeze(dim=0)
        if( type(self.lstm).__name__ == "LSTM"):
            if(self.bidirectional):
                hidden_states, (hn,cn) = self.lstm(in_nodes_features)  #for lstm
                if(self.max_mean_lstm):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    #hn = torch.cat((hn[0],hn[1]),-1)
                    hn = hn[-1]
            else:
                hidden_states, (hn,cn) = self.lstm(in_nodes_features)
        else:
            if(self.bidirectional):
                hidden_states, hn = self.lstm(in_nodes_features)  # for gru
                if(self.max_mean_lstm):
                    hn_mean = torch.mean(hn, 0) #for bidirectional lstm
                    hn_max,_ = torch.max(hn,0) 
                    hn = torch.cat((hn_mean,hn_max),1)
                else:
                    hn = hn[-1]
            else:
                hidden_states, hn = self.lstm(in_nodes_features)
        hn = torch.squeeze(hn)
        return hn


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
#np.set_printoptions(threshold=np.nan)

class StructuralFingerprintLayer(nn.Module):
    """
    adaptive structural fingerprint layer
    """

    def __init__(self, in_features, out_features, dropout, alpha,adj_ad, concat=True,):
        super(StructuralFingerprintLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_ad=adj_ad
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)


    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N* N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        s=self.adj_ad
        e = e.cuda()
        s=s.cuda()

        # combine sij and eij
        e=abs(self.W_ei)*e+abs(self.W_si)*s

        zero_vec = -9e15*torch.ones_like(e)
        k_vec=-9e15*torch.ones_like(e)
        adj=adj.cuda()
        #np.set_printoptions(threshold=np.nan)
        attention = torch.where(adj>0 , e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




