# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 17:31:51 2023

@author: Mengjie Chen
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import Config
import math
import scipy.sparse as sp
from torch_scatter import scatter
# from torch_geometric.utils import softmax
import random

args = Config.parse()




class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, num_cellline, num_protein, embeddingSize, device):
        super(BioEncoder, self).__init__()
        self.device = device
        # -------drug_layer-------
        self.drug = nn.Linear(dim_drug, embeddingSize) 
        self._init_weights(self.drug)

        
        # -------cell line_layer-------
        self.cell1 = nn.Linear(dim_cellline, dim_cellline//2)
        self._init_weights(self.cell1)
        self.cell2 = nn.Linear(dim_cellline//2, embeddingSize)
        self._init_weights(self.cell2)

        # -------protein_layer-------
        self.protein = nn.Embedding(num_protein, embeddingSize)
        nn.init.kaiming_uniform_(self.protein.weight.data, nonlinearity='relu')
        proteinIndices = [i for i in range(num_protein)]
        self.proteinIndices = torch.from_numpy(np.asarray(proteinIndices)).long().to(self.device)
        
        
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0.0)

    def forward(self, Drug_Features, Cell_Line_Feature):
        # -------drug_layer-------
        x_drug = torch.relu(self.drug(Drug_Features))

        # -------cell line_layer-------
        x_cell =torch.relu(self.cell1(Cell_Line_Feature))
        x_cell =torch.relu(self.cell2(x_cell))

        # -------protein_layer-------
        x_protein = self.protein(self.proteinIndices)
        
        return x_drug, x_cell, x_protein


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



class rhgcnConv(nn.Module):
    def __init__(self, edge_num, Xe_class_length, degV_dict, in_channels, out_channels, num_edge_types=3, negative_slope=0.2, use_norm = True):
        super().__init__()

        self.W = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=True) for _ in range(num_edge_types + 1)])
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.num_edge_types = num_edge_types
        self.reset_parameters()

        self.edge_num = edge_num
        self.Xe_class_length = Xe_class_length
        self.use_norm = use_norm
        self.degV_dict = degV_dict


    def reset_parameters(self):
        for layer in self.W:
            nn.init.kaiming_uniform_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)


    def forward(self, X, vertex, edges):

        X0 = self.W[0](X)
        Xve = []
        for i in range(self.num_edge_types):
            Xve.append((self.W[i + 1](X)*self.degV_dict[i])[vertex])

        Xe1 = []
        for i in range(self.num_edge_types):
            Xe1.append(scatter(Xve[i], edges, dim = 0, reduce = 'mean', dim_size = self.edge_num))

        Xe = Xe1[0][:self.Xe_class_length[0],:]
        for i in range(self.num_edge_types - 1):
            Xe = torch.cat((Xe,Xe1[i+1][self.Xe_class_length[i]:self.Xe_class_length[i+1],:]),0)

        Xev = Xe[edges]
        
        Xv = scatter(Xev, vertex, dim = 0, reduce = 'sum', dim_size = X.shape[0])
              
        Xv = Xv + X0
        
        if self.use_norm:
            Xv = normalize_l2(Xv)

        Xv = self.leaky_relu(Xv)

        return Xv


class RHGNN(nn.Module):
    def __init__(self, V, E, edge_num, Xe_class_length, degV_dict, nfeat, nhid, out_dim, num_edge_types, dropout):

        super().__init__()
        self.conv_in = rhgcnConv(edge_num, Xe_class_length, degV_dict, nfeat, nhid, num_edge_types)
        self.conv_out1 = rhgcnConv(edge_num, Xe_class_length, degV_dict, nhid, out_dim, num_edge_types)
        self.V = V
        self.E = E
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, X):

        X = self.conv_in(X, self.V, self.E)
        X = self.dropout(X)
        X = self.conv_out1(X, self.V, self.E)

        return  X




class ChannelAttention(nn.Module):
    def __init__(self, emb_size):
        super(ChannelAttention, self).__init__()
        self.weights = nn.ParameterDict({
            'attention': nn.Parameter(torch.randn(1, emb_size)),
        })
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights['attention'])


    def forward(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(torch.matmul(self.weights['attention'], embedding.t())) 
                     
        score = F.softmax(torch.stack(weights, dim=1), dim=1)
        mixed_embeddings = torch.zeros_like(channel_embeddings[0])
        
        for i in range(len(weights)):
            mixed_embeddings += (score[:, i]*channel_embeddings[i].t()).t()

        return mixed_embeddings, score



class WeightLocalModule(nn.Module):
    def __init__(self, feature_dim):
        super(WeightLocalModule, self).__init__()
        self.weight_local = nn.Parameter(torch.Tensor(feature_dim, 1), requires_grad=True)
        nn.init.xavier_normal_(self.weight_local)

        
    def forward(self, x):
        return torch.matmul(x, torch.diag(self.weight_local.squeeze()))


class BilinearDecoder(nn.Module):
    def __init__(self, feature_dim = 300, numDrug = 38, cellscount = 39, use_bias = False):
        super(BilinearDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.cellscount = cellscount
        self.numDrug = numDrug

        self.weights_global = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        nn.init.kaiming_uniform_(self.weights_global)

        self.weights_local = nn.ParameterDict()
        for cellidx in range(self.numDrug, self.numDrug + self.cellscount):
            weight_local = nn.Parameter(torch.Tensor(feature_dim, 1))
            nn.init.kaiming_uniform_(weight_local)
            # nn.init.xavier_normal_(weight_local)
            self.weights_local[f'weights_local_{cellidx}'] = weight_local

    def forward(self, embedding, index):
        outputs = {}

        for cellidx in range(self.numDrug, self.numDrug + self.cellscount):    
            relation = torch.diag(self.weights_local[f'weights_local_{cellidx}'].squeeze())
            DrugA = embedding[index[cellidx][:,0]]
            DrugB = embedding[index[cellidx][:,1]]
            x = torch.einsum('bn,nm,mj,jk,bk->b', DrugA, relation, self.weights_global, relation, DrugB)
            outputs[cellidx] = x
            
        return outputs



class Synergy(nn.Module):
    def __init__(self, numDrug, BioEncoder, encoder1, encoder2, attention, decoder):
        super(Synergy, self).__init__()
        self.BioEncoder = BioEncoder
        self.hgnn_encoder1 = encoder1
        self.hgnn_encoder2 = encoder2
        self.attention = attention
        self.decoder = decoder
        self.numDrug = numDrug
        
    def forward(self, Drug_Features, Cell_Line_Feature, combination_index):
        x_drug, x_cell, x_protein = self.BioEncoder(Drug_Features, Cell_Line_Feature)
        hypergraph1 = torch.cat((x_drug, x_protein), 0)
        hypergraph2 = torch.cat((x_drug, x_cell), 0)
        embedding1 = self.hgnn_encoder1(hypergraph1)
        embedding2 = self.hgnn_encoder2(hypergraph2)


        embedding, _ = self.attention(*[embedding1[:self.numDrug,:], embedding2[:self.numDrug,:]])
        result = self.decoder(embedding, combination_index)
        return result, embedding1[self.numDrug:,:]





def preprocess_adj(adj, device):

    adj_normalizer = aug_normalized_adjacency
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    r_adj = r_adj.to(device)
    return r_adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor.
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()



class EdgeMask:

    def __init__(self, adj, nhid, device, mask_ratio):
        self.adj = adj
        self.masked_edges = None
        self.device = device

        self.cached_adj_norm = None
        self.pseudo_labels = None
        self.linear = nn.Linear(nhid, 2).to(device)
        self.mask_ratio = mask_ratio
        self.transform_data(self.mask_ratio)

    def transform_data(self, mask_ratio):
        # randomly mask edges

        if self.cached_adj_norm is None:
            nnz = self.adj.nnz
            perm = np.random.permutation(nnz)
            preserve_nnz = int(nnz*(1 - mask_ratio))
            masked = perm[preserve_nnz: ]
            self.masked_edges = (self.adj.row[masked], self.adj.col[masked])
            perm = perm[:preserve_nnz]
            r_adj = sp.coo_matrix((self.adj.data[perm],
                                   (self.adj.row[perm],
                                    self.adj.col[perm])),
                                  shape=self.adj.shape)
            r_adj = preprocess_adj(r_adj, self.device)
            self.cached_adj_norm = r_adj


    def make_loss(self, embeddings):
        # link prediction loss
        edges = self.masked_edges

        if self.pseudo_labels is None:
            self.pseudo_labels = np.zeros(2*len(edges[0]))
            self.pseudo_labels[: len(edges[0])] = 1
            self.pseudo_labels = torch.LongTensor(self.pseudo_labels).to(self.device)
            self.neg_edges = self.neg_sample(k=len(edges[0]))

        neg_edges = self.neg_edges
        node_pairs = np.hstack((np.array(edges), np.array(neg_edges).transpose()))
        self.node_pairs = node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)

        loss = F.nll_loss(output, self.pseudo_labels)

        return loss

    def neg_sample(self, k):
        nonzero = set(zip(*self.adj.nonzero()))
        edges = self.random_sample_edges(self.adj, k, exclude=nonzero)
        return edges

    def random_sample_edges(self, adj, n, exclude):
        # 'exclude' is a set which contains the edges we do not want to sample and the edges already sampled
        itr = self.sample_forever(adj, exclude=exclude)
        return [next(itr) for _ in range(n)]

    def sample_forever(self, adj, exclude):
        # 'exclude' is a set which contains the edges we do not want to sample and the edges already sampled
       
        while True:
            
            t = tuple(random.sample(range(0, adj.shape[0]), 2))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))