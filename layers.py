import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
import torch.nn as nn

class GraphConvolution2(Module):
    def __init__(self, in_features, out_features, dropout,act=F.relu, alpha=0.2, bias=True):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        # print(x)
        Wh = torch.mm(x, self.weight) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        # e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj.to_dense()  > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.bias is not None:
            h_prime = h_prime + self.bias
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)
class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
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

    
class GraphGat(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout,act=F.relu, alpha=0.2, concat=True):
        super(GraphGat, self).__init__()
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
        edge = adj.to_dense().nonzero().t()

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
        
        output = self.act(h_prime)
        # if self.concat:
        #     # if this layer is not last layer,
        #     return F.elu(h_prime)
        # else:
        #     # if this layer is last layer,
        #     return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# class GraphConvolution(nn.Module):
#     def __init__(self, nfeat, nhid, dropout, act=F.relu, alpha=0.2, nheads=2,nclass=128):
#         """Sparse version of GAT."""
#         super(GraphConvolution, self).__init__()
#         self.dropout = dropout

#         self.attentions = [SpGraphAttentionLayer(nfeat, 
#                                                  nhid, 
#                                                  dropout=dropout, 
#                                                  alpha=alpha, 
#                                                  concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = SpGraphAttentionLayer(nhid * nheads, 
#                                              nclass, 
#                                              dropout=dropout, 
#                                              alpha=alpha, 
#                                              concat=False)

#     def forward(self, x, adj):
        
#         x = F.dropout(x, self.dropout, training=self.training)
#         print(x.shape,adj.shape,self.attentions)
#         x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, adj))
#         return x