import torch
import torch.nn as nn
# from layers import GCN, AvgReadout, Discriminator
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class DGIMAin(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGIMAin, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        
    def forward(self, seq1, seq2, seq3, adj, sparse, msk, samp_bias1, samp_bias2):

        h_1 = self.gcn(seq1, adj, sparse)
#         embeds=torch.reshape(seq1,[seq1.shape[1],seq1.shape[2]])
#         ipca = IncrementalPCA(n_components=n_h)
#         ipca.fit(embeds)
#         embeds = ipca.transform(embeds)
#         embeds=torch.tensor(embeds)
#         embeds=torch.reshape(embeds,[1,embeds.shape[0],embeds.shape[1]])
#         h_1=embeds.float()
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)
        
        h_3=self.gcn(seq3, adj, sparse)

        h_2=(h_2+h_3)/2
#         c1 = self.read(h_2, msk)
#         c1 = self.sigm(c1)
#         c=c1+c

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
#         from torch.autograd import Variable
#         cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         output=cos(seq1,seq2)
#         output=torch.cat((lbl_1,output),1)
#         output=Variable(output, requires_grad=True)

        return ret 
        # Detach the return variables
    def embed(self, seq, adj, sparse, msk):

        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach() 
class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        
        h_2 = self.gcn(seq2, adj, sparse)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
        return ret

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
class DGIDecoder(nn.Module):
    def __init__(self, n_in, n_h, activation,de_size):
        super(DGIDecoder, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.gcn1 = GCN(n_in, de_size, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        
    def forward(self, seq1, seq2, seq3, adj,lbl_1, sparse, msk, samp_bias1, samp_bias2):

        h_1 = self.gcn(seq1, adj, sparse)
#         embeds=torch.reshape(seq1,[seq1.shape[1],seq1.shape[2]])
#         ipca = IncrementalPCA(n_components=n_h)
#         ipca.fit(embeds)
#         embeds = ipca.transform(embeds)
#         embeds=torch.tensor(embeds)
#         embeds=torch.reshape(embeds,[1,embeds.shape[0],embeds.shape[1]])
#         h_1=embeds.float()
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, sparse)
        
        h_3=self.gcn(seq3, adj, sparse)
 
        h_2=(h_2+h_3)/2
        

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
#         from torch.autograd import Variable
#         cos = nn.CosineSimilarity(dim=2, eps=1e-6)
#         output=cos(seq1,seq2)
#         output=torch.cat((lbl_1,output),1)
#         output=Variable(output, requires_grad=True)

        return ret
        # Detach the return variables
    def embed(self, seq, adj, sparse, msk):

        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()  