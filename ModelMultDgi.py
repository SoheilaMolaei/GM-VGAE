import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import os
import numpy as np

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout,K):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        # self.gcc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        
        self.h2 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=lambda x: x)
        
        self.gc4 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=lambda x: x)
        self.gc5 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=lambda x: x)
        
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.mu_px = nn.ModuleList([GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)  for i in range(K)])
        self.logvar_px = nn.ModuleList([GraphConvolution(hidden_dim1, hidden_dim2,dropout, act=lambda x:x)  for i in range(K)])
        self.qz = GraphConvolution(hidden_dim1, K,dropout, act=lambda x:x)
        self.hidden_dim2=hidden_dim2
        self.K=K
    def priorGenerator(self, z_w,adj,batchSize):
      # batchSize = z_w.size(0)

      h = torch.tanh(self.h2(z_w,adj))
      device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
      mu_px = torch.empty(batchSize , self.hidden_dim2,self.K, device=device, requires_grad=False)
      logvar_px = torch.empty(batchSize, self.hidden_dim2,self.K, device=device, requires_grad=False)
      for i in range(self.K):
        mu_px[:,:,i] = self.mu_px[i](h,adj)
        logvar_px[:, :,i] = self.logvar_px[i](h,adj)

      return mu_px, logvar_px

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        # hidden2 = self.gcc2(hidden1, adj)
        fc_mu_x=self.gc2(hidden1, adj)
        fc_logvar_x=self.gc3(hidden1, adj)

        fc_mu_w=self.gc4(hidden1, adj)
        fc_logvar_w=self.gc5(hidden1, adj)
        return fc_mu_x, fc_logvar_x,fc_mu_w,fc_logvar_w,hidden1

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
      batchSize = x.size(0)
      mu_x, logvar_x,mu_w, logvar_w,hidden1 = self.encode(x, adj)

      z_x = self.reparameterize(mu_x, logvar_x)#X maqaleh
      z_w = self.reparameterize(mu_w, logvar_w)
      mu_px, logvar_px = self.priorGenerator(z_w,adj,batchSize)

      qz = F.softmax(self.qz(hidden1,adj), dim=1)#Z maqaleh

      return z_x, mu_x, logvar_x,z_w, mu_w, logvar_w,mu_px, logvar_px,qz


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

