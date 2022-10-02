from __future__ import division
from __future__ import print_function
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from ModelMultDgi import GCNModelVAE, GCNModelAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
from skipGram import SkipGram
from sklearn.cluster import KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import utils_data
import wandb

# wandb.init(project='texas')

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--lr_dw', type=float, default=0.01, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=0, help="whether to use context nodes for skipgram")
parser.add_argument('--n-clusters', default=7, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
args = parser.parse_args()

# from utils import load_npz
def preprocess_high_order_adj( adj, order, eps ):
    adj = row_normalize_adj( adj )

    adj_sum = adj
    cur_adj = adj
    for i in range( 1, order ):
        cur_adj = cur_adj.dot( adj )
        adj_sum += cur_adj
    adj_sum /= order

    adj_sum.setdiag( 0 )
    adj_sum.data[adj_sum.data<eps] = 0
    adj_sum.eliminate_zeros()

    adj_sum += sp.eye( adj.shape[0] )
    return sym_normalize_adj( adj_sum + adj_sum.T )
def row_normalize_adj( adj ):
    '''row normalize adjacency matrix'''

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_mat_inv = sp.diags( 1 / np.maximum( degree, np.finfo(float).eps ) )
    return d_mat_inv.dot( adj ).tocoo()  
def sym_normalize_adj( adj ):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.coo_matrix( adj )
    degree = np.array( adj.sum(1) ).flatten()
    d_inv_sqrt = np.power( np.maximum( degree, np.finfo(float).eps ), -0.5 )
    d_mat_inv_sqrt = sp.diags( d_inv_sqrt )
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
  
def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))


    g, features, true_labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset_str, None, 0.6, 0.2)
    adj=g.adj(scipy_fmt='coo')
    true_labels=true_labels.detach().numpy()


    args.n_clusters=true_labels.max()+1
    print(args.n_clusters,"ssssssss")
   
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)


    # Some preprocessing
    adj_norm = preprocess_graph(adj)


    adj_label = adj_train + sp.eye(adj_train.shape[0])

    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    z_x=torch.zeros(features.shape[0],args.hidden1)
    z_w=torch.zeros(features.shape[0],args.hidden2)
    z_shuffle=torch.cat((features,z_x,z_w),axis=1)
    n_nodes, feat_dim = z_shuffle.shape
    
    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout,args.n_clusters)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    z_x, mu_x, _,z_w, mu_w, _,_, logvar_px,qz = model(z_shuffle, adj_norm)
    z_shuffle=torch.cat((features,z_x.detach_(),z_w.detach_()),axis=1)
    hidden_emb = None
    for epoch in tqdm(range(args.epochs)):
        t = time.time()
        model.train()
        optimizer.zero_grad()


        z_x, mu_x, logvar_x,z_w, mu_w, logvar_w,mu_px, logvar_px,qz = model(z_shuffle, adj_norm)



        
        logvar_x1 = logvar_x.unsqueeze(-1)
        logvar_x1 = logvar_x1.expand(-1, args.hidden2, args.n_clusters)

        mu_x1 = mu_x.unsqueeze(-1)
        mu_x1 = mu_x1.expand(-1, args.hidden2, args.n_clusters)       
        if torch.cuda.is_available():       
          mu_x1=mu_x1.cuda()
          logvar_x1=logvar_x1.cuda()

        # KL( q(w) || p(w) )
        KLD_W = -0.5 / n_nodes* torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

        KLD_Z = -0.5 / n_nodes * torch.mean(torch.sum(1 +qz * torch.log(qz + 1e-10) , 1))


        KLD_QX_PX=loss_function(preds=model.dc(z_x), labels=adj_label,
                             mu=(mu_x1 - mu_px), logvar=(logvar_px - logvar_x1), n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        KLD_QX_PX = KLD_QX_PX = KLD_QX_PX.expand(n_nodes, 1, args.hidden2)
        E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz.unsqueeze(-1)/n_nodes))


        model.train()
        optimizer.zero_grad()
        lbl_1 = torch.ones(n_nodes)
        lbl_2 = torch.zeros(n_nodes)
        lbl = torch.cat((lbl_1, lbl_2))
        idx = np.random.permutation(n_nodes)

        shuf_fts = z_shuffle[idx,:]



        n_nodes1, feat_dim1 = z_shuffle.shape


        z_xL2, mu_xL2, logvar_xL2,z_wL2, mu_wL2, logvar_wL2,mu_pxL2, logvar_pxL2,qz2 = model(shuf_fts, adj_norm)
        KLD_W2 = -0.5 / n_nodes* torch.sum(1 + logvar_wL2 - mu_wL2.pow(2) - logvar_wL2.exp())
        KLD_Z2 = 0.5 / n_nodes * torch.mean(torch.sum(1 +qz2 * torch.log(qz2 + 1e-10) , 1))

        KLD_QX_PX2=loss_function(preds=model.dc(z_wL2), labels=adj_label,
                             mu=mu_wL2, logvar=logvar_wL2, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        KLD_QX_PX2 = KLD_QX_PX2.expand(n_nodes, 1, args.hidden2)
        E_KLD_QX_PX2 = torch.sum(torch.bmm(KLD_QX_PX2, qz2.unsqueeze(-1)/n_nodes))

        lossF = (1.0/loss_function(preds=model.dc(z_xL2), labels=adj_label,
                             mu=mu_xL2, logvar=logvar_xL2, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight))+\
                             (1.0/E_KLD_QX_PX2)+KLD_Z2+KLD_W2



        loss = loss_function(preds=model.dc(z_x), labels=adj_label,
                             mu=mu_x, logvar=logvar_x, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)+loss_function(preds=model.dc(z_w), labels=adj_label,
                             mu=mu_w, logvar=logvar_w, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight) +lossF+KLD_Z +E_KLD_QX_PX+KLD_W

        loss.backward(retain_graph=True)
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb =np.concatenate((mu_x.data.numpy(),mu_w.data.numpy()),axis=1) 


        roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)


        if (epoch + 1) % 10 == 0:
            tqdm.write("Evaluating intermediate results...")
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
            predict_labels = kmeans.predict(hidden_emb)

            pr=np.amax(kmeans.fit_transform(hidden_emb), axis=1)
            b_xent = nn.BCEWithLogitsLoss() 
            print(loss, lossF)           

            cm = clustering_metrics(true_labels, predict_labels)
            cm.evaluationClusterModelFromLabel(tqdm)
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            tqdm.write('ROC: {}, AP: {}'.format(roc_score, ap_score))

            print(loss, lossF)
            print("Kmeans ACC",purity_score(true_labels,predict_labels))


    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
    predict_labels = kmeans.predict(hidden_emb)
    cm = clustering_metrics(true_labels, predict_labels)
    cm.evaluationClusterModelFromLabel(tqdm)
    print("Kmeans ACC",purity_score(true_labels,predict_labels))

    if args.plot == 1:
        cm.plotClusters(tqdm, hidden_emb, true_labels)


if __name__ == '__main__':
    gae_for(args)
