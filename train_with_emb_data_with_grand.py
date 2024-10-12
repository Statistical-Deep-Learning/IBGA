import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import  numpy as np
from data import load_data, preprocess_features, preprocess_adj, load_data_all, normalize_adj, load_data_for_idx
from utils import masked_loss, masked_acc, sparse_dropout, assign_labels, assign_labels2, run_kmeans, one_hot_encode, get_mutual_information
from utils import compute_ib, get_grad_IB_B
import warnings
import argparse
import random
import time
import scipy.sparse as sp
from torch_scatter import scatter


warnings.filterwarnings("ignore")

import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import mode
import torch
from sklearn.cluster import KMeans
from precompute import propagation


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=True,
                 featureless=False):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.bias = bias

        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, x, support):
        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)
        if self.bias is not None:
            out += self.bias

        if self.activation:
            out = F.relu(out)

        return out

class GraphConvolution_b1in(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation=True,
                 featureless=False):
        super(GraphConvolution_b1in, self).__init__()
        self.dropout = dropout
        self.bias = bias

        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.num_features_nonzero = num_features_nonzero

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, x, support, B_1):
        if self.training and self.is_sparse_inputs:
            x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, self.dropout)

        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)
            else:
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight

        out = torch.sparse.mm(support, xw)
        if self.bias is not None:
            out += self.bias

        Z_1 = torch.mm(B_1, out)    #b1in

        if self.activation:
            output = F.relu(Z_1)
            return output, Z_1

        return Z_1, Z_1

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        self.attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(self.attention, Wh)

        if self.concat:
            return F.elu(h_prime), h_prime, Wh
        else:
            return h_prime, h_prime, Wh

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GIB(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero, args):
        super(GIB, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden
        self.dropoutgcn = args.dropoutgcn
        self.dropoutgat = args.dropoutgat
        self.B_1 = None
        self.f0mlp = args.f0mlp
        self.b1in = args.b1in    # if True, use B_1 is in gcnlayer, before compute Z_1 before relu

        #print('input dim:', input_dim)
        #print('output dim:', output_dim)
        #print('num_features_nonzero:', num_features_nonzero)
        self.gat = GraphAttentionLayer(self.input_dim, self.hidden_dim,
                                       dropout=self.dropoutgat,
                                       alpha=0.2, concat=True)

        if self.b1in:
            self.gcn_1 = GraphConvolution_b1in(self.hidden_dim, self.output_dim,
                                          num_features_nonzero,
                                    activation=True,
                                    dropout=self.dropoutgcn,
                                    is_sparse_inputs=False)
        else:
            self.gcn_1 = GraphConvolution(self.hidden_dim, self.output_dim,
                                          num_features_nonzero,
                                    activation=True,
                                    dropout=self.dropoutgcn,
                                    is_sparse_inputs=False)
        self.fc_0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.fc_0 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        #self.fc_1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        #self.fc_2 = nn.Linear(self.hidden_dim, self.output_dim)


    def forward(self, x, support, y, C_b_prime_np, update=True):
        if self.B_1 is None:
            self.B_1 = torch.eye(x.shape[0]).to(x.device)
        gatoutput, Z_0, F_0 = self.gat(x, support)
        B_0 = self.gat.attention#.cpu().detach().numpy()
        #print(B_0.shape)
        #print(f"F_0 shape: {F_0.shape}")
        #print(f"B_0 shape: {B_0.shape}")
        self.B_1 = self.B_1.detach()
        if update:
            #update B_1
                if self.f0mlp:
                    #F_0 = self.fc_1(F.relu(self.fc_0(F_0)))
                    F_0 = self.fc_0(F_0)
                grad_IB_B0 = get_grad_IB_B(x, Z_0, y, F_0, C_b_prime_np)
                B_1 = B_0 - grad_IB_B0
                #notmalize
                norms = B_1.norm(p=2, dim=1, keepdim=True)

                # Normalize each row to have unit norm
                #detach()?
                self.B_1 = B_1 / norms
                #print(f"F_1 shape: {F_1.shape}")
                #print(f"B_1 shape: {B_1.shape}")
        #print(f"B_1: {self.B_1[0]}")
        if self.b1in:
            #compute Z_1 in GCN
            output, Z_1 = self.gcn_1(gatoutput, support, self.B_1)
        else:
            #compute Z_1 out of GCN
            F_1 = self.gcn_1(gatoutput, support)
            Z_1 = torch.mm(self.B_1, F_1)
            output = Z_1
        #print(f"Z_1:{Z_1}")
        #Z_1_norms = Z_1.norm(p=2, dim=1, keepdim=True).detach() + 1e-10
        #Z_1 = Z_1/Z_1_norms
        #print(f"Z_1_norm:{Z_1}")
        #output = self.fc_2(Z_1)
        return output, Z_1, Z_0

    def l2_loss(self):
        loss = None
        #for p in self.gcn_0.parameters():
        #    if loss is None:
        #        loss = p.pow(2).sum()
        #    else:
        #        loss += p.pow(2).sum()
        for p in self.gcn_1.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()
        return loss



def sample_nodes(labels, num_samples):
    # return the indices of the sampled nodes
    num_classes = np.max(labels) + 1
    sampled_indices = []
    for i in range(num_classes):
        indices = np.where(labels == i)[0]
        sampled_indices.append(np.random.choice(indices, num_samples))
    return np.concatenate(sampled_indices)

# Write a function that samples 100 nodes from each class
def conditioned_sample_nodes(labels, num_samples, generated_neighbor, low_margin=2, high_margin=5):
    # return the indices of the sampled nodes
    num_classes = np.max(labels) + 1
    sampled_indices = []
    for i in range(num_classes):
        indices = np.where(labels == i)[0]
        for id in indices:
            if np.sum(generated_neighbor[id]) < low_margin or np.sum(generated_neighbor[id]) > high_margin:
                continue
            sampled_indices.append(id)
            if len(sampled_indices) == num_samples:
                break
    return sampled_indices

def add_sythnetic_node_to_graph(adj, generated_neighbors, threshold=1e-6):
    ori_num = adj.shape[0]
    new_num = generated_neighbors.shape[0]

    N = ori_num + new_num  
    extended_adjacency_matrix = np.zeros((N, N))
    extended_adjacency_matrix[:ori_num, :ori_num] = adj

    for new_node_idx in range(new_num):
        for old_node_idx in range(ori_num):
            # Since it's an undirected graph, we update both corresponding entries in the matrix
            if generated_neighbors[new_node_idx, old_node_idx] > 1-threshold:
                extended_adjacency_matrix[ori_num + new_node_idx, old_node_idx] = 1
                extended_adjacency_matrix[old_node_idx, ori_num + new_node_idx] = 1
    return extended_adjacency_matrix

def random_prop( feats, mat_scores, mat_idx, dropnode_rate):
        
        mat_scores = F.dropout(mat_scores, p=dropnode_rate, training=True)
        propagated_logits = scatter(feats * mat_scores[:, None], mat_idx[:, None],
                                    dim=0, dim_size=mat_idx[-1] + 1, reduce='sum')
        mat_sum_s = scatter(mat_scores[:,None], mat_idx[:,None],
                                    dim=0, dim_size=mat_idx[-1] + 1, reduce='sum')
        return propagated_logits / (mat_sum_s + 1e-12)

def sample_unlabel(idx_unlabel, unlabel_batch_size, shuffle=False):
    unlabel_numSamples = idx_unlabel.shape[0]
    indices = np.arange(unlabel_numSamples)
    if shuffle:
        np.random.shuffle(indices)
    excerpt = indices[:unlabel_batch_size]
    return idx_unlabel[excerpt]

def consis_loss(args, logps, tem, conf):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)

    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        if args.loss == 'kl':
            loss += torch.mean((-sharp_p * torch.log(p)).sum(1)[avg_p.max(1)[0] > conf])
        elif args.loss == 'l2':
            loss += torch.mean((p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
        else:
            raise ValueError(f"Unknown loss type: {args.loss}")
    loss = loss/len(ps)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--model', default='gcn')
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropoutgcn', type=float, default=0.0, help='Dropout for gcn layer.')
    parser.add_argument('--dropoutgat', type=float, default=0.5, help='Dropout for gat layer.')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--update_frequency', type=int, default=1)
    parser.add_argument('--tildes', type=int, default=0)
    parser.add_argument('--f0mlp', type=int, default=1)
    parser.add_argument('--b1in', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--log_each_epoch', action='store_true', help="If present, store loss for each epoch.")
    parser.add_argument('--save_ib', action='store_true', help="If present, store ib for each epoch.")
    parser.add_argument('--ps_labels_path', type=str, default="/home/local/ASUAD/changyu2/GIB/ps_labels_with_gt_gat_cora_83.npy")
    parser.add_argument('--emb_path', type=str, default="/home/local/ASUAD/changyu2/GIB/all_embs.npy")
    parser.add_argument('--log_path', type=str, default="")
    parser.add_argument('--gen_emb_path', type=str, default="/home/local/ASUAD/changyu2/GIB/cora_latents_8988_gvae_50000_64_decode_feat.npy")
    parser.add_argument('--gen_label_path', type=str, default="/home/local/ASUAD/changyu2/GIB/labels_8988_diffusion_3000_1.8.npy")
    parser.add_argument('--gen_neibor_path', type=str, default="/home/local/ASUAD/changyu2/GIB/cora_latents_8988_gvae_50000_64_decode_map.npy")
    parser.add_argument('--unlabel_num', type=int, default=-1, help="unlabeled node num (|U'|) for consistency regularization")
    parser.add_argument('--rmax', type=float, default=1e-7, help='GFPush threshold')
    parser.add_argument('--top_k', type=int, default=32, help='top neirghbor num')
    parser.add_argument('--sample', type=int, default=2, help='augmentation times per batch')
    parser.add_argument('--dropnode_rate', type=float, default=0.5, help='dropnode rate (1 - keep probability)')
    parser.add_argument('--prop_mode', type=str, default="ppr", help='propagation matrix $\Pi$, ppr or avg or single')
    parser.add_argument('--order', type=int, default=10, help='propagation step N')
    parser.add_argument('--lam', type=float, default=1, help='Lamda')
    parser.add_argument('--alpha', type=float, default=0.2, help='ppr teleport')
    parser.add_argument('--loss', type=str, default="l2", help="consistency loss function, l2 or kl")
    parser.add_argument('--tem', type=float, default=0.1, help='sharpening temperature')
    args = parser.parse_args()
    print(args)
    #set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    setting = f"w{args.warmup}_t{args.tildes}_gcnout{args.dropoutgcn}_gatout{args.dropoutgat}_lr{args.learning_rate}_wd{args.weight_decay}_hid{args.hidden}_K{args.sample}_lam{args.lam}_dropnode{args.dropnode_rate}_tem{args.tem}"
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = f"{setting}.txt"
    # load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all = load_data_all(args.dataset)
    #get the idx
    idx_train, idx_val, idx_test = load_data_for_idx(args.dataset)

    gt_labels = y_all.argmax(axis=1)
    #features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
    #supports = preprocess_adj(adj)

    device = torch.device('cuda')
    train_label = torch.from_numpy(y_train).long().to(device)
    num_classes = train_label.shape[1]
    train_label = train_label.argmax(dim=1)
    train_mask = torch.from_numpy(train_mask.astype(np.int64)).to(device)
    val_label = torch.from_numpy(y_val).long().to(device)
    val_label = val_label.argmax(dim=1)
    val_mask = torch.from_numpy(val_mask.astype(np.int64)).to(device)
    test_label = torch.from_numpy(y_test).long().to(device)
    test_label = test_label.argmax(dim=1)
    test_mask = torch.from_numpy(test_mask.astype(np.int64)).to(device)
    #extend train_mask with 1400 1s
    

    #i_f = torch.from_numpy(features[0]).long().to(device)
    #v_f = torch.from_numpy(features[1]).to(device)
    #feature = torch.sparse_coo_tensor(i_f.t(), v_f, features[2]).to_dense().to(device)
    #feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

    #i_s = torch.from_numpy(supports[0]).long().to(device)
    #v_s = torch.from_numpy(supports[1]).to(device)
    #support = torch.sparse_coo_tensor(i_s.t(), v_s, supports[2]).float().to_dense().to(device)
    
    #num_features_nonzero = torch.sparse.FloatTensor(i_f.t(), v_f, features[2])._nnz()


    num_features_nonzero = 0

    #use embedding as the feature
    embs = np.load(args.emb_path)
    #feature = torch.from_numpy(feature).float().to(device)
    feat_dim = embs.shape[1]
    # load synthetic data
    #generated_neighbors = np.load(args.gen_neibor_path)
    #generated_labels = np.load(args.gen_label_path)
    #generated_embeds = np.load(args.gen_emb_path)
    #here we use psudo labels generated by gcn/gat
    y_ps = np.load(args.ps_labels_path)
    y_ps_onehot = one_hot_encode(y_ps, num_classes)

    '''
    sample_per_class = 200
    # sampled_indices = conditioned_sample_nodes(generated_labels, sample_per_class, generated_neighbors)
    sampled_indices = sample_nodes(generated_labels, sample_per_class)
    sampled_generated_embeds = generated_embeds[sampled_indices]
    sampled_generated_labels = generated_labels[sampled_indices]
    sampled_generated_neighbors = generated_neighbors[sampled_indices]

    all_embs = np.concatenate((embs, sampled_generated_embeds), axis=0)
    all_labels_gt = np.concatenate((gt_labels, sampled_generated_labels), axis=0)
    all_labels_ps = np.concatenate((y_ps, sampled_generated_labels), axis=0)
    all_labels_gt_torch = torch.LongTensor(all_labels_gt).to(device)
    all_labels_ps_onehot = one_hot_encode(all_labels_ps, num_classes)

    new_adj = add_sythnetic_node_to_graph(adj.todense(), sampled_generated_neighbors, 1e-6)
    new_adj = new_adj + sp.eye(new_adj.shape[0])
    new_adj_crs = sp.csr_matrix(new_adj)
    norm_new_adj = normalize_adj(new_adj)

    #get adj and feature in tensor
    support = torch.from_numpy(norm_new_adj.todense()).float().to(device)
    feature = torch.from_numpy(all_embs).float().to(device)

    #get new masks
    train_mask = torch.cat((train_mask, torch.ones(sampled_indices.shape[0]).to(device).long()), dim=0)
    val_mask = torch.cat((val_mask, torch.zeros(sampled_indices.shape[0]).to(device).long()), dim=0)
    test_mask = torch.cat((test_mask, torch.zeros(sampled_indices.shape[0]).to(device).long()), dim=0)
    '''
    all_labels_gt_torch = torch.LongTensor(gt_labels).to(device)

    #get adj and feature in tensor
    norm_adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    support = torch.from_numpy(norm_adj.todense()).float().to(device)
    feature = torch.from_numpy(embs).float().to(device)

    adj_crs = sp.csr_matrix(adj + sp.eye(adj.shape[0]))

    #get augment features    
    unlabel_num = len(idx_test)
    time_s1 = time.time()
    idx_sample = np.random.permutation(
        idx_test)[:unlabel_num]
    idx_unlabel = np.concatenate([idx_val, idx_sample]) 
    #idx_train_new = np.concatenate([idx_train, np.arange(features.shape[0], features.shape[0]+len(sampled_indices))])

    idx_train_unlabel = np.concatenate(
        [idx_train, idx_unlabel])
    indptr = np.array(adj_crs.indptr, dtype=np.int32)
    indices = np.array(adj_crs.indices, dtype=np.int32)
    graph = propagation.Graph(indptr, indices, args.seed)
    row_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    col_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    mat_value = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.float64)
    if args.prop_mode == 'avg':
        coef = list(np.ones(args.order + 1, dtype=np.float64))
    elif args.prop_mode == 'ppr':
        coef = [args.alpha]
        for i in range(args.order):
            coef.append(coef[-1] * (1-args.alpha))
    elif args.prop_mode == 'single':
        coef = list(np.zeros(args.order + 1, dtype=np.float64))
        coef[-1] = 1.0
    else:
        raise ValueError(f"Unknown propagation mode: {args.prop_mode}")
    print(f"propagation matrix: {args.prop_mode}")
    coef = np.asarray(coef) / np.sum(coef)
    graph.gfpush_omp(idx_train_unlabel, row_idx, col_idx, mat_value, coef, args.rmax, args.top_k)
    
    topk_adj = sp.coo_matrix((mat_value, (row_idx, col_idx)), (
        feature.shape[0], feature.shape[0]))
    topk_adj = topk_adj.tocsr()
    time_preprocessing = time.time() - time_s1
    print(f"preprocessing done, time: {time_preprocessing}")
    #features_np = features
    #features, labels = totensor(features, labels)
    #n_class = labels.max().item() + 1

    #unlabel_index_batch = sample_unlabel(idx_sample, len(idx_test), shuffle=True)   #use all test data
    #batch_index = np.concatenate((idx_train, unlabel_index_batch))
    batch_topk_adj = topk_adj#[batch_index]

    source_idx, neighbor_idx = batch_topk_adj.nonzero()
    mat_scores = batch_topk_adj.data
    batch_feat = feature[neighbor_idx].to(device)
    mat_scores = torch.tensor(mat_scores, dtype=torch.float32).to(device)
    source_idx = torch.tensor(source_idx, dtype=torch.long).to(device)
    
    aug_feature_list = []
    K = args.sample


    #net = GIB(feat_dim, num_classes, num_features_nonzero, args)
    net = GIB(feat_dim, num_classes, num_features_nonzero, args)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    net.train()
    acc_test_history=[]

    best_acc = 0

    train_loss_hist = []
    test_loss_hist = []
    acc_hist = []
    ib0_hist = []
    ib1_hist = []

    #if args.tildes == 1:
    #    feature = torch.mm(support, feature)
    #elif args.tildes == 2:
    #    feature = torch.mm(support, torch.mm(support, feature))



    X_np = feature.cpu().numpy()
    C_b_prime_np_origin = run_kmeans(X_np, y_ps, num_classes) #no change

    for epoch in range(args.epochs):
        #t1 = time.time()
        torch.autograd.set_detect_anomaly(True)
        net.train()
        if epoch>= args.warmup and epoch%args.update_frequency ==0:
            update=True
        else:
            update=False
        #print(f"update:{update}")

        output_list = []
        loss = 0.
        for i in range(K):
            batch_feat_aug = random_prop(batch_feat, mat_scores, source_idx, args.dropnode_rate).detach()
            C_b_prime_np = run_kmeans(batch_feat_aug.cpu().numpy(), y_ps, num_classes)
            out, Z_1, Z_0 = net(batch_feat_aug, support, y_ps, C_b_prime_np, update=update)
            loss += masked_loss(out, all_labels_gt_torch, train_mask)
            log_softmax_out = torch.log_softmax(out, dim=-1)
            output_list.append(log_softmax_out[(1-train_mask).bool()])
        loss = loss/K
        args.conf = 2./num_classes
        con_loss = consis_loss(args, output_list, args.tem, args.conf)
        loss += args.lam * con_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        net.eval()

        out, Z_1, Z_0  = net(feature, support, y_ps, C_b_prime_np_origin, update=update)
        val_loss = masked_loss(out, all_labels_gt_torch, val_mask)
        test_loss = masked_loss(out, all_labels_gt_torch, test_mask)
        acc_train = masked_acc(out, all_labels_gt_torch, train_mask)
        acc_val = masked_acc(out, all_labels_gt_torch, val_mask)
        acc_test = masked_acc(out, all_labels_gt_torch, test_mask)
        if args.log_each_epoch:
            with open(log_path, 'a') as file:
                file.write(f"Epoch:{epoch+1}, train loss: {loss.item()}, val loss: {val_loss.item()}, test loss:{test_loss.item()}, train acc: {acc_train.item()}, val acc: {acc_val.item()}, test acc: {acc_test.item()} \n")
        
        if args.save_ib:
            Z_1 = Z_1.cpu().data.numpy()
            Z_0 = Z_0.cpu().data.numpy()
            input_feat = feature.to_dense().cpu().data.numpy()
            
            ib_0, MI_X_Z0, MI_Z0_Y,  = compute_ib(input_feat, Z_0, y_ps, num_classes, y_ps)
            ib_1, MI_X_Z1, MI_Z1_Y,  = compute_ib(input_feat, Z_1, y_ps, num_classes, y_ps)
            with open(log_path, 'a') as file:
                file.write(f"ib_0:{ib_0}, ib_1: {ib_1} \n")
        #print('Epoch:', epoch+1, 'ib_0:', '%.6f' % ib_0, 'ib_1', '%.6f' % ib_1, 'MI_X_Z0:', '%.6f' % MI_X_Z0, 'MI_Z0_Y:', '%.6f' % MI_Z0_Y, 'MI_X_Z1:', '%.6f' % MI_X_Z1, 'MI_Z1_Y:', '%.6f' % MI_Z1_Y)



        train_loss_hist.append(loss.item())
        test_loss_hist.append(test_loss.item())
        acc_hist.append(acc_test.item())
        acc_test_history.append(acc_test.item())
        acc_test_all = np.array(acc_test_history)
        #ib1_hist.append(ib_1)
        #if np.max(acc_test_all) > best_acc:
        #    best_acc = np.max(acc_test_all)
        #    np.save('gcn_out_'+args.dataset+'.npy', out.cpu().data.numpy())
        #    torch.save(net, save_path)
        #t2 = time.time()
        #print(t2-t1)
        if epoch%10 == 0:
            if epoch> args.warmup:
                print('Epoch:', epoch+1, 'train loss:', '%.6f' % loss.item(), 'test loss:', '%.6f' % test_loss.item(), 'con loss:', '%.6f' % con_loss.item(), 'train acc:', '%.4f' % acc_train.item(), 'val acc:', '%.4f' % acc_val.item(), 'test acc:', '%.4f' % acc_test.item(), 'best acc:', '%.4f' % max(acc_test_history[args.warmup:]))
            else:
                print('Epoch:', epoch+1, 'train loss:', '%.6f' % loss.item(), 'test loss:', '%.6f' % test_loss.item(), 'con loss:', '%.6f' % con_loss.item(), 'train acc:', '%.4f' % acc_train.item(), 'val acc:', '%.4f' % acc_val.item(), 'test acc:', '%.4f' % acc_test.item(), 'best acc:', '%.4f' % max(acc_test_history))

    net.eval()
    out, Z_1, Z_0 = net(feature, support, y_ps, C_b_prime_np_origin, update=True)
    # out = out[0]
    acc = masked_acc(out, all_labels_gt_torch, test_mask)
    acc_test_all = np.array(acc_test_history)
    with open(log_path, 'a') as file:
        file.write(f'args: {setting}, seed: {args.seed}, best test acc:{max(acc_test_history[args.warmup:])} \n')
        if args.save_ib:
            file.write(f"min ib_0:{min(ib0_hist)}, min ib_1:{min(ib1_hist)} \n")

if __name__ == '__main__':
    main()