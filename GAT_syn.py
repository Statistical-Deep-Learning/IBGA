import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj, load_data_all, load_data_all_for_gat, normalize_adj
from    utils import masked_loss, masked_acc, sparse_dropout, accuracy
import warnings


import glob
import time
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable

warnings.filterwarnings("ignore")

import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import mode
import torch
from sklearn.cluster import KMeans


import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp

def assign_labels2(true_labels, cluster_labels):
    # Assume `cluster_labels` is the array of labels from the clustering algorithm
    # and `true_labels` is the array of ground truth labels


    # Build a confusion matrix
    size = max(cluster_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((size, size))

    for a, b in zip(cluster_labels, true_labels):
        cost_matrix[a, b] += 1

    # Convert the problem to a maximization problem since the Hungarian algorithm
    # minimizes the cost
    cost_matrix = cost_matrix.max() - cost_matrix

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # `row_ind` are the original cluster labels, `col_ind` are the best matching true labels
    # for minimal mismatch (maximal matching)
    matching = dict(zip(row_ind, col_ind))

    return matching

def run_kmeans(x, y, num_cluster, niter=100):
    #print('performing kmeans clustering')

    # Run kmeans to find centroids
    # centroids, distortion = kmeans(x, num_cluster, iter=niter)

    # # Assign samples to the nearest centroids
    # cluster_assignments, _ = vq(x, centroids)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0)
    cluster_assignments = kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_

    #unique_elements, counts = np.unique(cluster_assignments, return_counts=True)
    #print(f"cluster_cnt:{counts}")

    # Map each cluster to the most frequent class label and reorder centroids
    # 这里如果两个cluster的mode of lable 要是一致的话，后面的centroids会替换掉前面的
    reordered_centroids = np.zeros_like(centroids)
    assigned_label_cluster = assign_labels2(y, cluster_assignments)
    #assigned_label_cluster = assign_labels(y, cluster_assignments)
    reordered_centroids = np.zeros_like(centroids)
    for cluster, label in assigned_label_cluster.items():
    #for label, cluster in assigned_label_cluster:
        reordered_centroids[label] = centroids[cluster]
        #print(reordered_centroids)
    return reordered_centroids


def one_hot_encode(labels, num_classes):
    # Create a numpy array filled with zeros and of appropriate size
    one_hot = np.zeros((labels.shape[0], num_classes))
    # Use fancy indexing to place ones where the class label indicates
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


def get_mutual_information(p_1, p_2, p_12, eps=1e-10):
    # Add epsilon to avoid taking log(0)
    p_12 = p_12 + eps

    p1_p2 = np.outer(p_1, p_2)  # A x B
    p1_p2 = p1_p2 + eps  # Add epsilon to the denominator to prevent division by zero

    mi = np.sum(p_12 * np.log(p_12 / p1_p2))
    return mi

def get_clust_score(feat, centroids, beta=1.0):
    # compute cluster score for each feature
    # feat: N x D
    # centroids: K x D
    # return: N x K

    N = feat.shape[0]
    K = centroids.shape[0]
    score = np.zeros((N, K))
    feat_normalized = feat / np.linalg.norm(feat, axis=1, keepdims=True)  # Normalize features

    #这里的centroids 也应该normalize一下？

    for i in range(K):
        score[:, i] = np.linalg.norm(feat_normalized - centroids[i], axis=1) #** 2
    score = -beta * score
    score = np.exp(score)
    score /= np.sum(score, axis=1, keepdims=True)  # softmax
    return score
def sparse_tuple_to_dense(sparse_tuple):
    """Convert sparse tuple representation to dense tensor."""
    indices = torch.LongTensor(sparse_tuple[0]).t()
    #print(indices[:16])
    values = torch.tensor(sparse_tuple[1])
    #print(values[:16])
    shape = torch.Size(sparse_tuple[2])
    #print(shape)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.to_dense()

def compute_ib(input_feat, inter_feat, y_all_1d):
    #input_feat: input feature
    #inter_feat: output feature from one layer
    #y_all_1d: labels
    inter_centroids = run_kmeans(inter_feat, y_all_1d, num_classes)
    input_centroids = run_kmeans(input_feat, y_all_1d, num_classes)

    inter_score = get_clust_score(inter_feat, inter_centroids)
    input_score = get_clust_score(input_feat, input_centroids)

    clust_score_input =  get_clust_score(input_feat, input_centroids)
    p_in = np.sum(clust_score_input, axis=0)

    clust_score_output = get_clust_score(inter_feat, inter_centroids)
    p_out = np.sum(clust_score_output, axis=0)

    one_hot_target = y_all
    p_label = np.sum(one_hot_target, axis=0)

    p_in_out = np.sum(np.matmul(clust_score_input[:, :, np.newaxis],
                            clust_score_output[:, np.newaxis, :]), axis=0)

    p_out_label = np.sum(np.matmul(clust_score_output[:, :, np.newaxis],
                               one_hot_target[:, np.newaxis, :]), axis=0)

    p_in = p_in / np.sum(p_in)
    p_out = p_out / np.sum(p_out)
    p_label = p_label / np.sum(p_label)
    p_in_out = p_in_out / np.sum(p_in_out)
    p_out_label = p_out_label / np.sum(p_out_label)

    #print(f"p_in:{p_in}")
    #print(f"p_out:{p_out}")
    #print(f"p_label:{p_label}")
    #print(f"p_in_out:{p_in_out}")
    #print(f"p_out_label:{p_out_label}")

    MI_in_out = get_mutual_information(p_in, p_out, p_in_out)
    MI_out_label = get_mutual_information(p_out, p_label, p_out_label)
    information_bottleneck = MI_in_out - MI_out_label
    #print('The MI_in_out is: ', MI_in_out.item())
    #print('The MI_out_label is: ', MI_out_label.item())
    #print('The information bottleneck is: ', information_bottleneck)
    return information_bottleneck, MI_in_out, MI_out_label

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
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

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




class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        Z_0 = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(Z_0, adj))
        return F.log_softmax(x, dim=1), Z_0, x

if __name__ == '__main__':
    #Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=5000, help='Patience')

    parser.add_argument('--emb_path', type=str, default="/home/local/ASUAD/changyu2/GIB/data/cora_all_embs.npy")
    parser.add_argument('--log_path', type=str, default="/home/local/ASUAD/changyu2/GIB/gat_syn_log.txt")
    parser.add_argument('--gen_emb_path', type=str, default="/home/local/ASUAD/changyu2/GIB/data/cora_latents_8988_gvae_50000_64_decode_feat.npy")
    parser.add_argument('--gen_label_path', type=str, default="/home/local/ASUAD/changyu2/GIB/data/cora_labels_8988_diffusion_3000_1.8.npy")
    parser.add_argument('--gen_neibor_path', type=str, default="/home/local/ASUAD/changyu2/GIB/data/cora_latents_8988_gvae_50000_64_decode_map.npy")
    parser.add_argument('--sample_per_class', type=int, default=20)
    #args = parser.parse_args(['--dataset', 'citeseer'])
    args = parser.parse_args()
    #args.epochs = 67
    args.hidden = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #for seed in range(0, 20):
    seed = args.seed
    print(f"seed:{seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data_all_for_gat(args.dataset)
    adj = adj.todense()
    num_classes = labels.max().item()+1
    y_all = one_hot_encode(labels.numpy(), num_classes)
    
    """
    if args.cuda:
        model.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    """

    gt_labels = labels.numpy()
    #import synthetic nodes and labels and insert synthetic nodes into 
    #use embedding as the feature
    embs = np.load(args.emb_path)
    #feature = torch.from_numpy(feature).float().to(device)
    feat_dim = embs.shape[1]
    # load synthetic data
    generated_neighbors = np.load(args.gen_neibor_path)
    generated_labels = np.load(args.gen_label_path)
    generated_embeds = np.load(args.gen_emb_path)


    sample_per_class = args.sample_per_class
    # sampled_indices = conditioned_sample_nodes(generated_labels, sample_per_class, generated_neighbors)
    sampled_indices = sample_nodes(generated_labels, sample_per_class)
    sampled_generated_embeds = generated_embeds[sampled_indices]
    sampled_generated_labels = generated_labels[sampled_indices]
    sampled_generated_neighbors = generated_neighbors[sampled_indices]

    all_embs = np.concatenate((embs, sampled_generated_embeds), axis=0)
    all_labels_gt = np.concatenate((gt_labels, sampled_generated_labels), axis=0)
    all_labels_gt_torch = torch.LongTensor(all_labels_gt).to(device)

    new_adj = add_sythnetic_node_to_graph(adj, sampled_generated_neighbors, 1e-6)
    norm_new_adj = normalize_adj(new_adj + sp.eye(new_adj.shape[0]))

    #get adj and feature in tensor
    support = torch.from_numpy(norm_new_adj.todense()).float().to(device)
    feature = torch.from_numpy(all_embs).float().to(device)

    #concatenate the idx_train with new indexes
    idx_train = torch.cat((idx_train, torch.LongTensor(np.arange(len(gt_labels), len(gt_labels)+len(sampled_indices)))),0)


    features, adj, labels = Variable(feature), Variable(support), Variable(all_labels_gt_torch)

    # Model and optimizer
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    else:
        model = GAT(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=args.nb_heads,
                    alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(),
                        lr=args.lr,
                        weight_decay=args.weight_decay)
    model = model.to(device)


    def train(epoch):

        t = time.time()
        model.train()
        optimizer.zero_grad()
        output, Z_0, Z_1 = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output, Z_0, Z_1 = model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if (epoch+1) % 10000 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.data.item()),
            'acc_train: {:.4f}'.format(acc_train.data.item()),
            'loss_val: {:.4f}'.format(loss_val.data.item()),
            'acc_val: {:.4f}'.format(acc_val.data.item()),
            'time: {:.4f}s'.format(time.time() - t))
            #compute_test()
            #compute IB
            #Z_1 = Z_1.cpu().data.numpy()
            #Z_0 = Z_0.cpu().data.numpy()
            #input_feat = features.cpu().data.numpy()
            #y_all_1d = labels.cpu().numpy()
            #ib_0, MI_X_Z0, MI_Z0_Y,  = compute_ib(input_feat, Z_0, y_all_1d)
            #ib_1, MI_X_Z1, MI_Z1_Y,  = compute_ib(input_feat, Z_1, y_all_1d)
            #print('Epoch:', epoch+1, 'ib_0:', '%.6f' % ib_0, 'ib_1', '%.6f' % ib_1, 'MI_X_Z0:', '%.6f' % MI_X_Z0, 'MI_Z0_Y:', '%.6f' % MI_Z0_Y, 'MI_X_Z1:', '%.6f' % MI_X_Z1, 'MI_Z1_Y:', '%.6f' % MI_Z1_Y)



        return loss_val.data.item()


    def compute_test():
        model.eval()
        output, Z_0, Z_1 = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        '''
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))
        '''
        return acc_test

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    test_acc_hist = []
    for epoch in range(args.epochs):
        #print(epoch)
        loss_values.append(train(epoch))

        #torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        test_acc = compute_test()
        test_acc_hist.append(test_acc)


    print(f"best acc:{max(test_acc_hist)}")
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    #print('Loading {}th epoch'.format(best_epoch))
    #model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    #compute_test()
    with open(args.log_path, 'a') as file:
        file.write(f'sample_per_class: {sample_per_class}, seed: {seed},  best test acc:{max(test_acc_hist)} \n')