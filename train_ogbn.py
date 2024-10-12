import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import  numpy as np
from data import load_data, preprocess_features, preprocess_adj, load_data_all, load_data_cs
from utils import masked_loss, masked_acc, sparse_dropout, assign_labels, assign_labels2, run_kmeans, one_hot_encode, get_mutual_information
from utils import compute_ib, get_grad_IB_B, get_grad_IB_B_large, get_clust_score_pytorch, get_U
import warnings
import argparse
import random
import time

warnings.filterwarnings("ignore")
import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import mode
import torch
from sklearn.cluster import KMeans
import scipy.sparse as sp
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader, ImbalancedSampler

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

class GIB_large(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero, args):
        super(GIB_large, self).__init__()

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


    def forward(self, x, support, y, C_b_prime_np, Q, update=True):
        gatoutput, Z_0, F_0 = self.gat(x, support)
        B_0 = self.gat.attention#.cpu().detach().numpy()
        self.B_1 = torch.eye(x.shape[0]).to(x.device)       ######initialize B_1, if update is false, we can use this for warmup
        if update:
            #update B_1
                if self.f0mlp:
                    #F_0 = self.fc_1(F.relu(self.fc_0(F_0)))
                    F_0 = self.fc_0(F_0)
                grad_IB_B0, Z0_clust_score = get_grad_IB_B_large(x, Z_0, y, F_0, C_b_prime_np, Q)
                B_1 = B_0 - grad_IB_B0
                #notmalize
                norms = B_1.norm(p=2, dim=1, keepdim=True)

                # Normalize each row to have unit norm
                #detach()?
                self.B_1 = B_1 / norms
        else:
            num_classes = len(np.unique(y))
            Z_np = Z_0.detach().cpu().numpy() #calculate each step
            C_a_np = run_kmeans(Z_np, y, num_classes)       #calculate each step
            C_a = torch.from_numpy(C_a_np).to(x.device) 
            Z0_clust_score = get_clust_score_pytorch(Z_0, C_a).to(x.device).detach() 
        if self.b1in:
            #compute Z_1 in GCN
            output, Z_1 = self.gcn_1(gatoutput, support, self.B_1)
        else:
            #compute Z_1 out of GCN
            F_1 = self.gcn_1(gatoutput, support)
            Z_1 = torch.mm(self.B_1, F_1)
            output = Z_1
        return output, Z_1, Z_0, Z0_clust_score

    def l2_loss(self):
        loss = None
        for p in self.gcn_1.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()
        return loss

def row_normalize(matrix, epsilon=1e-10):

    # Step 1: Calculate the sum of each row
    row_sums = matrix.sum(axis=1, keepdims=True)
    
    # Step 2: Avoid division by zero by adding epsilon to zero rows
    row_sums[row_sums == 0] += epsilon
    
    # Step 3: Normalize each row by dividing by the row sum
    normalized_matrix = matrix / row_sums
    
    return normalized_matrix
def get_grad_IB_B_large(X, Z, y_np, F_, C_b_prime_np, Q):
    num_classes = len(np.unique(y_np))
    X_np = X.cpu().numpy()
    Z_np = Z.detach().cpu().numpy() #calculate each step

    y_one_hot_np = one_hot_encode(y_np, num_classes)   #no change
    y_one_hot = torch.from_numpy(y_one_hot_np).to(torch.float32).to(X.device)


    C_a_np = run_kmeans(Z_np, y_np, num_classes)       #calculate each step
    #C_b_prime_np = run_kmeans(X_np, y_np, num_classes) #no change     differnet from the function in utils.py



    C_a = torch.from_numpy(C_a_np).to(X.device)                    #calculate each step
    C_b_prime = torch.from_numpy(C_b_prime_np).to(X.device)        #no change

    #compute gradient
    #phi_Z_a shape: N*K   3*2
    phi_Z_a =  get_clust_score_pytorch(Z, C_a).to(X.device).detach()   #calculate each step   #G
    phi_X_b =  get_clust_score_pytorch(X, C_b_prime).to(X.device).detach()    #no change


    log_Q = torch.log(Q)
    log_phi_X_b = torch.log(phi_X_b)


    #gradient_I_Z
    n = X.shape[0]

    U = get_U(Z, C_a, phi_Z_a)
    phi_Z_a_exp = torch.unsqueeze(phi_Z_a, 1).permute(2, 0, 1)
    mul_phi_Z_a_log_phi_X_b = phi_Z_a_exp * log_phi_X_b
    sum_phi_Z_a_log_phi_X_b = mul_phi_Z_a_log_phi_X_b.sum(2).permute(1, 0)


    #y_one_hot_np = one_hot_encode(y.numpy(), 3) 
    #y_one_hot = torch.from_numpy(y_one_hot_np).to(torch.float32)
    sum_y_logQ = torch.matmul(y_one_hot, log_Q)

    diff_b = sum_phi_Z_a_log_phi_X_b - sum_y_logQ

    U_mul_FT = torch.matmul(U, F_.T) 
    gradient_IBB_B =  (U_mul_FT * (diff_b.unsqueeze(2))).sum(1) / n
    return gradient_IBB_B, phi_Z_a

def get_clust_score_pytorch(feat, centroids, beta=1.0):
    # pytroch version of get_clust_score
    # compute cluster score for each feature (phi in paper)
    # feat: N x D
    # centroids: K x D
    # return: N x K
    N = feat.shape[0]
    K = centroids.shape[0]
    score = torch.zeros((N, K))
    for i in range(K):
        score[:, i] = torch.linalg.norm(feat - centroids[i], dim=1)   #**2
    score = -beta * score
    score = torch.exp(score)
    score = score + 1e-10
    score = score/torch.sum(score, dim=1, keepdim=True)
    return score

def insert_label_lacked_nodes_batch(data, batch, y_ps, features, adj, y_ps_batch, num_classes):
    #features, adj is only for the batch
    unique_y_ps = np.unique(y_ps_batch)
    all_classes = np.arange(num_classes)
    labels_lacked = all_classes[~np.isin(all_classes, unique_y_ps)]
    num_lacked = len(labels_lacked)
    add_features = torch.zeros((num_lacked, data.x.shape[1]))
    add_adj = np.zeros((num_lacked, batch.x.shape[0]))

    
    for i in range(num_lacked):
        #sample a nodes from all data with lacked classes
        indices = np.where(y_ps == labels_lacked[i])[0]
        random_index = np.random.choice(indices)
        #get the neighbor map of the sampled nodes
        neighbors_0 = data.edge_index[0][data.edge_index[1] == random_index].numpy()
        neighbors_1 = data.edge_index[1][data.edge_index[0] == random_index].numpy()
        #np concatentation
        neighbors = np.concatenate((neighbors_0, neighbors_1), 0)

        nei_idx_in_batch = np.where(np.isin(neighbors, batch.n_id.numpy()))[0]
        add_adj[i, nei_idx_in_batch] = 1
        #get the feature of the sampled nodes
        add_features[i] = data.x[random_index]
    #insert the sampled nodes into the batch
    features = torch.cat((features, add_features), 0)

    adj = np.concatenate((adj, add_adj.T), 1)

    #add num_lacked columns with zeros to adj
    adj = np.concatenate((adj, np.zeros((num_lacked, adj.shape[1] ))), 0) 

    #add labels tp y_ps_batch
    y_ps_batch = np.concatenate((y_ps_batch, labels_lacked), 0)   

    return features, adj, y_ps_batch, num_lacked



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--model', default='gcn')
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropoutgcn', type=float, default=0.5, help='Dropout for gcn layer.')
    parser.add_argument('--dropoutgat', type=float, default=0.5, help='Dropout for gat layer.')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--update_frequency', type=int, default=1)
    parser.add_argument('--tildes', type=int, default=0)
    parser.add_argument('--f0mlp', type=int, default=1)
    parser.add_argument('--b1in', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--log_each_epoch', action='store_true', help="If present, store loss for each epoch.")
    parser.add_argument('--save_ib', action='store_true', help="If present, store ib for each epoch.")
    parser.add_argument('--ps_labels_path', type=str, default="/home/local/ASUAD/changyu2/GIB/data/all_ps_labels_ogbn.npy")
    parser.add_argument('--log_path', type=str, default="ogbn_log.txt")
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--test_per_epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="/data-drive/backup/changyu/expe/gib/ogbn/")
    args = parser.parse_args((["--warmup", "0"])  )  
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    setting = f"{args.dataset}_w{args.warmup}_up{args.update_frequency}_t{args.tildes}_gcnout{args.dropoutgcn}_gatout{args.dropoutgat}_lr{args.learning_rate}_wd{args.weight_decay}_hid{args.hidden}_numnei_{args.num_neighbors}_trainb{args.train_batch_size}_testb{args.test_batch_size}_seed{args.seed}"
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = f"{setting}.txt"
    
    # load data
    target_dataset = 'ogbn-arxiv'
    
    # This will download the ogbn-arxiv to the 'networks' folder
    dataset = PygNodePropPredDataset(name=target_dataset, root='networks')
    data = dataset[0]
    
    split_idx = dataset.get_idx_split() 
            
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    #normalize features
    data.x = row_normalize(data.x)
    num_classes = dataset.y.max().item() + 1


     
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[args.num_neighbors], 
                                 shuffle=True, num_workers=4, 
                                 batch_size=args.train_batch_size,)
    
    total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                 batch_size=args.test_batch_size, shuffle=False, 
                                 num_workers=4)
    test_loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=[-1],
                                 batch_size=args.test_batch_size, shuffle=False, 
                                 num_workers=4)
    '''
    train_loader = NeighborLoader(data, input_nodes=valid_idx, num_neighbors=[1]*2, 
                                 shuffle=True, num_workers=4, 
                                 batch_size=args.train_batch_size,)
    
    total_loader = NeighborLoader(data, input_nodes=valid_idx, num_neighbors=[1],
                                 batch_size=args.test_batch_size, shuffle=False, 
                                 num_workers=4)


    y_ps = np.load(args.ps_labels_path)
    '''


    #net = GIB(feat_dim, num_classes, num_features_nonzero, args)
    net = GIB_large(data.x.shape[1], num_classes, 0, args)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    net.train()
    acc_test_history=[]

    best_acc = 0
    best_epoch = 0
    train_loss_hist = []
    acc_hist = []


    #here we use psudo labels generated by gcn
    y_ps = np.load(args.ps_labels_path)
    #y_ps_onehot = one_hot_encode(y_ps, num_classes)

    X_np = data.x.cpu().numpy()
    C_b_prime_np = run_kmeans(X_np, y_ps, num_classes) #no change
    Q = (torch.ones(num_classes, num_classes)/num_classes).to(device)

    
    all_classes = np.arange(num_classes)

    for epoch in range(args.epochs):
        t1 = time.time()
        net.train()
        if epoch>= args.warmup and epoch%args.update_frequency ==0:
            update=True
        else:
            update=False
        #print(f"update:{update}")

        

        total_loss = 0
        total_correct = 0
        approx_acc = 0

        Q_numerator = torch.zeros([num_classes, num_classes]).to(device)
        Q_denominator = torch.zeros([num_classes]).to(device)
    
        for batch in train_loader:
            print(f"batch size: {batch.x.shape}")

            adj = (to_dense_adj(batch.edge_index)[0]).numpy()
            #make the adj symmetric

            features = batch.x

            #get ps_labels:
            y_ps_batch = y_ps[(batch.n_id).numpy()]
            
            if len(np.unique(y_ps_batch)) < num_classes:    #40
                #get the get the difference between all classes and unique classes of his batch
                features, adj, y_ps_batch, num_lacked = insert_label_lacked_nodes_batch(data, batch, y_ps, features, adj, y_ps_batch, num_classes)
            
            adj = adj + adj.T
            adj = np.asarray(adj + sp.eye(adj.shape[0]))
            adj = row_normalize(adj)
            adj = torch.FloatTensor(adj).to(device)
            features = features.to(device)

            y_ps_batch_onehot = one_hot_encode(y_ps_batch, num_classes)
            y_ps_batch_onehot = torch.from_numpy(y_ps_batch_onehot).to(torch.float32).to(device)

            out, Z_0, Z_1, Z0_clust_score = net(features, adj, y_ps_batch, C_b_prime_np, Q, update)
        
        
            out = out[:args.train_batch_size]
            batch_y = batch.y[:args.train_batch_size].to(device)
            batch_y = torch.reshape(batch_y, (-1,))
            batch_y_onehot = torch.from_numpy(one_hot_encode(batch_y.cpu().numpy(), num_classes)).to(torch.float32).to(device)

            if len(out) != len(batch_y):
                continue 
            loss = F.cross_entropy(out, batch_y, reduction='none')
            loss = loss.mean()
            loss_all = loss + args.weight_decay * net.l2_loss()
            loss.backward()
            optimizer.step()
        
            total_loss += float(loss)
            batch_correct = int(out.argmax(dim=-1).eq(batch_y).sum())
            total_correct += batch_correct

            batch_approx_acc = batch_correct / batch_y.size(0)
            print(f"batch_approx_acc:{batch_approx_acc}")
            approx_acc += batch_approx_acc


    
            #compute Q: only use train data to compute Q    #if the result is not good, we can use all data to compute Q
            Q_numerator = Q_numerator + torch.matmul(Z0_clust_score[:args.train_batch_size].T, batch_y_onehot)
            Q_denominator = Q_denominator + torch.sum(batch_y_onehot, dim=0)


        loss = total_loss / len(train_loader)
        approx_acc = approx_acc / len(train_loader)
        t2 = time.time()
        print(f"Epoch:{epoch+1}, train loss: {loss}, approx_acc: {approx_acc}, time: {t2-t1} \n")
        Q = Q_numerator / Q_denominator
        net.Q = Q
        if args.log_each_epoch:
            with open(log_path, 'a') as file:
                file.write(f"Epoch:{epoch+1}, train loss: {loss}, approx_acc: {approx_acc}, time: {t2-t1} \n")

        #TODO:add compute acc
        if epoch+1 > args.warmup and epoch % args.test_per_epochs == 0:
            t3 = time.time()
            net.eval()
            pred_test = torch.tensor([]).to(device)
            gt_y_all = torch.tensor([]).to(device)
            total_correct = 0
            for batch in test_loader:
                adj = (to_dense_adj(batch.edge_index)[0]).numpy()
                features = batch.x

                #get ps_labels:
                y_ps_batch = y_ps[(batch.n_id).numpy()]
                
                if len(np.unique(y_ps_batch)) < num_classes:    #40
                    #get the get the difference between all classes and unique classes of his batch
                    features, adj, y_ps_batch, num_lacked = insert_label_lacked_nodes_batch(data, batch, y_ps, features, adj, y_ps_batch, num_classes)
                
                adj = adj + adj.T
                adj = np.asarray(adj + sp.eye(adj.shape[0]))
                adj = row_normalize(adj)
                adj = torch.FloatTensor(adj).to(device)
                features = features.to(device)

                #y_ps_batch_onehot = one_hot_encode(y_ps_batch, num_classes)
                #y_ps_batch_onehot = torch.from_numpy(y_ps_batch_onehot).to(torch.float32).to(device)

                out, Z_0, Z_1, Z0_clust_score = net(features, adj, y_ps_batch, C_b_prime_np, net.Q, update)
                out = out[:args.test_batch_size]
                gt_y = batch.y[:args.test_batch_size].reshape(-1).to(device)
                pred_test_batch = out.argmax(dim=-1)
                pred_test = torch.cat((pred_test, pred_test_batch))
                gt_y_all = torch.cat((gt_y_all, gt_y))
            
            #train_acc = int(pred_test.cpu()[train_idx].eq(data.y[train_idx].reshape(-1)).sum())/ len(train_idx)
            test_acc = int(pred_test[:len(test_idx)].eq(gt_y_all[:len(test_idx)]).sum())/ len(test_idx)
            #val_acc = int(pred_test.cpu()[valid_idx].eq(data.y[valid_idx].reshape(-1)).sum())/ len(valid_idx)
            train_acc = val_acc = 0.5
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net, f"{args.save_path}{setting}.pt")
                best_epoch = epoch

            acc_test_history.append(test_acc)
            t4 = time.time()

            print(f"Epoch:{epoch+1}, train loss: {loss}, approx_acc: {approx_acc}, train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc}, best test acc:{best_acc}, best epoch: {best_epoch}, time: {t4-t3} \n")
            with open(log_path, 'a') as file:
                file.write(f"Epoch:{epoch+1}, train loss: {loss}, approx_acc: {approx_acc}, train acc: {train_acc}, val acc: {val_acc}, test acc: {test_acc},best test acc:{best_acc}, best epoch: {best_epoch}, time: {t4-t3} \n")

        train_loss_hist.append(loss)
        

    acc_test_all = np.array(acc_test_history)
    with open(log_path, 'a') as file:
        file.write(f'args: {setting}, seed: {args.seed}, best test acc:{best_acc}, best epoch: {best_epoch}\n')
        if args.save_ib:
            file.write(f"min ib_0:{min(ib0_hist)}, min ib_1:{min(ib1_hist)} \n")


if __name__ == '__main__':
    main()