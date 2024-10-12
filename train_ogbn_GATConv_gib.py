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
from torch_geometric.nn import GATConv
from GATConv import GATConv

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

def insert_label_lacked_nodes_batch_for_GATConv(data, batch, y_ps, features, edge_index, y_ps_batch, num_classes):
    #features, adj is only for the batch
    unique_y_ps = np.unique(y_ps_batch)
    all_classes = np.arange(num_classes)
    labels_lacked = all_classes[~np.isin(all_classes, unique_y_ps)]
    num_lacked = len(labels_lacked)
    add_features = torch.zeros((num_lacked, data.x.shape[1]))

    
    for i in range(num_lacked):
        #sample a nodes from all data with lacked classes
        indices = np.where(y_ps == labels_lacked[i])[0]
        random_index = np.random.choice(indices)
        add_features[i] = data.x[random_index]
        #insert edge index
        edge_index = torch.cat([edge_index, torch.tensor([[random_index], [random_index]])], dim=1)
        
    #insert the sampled nodes into the batch
    features = torch.cat((features, add_features), 0)

    #add labels tp y_ps_batch
    y_ps_batch = np.concatenate((y_ps_batch, labels_lacked), 0)   

    return features, edge_index, y_ps_batch, num_lacked

class GIB_GATConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIB_GATConv, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * 1, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = torch.dropout(x, p=0.6, train=self.training)
        x = self.conv2(x, edge_index)
        return x

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
    args = parser.parse_args()  
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




    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    split_idx = dataset.get_idx_split() 
            
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    #normalize features
    #data.x = row_normalize(data.x)
    num_classes = dataset.y.max().item() + 1

    train_batch_size = args.train_batch_size
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[10] * 2, 
                                 shuffle=True, num_workers=4, 
                                 batch_size=train_batch_size)
    
    total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                 batch_size=train_batch_size, shuffle=False, 
                                 num_workers=4)
    test_loader = NeighborLoader(data, input_nodes=test_idx, num_neighbors=[-1],
                                 batch_size=train_batch_size, shuffle=False, 
                                 num_workers=4)

    


    model = GIB_GATConv(in_channels=dataset.num_features, hidden_channels=100, out_channels=dataset.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = torch.nn.CrossEntropyLoss()


    acc_test_history=[]
    best_acc = 0
    best_epoch = 0
    train_loss_hist = []
    acc_hist = []

    y_ps = np.load(args.ps_labels_path)

    
    X_np = data.x.cpu().numpy()
    #C_b_prime_np = run_kmeans(X_np, y_ps, num_classes) #no change
    Q = (torch.ones(num_classes, num_classes)/num_classes).to(device)
    all_classes = np.arange(num_classes)
    
    for epoch in range(100):  # Number of epochs
        t1 = time.time()
        model.train()
        if epoch>= args.warmup and epoch%args.update_frequency ==0:
            update=True
        else:
            update=False

        total_loss = 0
        total_correct = 0
        total_len = 0
        approx_acc = 0
        for batch in train_loader:
            #batch = batch.to(device)
            y_ps_batch = y_ps[(batch.n_id).numpy()]


            #print("shape of x:", batch.x.shape)
            if batch.x.shape[0] < train_batch_size:
                #print("skip")
                continue
            
            features = batch.x
            edge_index = batch.edge_index
            if len(np.unique(y_ps_batch)) < num_classes:    #40
                #get the get the difference between all classes and unique classes of his batch
                features, edge_index, y_ps_batch, num_lacked = insert_label_lacked_nodes_batch_for_GATConv(data, batch, y_ps, features, batch.edge_index, y_ps_batch, num_classes)

            
            features = features.to(device)
            edge_index = edge_index.to(device)
            batch.y = batch.y.to(device)

            optimizer.zero_grad()
            out = model(features, edge_index)
            loss = F.cross_entropy(out[:train_batch_size], batch.y[:train_batch_size].squeeze(1), reduction='none')
            #loss = criterion(out[:train_batch_size], batch.y[:train_batch_size].squeeze(1))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            out = out[:train_batch_size]

            batch_y = batch.y[:train_batch_size].to(device)
            batch_y = torch.reshape(batch_y, (-1,))

            batch_correct = int(out.argmax(dim=-1).eq(batch_y).sum())
            total_correct += batch_correct
            total_len += batch_y.size(0)


            batch_approx_acc = batch_correct / batch_y.size(0)
            #print(f"batch_approx_acc:{batch_approx_acc}")
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, approx_acc: {total_correct/total_len}')
    

if __name__ == '__main__':
    #main()
    X = torch.randn(475, 128)
    F_ = torch.randn(1903, 128)
    Z = torch.randn(475, 128)
    C_b_prime_np = torch.randn(40, 128).numpy()
    Q = torch.randn(40, 40)
    y_np = np.random.randint(0, 40, size=475)
    a = get_grad_IB_B_large(X, Z, y_np, F_, C_b_prime_np, Q)