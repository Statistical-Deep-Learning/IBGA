import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import  numpy as np
from data import load_data, preprocess_features, preprocess_adj, load_data_all, load_data_cs
from utils import masked_loss, masked_acc, sparse_dropout, assign_labels, assign_labels2, run_kmeans, one_hot_encode, get_mutual_information
from utils import compute_ib, get_grad_IB_B, get_grad_IB_B_large, get_clust_score_pytorch, get_U, get_clust_score
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
#from GATConv import GATConv
from torch_geometric.nn import GCNConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(128, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.norms.append(torch.nn.LayerNorm(heads * hidden_channels, elementwise_affine=True))


        self.skips = torch.nn.ModuleList()
        #self.skips.append(Lin(dataset.num_features, hidden_channels * heads))
        #for _ in range(num_layers - 2):
        #    self.skips.append(
        #        Lin(hidden_channels * heads, hidden_channels * heads))
        #self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        #for skip in self.skips:
        #    skip.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index) #+ skip(x)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

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
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--test_per_epochs', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="/data-drive/backup/changyu/expe/gib/ogbn/")
    args = parser.parse_args(["--warmup", "1", "--train_batch_size", "128"]) 
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

    #normalize features: will degrade performance to 0.3
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

    

    model = GAT(128, 256, 40, num_layers=2, heads=1).to(device)
    #model = GIB_GATConv(in_channels=128, hidden_channels=256, out_channels=40)
    #model = GCN(hidden_channels=64)
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
    C_b_prime_np = run_kmeans(X_np, y_ps, num_classes) #no change
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

        Q_numerator = torch.zeros([num_classes, num_classes]).to(device)
        Q_denominator = torch.zeros([num_classes]).to(device)
        for batch in train_loader:
            #batch = batch.to(device)
            y_ps_batch = y_ps[(batch.n_id).numpy()]


            #print("shape of x:", batch.x.shape)
            if batch.x.shape[0] < train_batch_size:
                #print("skip")
                continue
            
            features = batch.x
            edge_index = batch.edge_index
            
            features = features.to(device)
            edge_index = edge_index.to(device)
            batch.y = batch.y.to(device)

            optimizer.zero_grad()
            out= model(features, edge_index)
            #loss = F.cross_entropy(out[:train_batch_size], batch.y[:train_batch_size].squeeze(1), reduction='none')
            #loss = criterion(out[:train_batch_size], batch.y[:train_batch_size].squeeze(1))
            #loss = loss.mean()
            loss = F.cross_entropy(out[:train_batch_size], batch.y[:train_batch_size].squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            out = out[:train_batch_size]

            batch_y = batch.y[:train_batch_size].to(device)
            batch_y = torch.reshape(batch_y, (-1,))
            batch_y_onehot = torch.from_numpy(one_hot_encode(batch_y.cpu().numpy(), num_classes)).to(torch.float32).to(device)

            batch_correct = int(out.argmax(dim=-1).eq(batch_y).sum())
            total_correct += batch_correct
            total_len += batch_y.size(0)


            batch_approx_acc = batch_correct / batch_y.size(0)
            #print(f"batch_approx_acc:{batch_approx_acc}")

            #Q_numerator = Q_numerator + torch.matmul(Z0_clust_score[:args.train_batch_size].T, batch_y_onehot)
            #Q_denominator = Q_denominator + torch.sum(batch_y_onehot, dim=0)
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}, approx_acc: {total_correct/total_len}')
    

if __name__ == '__main__':
    main()