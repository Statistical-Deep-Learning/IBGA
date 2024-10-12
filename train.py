import  torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import  numpy as np
from data import load_data, preprocess_features, preprocess_adj, load_data_all, load_data_cs
from utils import masked_loss, masked_acc, sparse_dropout, assign_labels, assign_labels2, run_kmeans, one_hot_encode, get_mutual_information
from utils import compute_ib, get_grad_IB_B
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
    parser.add_argument('--log_path', type=str, default="")
    parser.add_argument('--data_path', type=str, default="")
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    setting = f"{args.dataset}_w{args.warmup}_up{args.update_frequency}_t{args.tildes}_gcnout{args.dropoutgcn}_gatout{args.dropoutgat}_lr{args.learning_rate}_wd{args.weight_decay}_hid{args.hidden}"
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = f"{setting}.txt"
    # load data
    if args.dataset == 'cs':
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all = load_data_cs(args.data_path)
    else:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_all = load_data_all(args.dataset)

    features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
    supports = preprocess_adj(adj)

    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    i_f = torch.from_numpy(features[0]).long().to(device)
    v_f = torch.from_numpy(features[1]).to(device)
    feature = torch.sparse_coo_tensor(i_f.t(), v_f, features[2]).to_dense().to(device)
    feature = feature.float()
    #feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

    i_s = torch.from_numpy(supports[0]).long().to(device)
    v_s = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse_coo_tensor(i_s.t(), v_s, supports[2]).float().to_dense().to(device)

    num_features_nonzero = torch.sparse.FloatTensor(i_f.t(), v_f, features[2])._nnz()
    feat_dim = feature.shape[1]

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

    if args.tildes == 1:
        feature = torch.mm(support, feature)
    elif args.tildes == 2:
        feature = torch.mm(support, torch.mm(support, feature))

    

    #here we use psudo labels generated by gcn
    y_ps = np.load(args.ps_labels_path)
    y_ps_onehot = one_hot_encode(y_ps, num_classes)

    X_np = feature.cpu().numpy()
    C_b_prime_np = run_kmeans(X_np, y_ps, num_classes) #no change

    for epoch in range(args.epochs):
        #t1 = time.time()
        torch.autograd.set_detect_anomaly(True)
        net.train()
        if epoch>= args.warmup and epoch%args.update_frequency ==0:
            update=True
        else:
            update=False
        #print(f"update:{update}")


        out, Z_1, Z_0 = net(feature, support, y_ps, C_b_prime_np, update=update)

        loss = masked_loss(out, train_label, train_mask)
        #这个loss为什么要+后面l2loss？
        loss_all = loss + args.weight_decay * net.l2_loss()
        # loss_all = loss
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        net.eval()
        out, Z_1, Z_0  = net(feature, support, y_ps, C_b_prime_np, update=update)
        val_loss = masked_loss(out, val_label, val_mask)
        test_loss = masked_loss(out, test_label, test_mask)
        acc_train = masked_acc(out, train_label, train_mask)
        acc_val = masked_acc(out, val_label, val_mask)
        acc_test = masked_acc(out, test_label, test_mask)
        if args.log_each_epoch:
            with open(log_path, 'a') as file:
                file.write(f"Epoch:{epoch+1}, train loss: {loss.item()}, val loss: {val_loss.item()}, test loss:{test_loss.item()}, train acc: {acc_train.item()}, val acc: {acc_val.item()}, test acc: {acc_test.item()} \n")
        if epoch%10 == 0:
            print('Epoch:', epoch+1, 'train loss:', '%.6f' % loss.item(), 'test loss:', '%.6f' % test_loss.item(), 'train acc:', '%.4f' % acc_train.item(), 'val acc:', '%.4f' % acc_val.item(), 'test acc:', '%.4f' % acc_test.item())
        if args.save_ib:
            Z_1 = Z_1.cpu().data.numpy()
            Z_0 = Z_0.cpu().data.numpy()
            input_feat = feature.to_dense().cpu().data.numpy()
            y_all_1d = y_all.argmax(axis=1)
            ib_0, MI_X_Z0, MI_Z0_Y,  = compute_ib(input_feat, Z_0, y_ps, num_classes, y_ps_onehot)
            ib_1, MI_X_Z1, MI_Z1_Y,  = compute_ib(input_feat, Z_1, y_ps, num_classes, y_ps_onehot)
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
    net.eval()
    out, Z_1, Z_0 = net(feature, support, y_ps, C_b_prime_np, update=True)
    # out = out[0]
    acc = masked_acc(out, test_label, test_mask)
    acc_test_all = np.array(acc_test_history)
    with open(log_path, 'a') as file:
        file.write(f'args: {setting}, seed: {args.seed}, best test acc:{max(acc_test_history[args.warmup:])} \n')
        if args.save_ib:
            file.write(f"min ib_0:{min(ib0_hist)}, min ib_1:{min(ib1_hist)} \n")


if __name__ == '__main__':
    main()