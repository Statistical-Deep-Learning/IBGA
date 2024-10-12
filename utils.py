import  torch
from    torch import nn
from    torch.nn import functional as F

import numpy as np
from scipy.cluster.vq import kmeans, vq
from scipy.stats import mode
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def all_loss(out, label, mask):
    
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    return loss

def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss
    
def masked_loss_reg(out, label, mask):

    loss_f = nn.MSELoss()
    loss = loss_f(out, label)
    return loss

def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc



def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def assign_labels(label, pred_cluster):
    """
    Assign labels to nodes in data.
    data: two columns, first column is ground labels, second column is cluster index.
    We need to assign labels to each cluster index.
    The strategy is to assign the most frequent ground label to each cluster index.
    After assigning labels to a cluster index, we remove data with cluster index from data.
    And also remove the data with the assigned label from data.
    Then assign labels to the next cluster index.
    Then repeat the above steps until all cluster indexes are assigned labels.
    """

    unique_label = np.unique(label)
    unique_cluster = np.unique(pred_cluster)
    data = np.column_stack((label, pred_cluster))
    data_copy = data.copy()
    assigned_cluster_label = np.zeros((0, 2)).astype(int)

    while data_copy.shape[0]>0:

        unique_rows, indices, counts = np.unique(data_copy, axis=0, return_inverse=True, return_counts=True)

        max_count_idx = counts.argmax()
        max_mode = unique_rows[max_count_idx].reshape(1,2)
        assigned_cluster_label = np.vstack((assigned_cluster_label, max_mode))

        not_in_filter1 = ~np.isin(data_copy[:, 0], assigned_cluster_label[:,0])
        not_in_filter2 = ~np.isin(data_copy[:, 1], assigned_cluster_label[:,1])
        combined_filter = not_in_filter1 & not_in_filter2
        data_copy = data_copy[combined_filter]

    if assigned_cluster_label.shape[0] < unique_label.shape[0]:
        label_left = unique_label[~np.isin(unique_label, assigned_cluster_label[:, 0])]
        cluster_left = unique_cluster[~np.isin(unique_cluster, assigned_cluster_label[:, 1])]
        left_cluster_label = np.column_stack((label_left, cluster_left))
        assigned_cluster_label = np.vstack((assigned_cluster_label, left_cluster_label))

    return assigned_cluster_label




def assign_labels2(true_labels, cluster_labels):
    # Assume `cluster_labels` is the array of labels from the clustering algorithm
    # and `true_labels` is the array of ground truth labels


    # Build a confusion matrixd
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

    for i in range(K):
        score[:, i] = np.linalg.norm(feat_normalized - centroids[i], axis=1) #** 2
    score = -beta * score
    score = np.exp(score)
    score /= np.sum(score, axis=1, keepdims=True)  # softmax
    return score

def get_clust_score_pytorch(feat, centroids, beta=1.0):
    #pytroch version of get_clust_score
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
    score = score/torch.sum(score, dim=1, keepdim=True)
    return score

def compute_ib(input_feat, inter_feat, y_all_1d, num_classes, y_all):
    #input_feat: input feature
    #inter_feat: output feature from one layer
    #y_all_1d: labels
    inter_centroids = run_kmeans(inter_feat, y_all_1d, num_classes)
    #print(inter_centroids.sum(1))
    input_centroids = run_kmeans(input_feat, y_all_1d, num_classes)
    #print(input_centroids.sum(1))
    inter_score = get_clust_score(inter_feat, inter_centroids)

    input_score = get_clust_score(input_feat, input_centroids)

    clust_score_input =  get_clust_score(input_feat, input_centroids)
    p_in = np.sum(clust_score_input, axis=0)
    #print(f"p_in:{p_in}")

    clust_score_output = get_clust_score(inter_feat, inter_centroids)
    p_out = np.sum(clust_score_output, axis=0)
    #print(f"p_out:{p_out}")

    one_hot_target = y_all
    p_label = np.sum(one_hot_target, axis=0)
    #print(f"p_label:{p_label}")


    p_in_out = np.sum(np.matmul(clust_score_input[:, :, np.newaxis],
                            clust_score_output[:, np.newaxis, :]), axis=0)

    #print(f"p_in_out:{p_in_out}")

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

def get_U(Z, C_a, phi_Z_a):
    #How to use matrix operation to simplify calculation of U
    Z_expanded = Z.unsqueeze(1)  # Shape (n, 1, d)
    C_a_expanded = C_a.unsqueeze(0)  # Shape (1, k, d)

    # Compute differences between each Z and each C_a
    differences = Z_expanded - C_a_expanded  # Shape (n, k, d)
    #differences_expanded = differences.unsqueeze(2)


    # Calculate sum term across samples (reduction across the dimension of samples)
    # First, expand phi_Z_a from (n, k) to (n, k, 1) for multiplication with differences
    phi_Z_a_expanded = phi_Z_a.unsqueeze(-1)  # Shape (n, k, 1)

    # Compute the weighted sum across the samples for each class and dimension
    #term = phi_Z_a_expanded * differences
    sum_term = torch.sum(phi_Z_a_expanded * differences, dim=1)
    sum_term_expanded = sum_term.unsqueeze(1)
    #second_term = differences - sum_term_expanded

    # Final U calculation
    U = -2 * phi_Z_a_expanded * (differences - sum_term_expanded)
    return U # Should be (n, k, d)


def get_gradient_I_Z_B(Z, F, U, sum_G_col, sum_G_H, sum_H_col, phi_X_b, n, num_classes,):
    #Z = BF       #calculate each step
    #F, U         #calculate each step
    #sum_G_col    #calculate each step
    #sum_G_H,     #calculate each step   #replace X with Y
    #sum_H_col    #no change    #replace X with Y
    #phi_X_b      #no change    #replace X with Y
    #n, num_classes       #no change hnbb

    log_n = np.log(n)
    #I(X,Z)gradient respected Z
    log_sum_G_H = torch.log(sum_G_H)
    log_sum_Ga = torch.log(sum_G_col)
    log_sum_Hb = torch.log(sum_H_col)


    # Expand dims for broadcasting
    phi_X_b_exp = phi_X_b.unsqueeze(1) # (n, 1, k)
    U_exp = U.unsqueeze(2)  # (n, k, 1, d)
    sum_G_H_exp = sum_G_H.unsqueeze(0)

    # Compute the log terms for all combinations
    log_terms = log_n + log_sum_G_H - log_sum_Ga.unsqueeze(1) - log_sum_Hb.unsqueeze(0)  # (k, k)
    log_terms_exp = log_terms.unsqueeze(0)# (1, k, k)

    # Compute ratios and constants
    ratio_terms = phi_X_b_exp / sum_G_H_exp  # (n, k, k)
    constant_terms = 1 / sum_G_col.unsqueeze(0).unsqueeze(-1)  # (1, k, 1)


    first_term = phi_X_b_exp * log_terms_exp      # (n, k, k)
    second_term = sum_G_H_exp * (ratio_terms - constant_terms) #(n, k, k)
    sum_terms = (first_term + second_term).unsqueeze(3)  #n,k,k,1
    # Calculate the gradient contribution for each i, a, b
    gradient_contrib = U_exp * sum_terms

    # Sum over classes a and b
    gradient_I_Z = 1/n * gradient_contrib.sum(dim=[1, 2])  # Sum across a and b
    #gradient_I_Z = 1/n * gradient_I_Z

    #I(X,Z)gradient respected B
    gradient_I_Z_B = torch.mm(gradient_I_Z, F.t())
    return gradient_I_Z_B

"""
def get_grad_IB_B(X, Z, y_np, F_):
    num_classes = len(np.unique(y_np))
    X_np = X.cpu().numpy()
    Z_np = Z.detach().cpu().numpy() #calculate each step

    y_one_hot_np = one_hot_encode(y_np, num_classes)   #no change
    y_one_hot = torch.from_numpy(y_one_hot_np).to(torch.float32).to(X.device)

    C_a_np = run_kmeans(Z_np, y_np, num_classes)       #calculate each step
    C_b_prime_np = run_kmeans(X_np, y_np, num_classes) #no change

    C_a = torch.from_numpy(C_a_np).to(X.device)                    #calculate each step
    C_b_prime = torch.from_numpy(C_b_prime_np).to(X.device)        #no change

    #compute gradient
    #phi_Z_a shape: N*K   3*2
    phi_Z_a =  get_clust_score_pytorch(Z, C_a).to(X.device).detach()   #calculate each step   #G
    phi_X_b =  get_clust_score_pytorch(X, C_b_prime).to(X.device).detach()    #no change

    #gradient_I_Z
    n = X.shape[0]
    log_n = np.log(n)                           #no change
    sum_H_col = torch.sum(phi_X_b,dim=0)        #no change
    sum_Y_col = torch.sum(y_one_hot,dim=0)      #no change
    sum_G_col = torch.sum(phi_Z_a,dim=0)        #calculate each step
    sum_G_H = torch.mm(phi_Z_a.t(), phi_X_b)    #calculate each step
    sum_G_Y = torch.mm(phi_Z_a.t(), y_one_hot)  #calculate each step

    #U = torch.zeros((Z.shape[0], num_classes, Z.shape[1]))  #U calculate each step
    #for i in range(Z.shape[0]):
    #    for a in range(num_classes):
    #        U_ia = -2 * phi_Z_a[i,a]*(Z[i]-C_a[a] - torch.sum(phi_Z_a[i:i+1].t()*(Z[i]-C_a), dim=0))
    #        U[i,a] = U_ia
    U = get_U(Z, C_a, phi_Z_a)


    gradient_I_Z_X_B = get_gradient_I_Z_B(Z, F_, U, sum_G_col, sum_G_H, sum_H_col, phi_X_b, n, num_classes)

    #for replace H with Y: replacing phi_X_b with y_one_hot
    gradient_I_Z_Y_B = get_gradient_I_Z_B(Z, F_, U, sum_G_col, sum_G_Y, sum_Y_col, y_one_hot, n, num_classes)

    return gradient_I_Z_X_B - gradient_I_Z_Y_B
"""
def get_grad_IB_B(X, Z, y_np, F_, C_b_prime_np):
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

    #gradient_I_Z
    n = X.shape[0]
    log_n = np.log(n)                           #no change
    sum_H_col = torch.sum(phi_X_b,dim=0)        #no change
    sum_Y_col = torch.sum(y_one_hot,dim=0)      #no change
    sum_G_col = torch.sum(phi_Z_a,dim=0)        #calculate each step
    sum_G_H = torch.mm(phi_Z_a.t(), phi_X_b)    #calculate each step
    sum_G_Y = torch.mm(phi_Z_a.t(), y_one_hot)  #calculate each step

    #U = torch.zeros((Z.shape[0], num_classes, Z.shape[1]))  #U calculate each step
    #for i in range(Z.shape[0]):
    #    for a in range(num_classes):
    #        U_ia = -2 * phi_Z_a[i,a]*(Z[i]-C_a[a] - torch.sum(phi_Z_a[i:i+1].t()*(Z[i]-C_a), dim=0))
    #        U[i,a] = U_ia
    U = get_U(Z, C_a, phi_Z_a)


    gradient_I_Z_X_B = get_gradient_I_Z_B(Z, F_, U, sum_G_col, sum_G_H, sum_H_col, phi_X_b, n, num_classes)

    #for replace H with Y: replacing phi_X_b with y_one_hot
    gradient_I_Z_Y_B = get_gradient_I_Z_B(Z, F_, U, sum_G_col, sum_G_Y, sum_Y_col, y_one_hot, n, num_classes)

    return gradient_I_Z_X_B - gradient_I_Z_Y_B

def computer_kmeans_accuracy(x, y, num_cluster, niter=100):
    #print('performing kmeans clustering')

    # Run kmeans to find centroids
    # centroids, distortion = kmeans(x, num_cluster, iter=niter)

    # # Assign samples to the nearest centroids
    # cluster_assignments, _ = vq(x, centroids)
    kmeans = KMeans(n_clusters=num_cluster, random_state=0)
    cluster_assignments = kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_

    # Map each cluster to the most frequent class label and reorder centroids
    # 这里如果两个cluster的mode of lable 要是一致的话，后面的centroids会替换掉前面的
    predict_label = np.ones_like(y)*(-1)
    reordered_centroids = np.zeros_like(centroids)
    for cluster in range(num_cluster):
        indices = np.where(cluster_assignments == cluster)[0]  # Indices of points in this cluster
        if len(indices) > 0:
            cluster_label = mode(y[indices]).mode#[0]  # Most common label in the cluster
            #print(f"cluster_label:{cluster_label}")
            predict_label[indices] = cluster_label
    #print(f"predict_label:{predict_label}")
    accuracy = (predict_label==y).sum()/y.shape[0]
    print(f"accuracy:{accuracy}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_reduced = tsne.fit_transform(x)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_all_1d, cmap='tab10', edgecolor='k', alpha=0.6, s=50)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of the digits dataset')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.show()


def get_grad_IB_B_large(X, Z, y_np, F_, C_b_prime_np, Q):
    num_classes = len(np.unique(y_np))
    X_np = X.cpu().numpy()
    Z_np = Z.detach().cpu().numpy() #calculate each step

    y_one_hot_np = one_hot_encode(y_np, num_classes)   #no change
    y_one_hot = torch.from_numpy(y_one_hot_np).to(torch.float32).to(X.device)

    if np.isnan(Z_np).any():
        print(np.isnan(Z_np).sum())
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
    return gradient_IBB_B

def row_normalize(matrix, epsilon=1e-10):

    # Step 1: Calculate the sum of each row
    row_sums = matrix.sum(axis=1, keepdims=True)
    
    # Step 2: Avoid division by zero by adding epsilon to zero rows
    row_sums[row_sums == 0] += epsilon
    
    # Step 3: Normalize each row by dividing by the row sum
    normalized_matrix = matrix / row_sums
    
    return normalized_matrix