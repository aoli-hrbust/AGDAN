import torch
import numpy as np
import torch.nn as nn
import cvxopt
import sys
import os

from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
import scipy.io as scio
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_c(Z, A , gamma=1):
    sys.stdout = open(os.devnull, 'w')
    c_min = torch.rand(A.shape[1], Z.shape[1]).to(device)
    H_0=2*gamma*torch.eye(A.shape[1]).to(device)+2*torch.mm(A.T,A)
    H=(H_0+H_0.T)/2
    H=H.cpu().detach().data.numpy()
    r=A.shape[1]
    for i in range(Z.shape[1]):
        f=-2*torch.matmul(Z[:,i].T,A)
        f=f.T
        f=f.cpu().detach().data.numpy()
        if i==0:
            c_min=quadprog(H,f, L=-np.eye(r), k=np.zeros((r,1),dtype=float),
                           Aeq=np.ones((1,r),dtype=float),beq=1)
            c_min=torch.tensor(c_min)
        else:
            tem= quadprog(H,f, L=-np.eye(r), k=np.zeros((r,1),dtype=float),
                           Aeq=np.ones((1,r),dtype=float),beq=1)
            c_min=torch.cat((c_min,torch.tensor(tem)),dim=1)
    sys.stdout = sys.__stdout__

    c_min=torch.clamp(c_min,0,1)
    return (c_min).float()

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]
    H = np.float64(H)
    f= np.float64(f)
    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])

def make_qp(x, centroids):
    q = 1.0 / (1.0*0.001 + torch.sum(torch.pow(x.unsqueeze(1).to(device) - torch.tensor(centroids).to(device), 2), 2))
    q = (q.t() / torch.sum(q, 1)).t()
    p = target_distribution(q)
    return q, p

def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def embeddingcontras(h_i ,h_j, data_size):
    loss_function=nn.CrossEntropyLoss(reduction="sum")
    N = 2 * data_size
    h = torch.cat((h_i, h_j), dim=0)
    sim = torch.matmul(h, h.T) / 0.5
    sim_i_j = torch.diag(sim, data_size)
    sim_j_i = torch.diag(sim, -data_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N // 2):
        mask[i, N // 2 + i] = 0
        mask[N // 2 + i, i] = 0
    mask = mask.bool()
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = loss_function(logits, labels)
    loss /= N
    return loss

def labelcontras(q_i, q_j, n_clusters):
    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = torch.log(torch.tensor(p_i.size(0))) + (p_i * torch.log(p_i)).sum()
    p_j = q_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = torch.log(torch.tensor(p_j.size(0))) + (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    q_i = q_i.t()
    q_j = q_j.t()
    N = 2 * n_clusters
    q = torch.cat((q_i, q_j), dim=0)

    sim = nn.CosineSimilarity(dim=2)(q.unsqueeze(1), q.unsqueeze(0)) /1
    sim_i_j = torch.diag(sim, n_clusters)
    sim_j_i = torch.diag(sim, -n_clusters)

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, labels)
    loss /= N


    return loss + entropy

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def TPL(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links is not 0:
        links = torch.Tensor(links).to('mps')
        weights += torch.eye(size).to('mps')
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.to('mps')
    weights = weights.to('mps')
    return weights, raw_weights


def get_Laplacian_from_weights(weights):
    # W = torch.eye(weights.shape[0]).cuda() + weights
    # degree = torch.sum(W, dim=1).pow(-0.5)
    # return (W * degree).t()*degree
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t()*degree


def noise(weights, ratio=0.1):
    sampling = torch.rand(weights.shape).to('mps') + torch.eye(weights.shape[0]).to('mps')
    sampling = (sampling > ratio).type(torch.IntTensor).to('mps')
    return weights * sampling


if __name__ == '__main__':
    tX = torch.rand(3, 8)
    print(cal_weights_via_CAN(tX, 3))
