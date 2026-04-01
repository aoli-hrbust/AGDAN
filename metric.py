from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, f1_score
import warnings

from idecutils import best_map, purity_score
from torch_utils import convert_numpy, convert_tensor

warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, k_means
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import Tensor
from kmeans_pytorch import kmeans as kmeans_torch
from kmeans_pytorch import pairwise_distance as pairwise_distance_torch
from typing import List
import torch.nn.functional as F

def compute_inertia(X, centroid):
    dist = pairwise_distance_torch(X, centroid)
    min_dist = torch.min(dist, dim=1)[0]
    inertia = torch.sum(min_dist)
    return inertia
def KMeans_Torch(X: Tensor, *, n_clusters, n_init=20, max_iter=1000, verbose=False):
    """
    Return: centroid, ypred
    """
    inertia_best = None
    ypred_best = None
    centroid_best = None
    iter_best = None
    for i in range(n_init):
        ypred, centroid = kmeans_torch(
            X,
            num_clusters=n_clusters,
            device=X.device,
            tqdm_flag=False,
            iter_limit=max_iter,
            # seed=args.seed,
        )
        inertia = compute_inertia(X, centroid)
        if verbose:
            print(f"iter {i:04} inertia {inertia:.4f}")
        if inertia_best is None or inertia < inertia_best:
            iter_best = i
            inertia_best = inertia
            ypred_best = ypred
            centroid_best = centroid

    if verbose:
        print(f"best iter {iter_best:04} inertia {inertia_best:.4f}")
    return centroid_best, ypred_best  # follow sklearn.


def KMeans_Evaluate(
    X,
    Y,
    clusterNum,
    *,
    return_centroid=False,
    n_init=20,
    max_iter=1000,
):
    label = Y
    n_clusters = clusterNum
    if isinstance(X, np.ndarray):
        centroid, ypred, *_ = k_means(
            X, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter
        )
    else:
        assert isinstance(X, Tensor)
        centroid, ypred = KMeans_Torch(
            X,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
        )
        ypred_tensor = ypred
        ypred = convert_numpy(ypred)

    metrics = dict(
        ACC=cluster_acc(label, ypred),
        NMI=v_measure_score(label, ypred),
        PUR=purity_score(label, ypred),
        F1=cluster_f1_score(label, ypred),
        # ARI=adjusted_rand_score(label, ypred),
    )
    if return_centroid:
        return metrics, centroid, ypred_tensor
    return metrics

def mse_missing_part(X_hat: List[Tensor], X: List[Tensor], M: Tensor):
    if not isinstance(M, Tensor):
        X_hat = convert_tensor(X_hat)
        X = convert_tensor(X)
        M = convert_tensor(M, dtype=torch.bool)

    loss = 0
    for v in range(len(X_hat)):
        loss += F.mse_loss(X_hat[v][M[:, v]], X[v][M[:, v]])
    loss /= len(X_hat)
    return loss.item()

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def cluster_f1_score(ytrue, ypred):
    return f1_score(y_true=ytrue, y_pred=best_map(ytrue, ypred), average="macro")


def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f1 = cluster_f1_score(label, pred),
    return acc, nmi, pur, ari, f1


def inference(loader, model, device, view, data_size):
    model.eval()
    commonZ = []
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            # xrs, zs, hs, commonz, S = model(xs)
            commonz = model.dim_fusion(xs)
            # commonz = commonz.detach()
            commonZ.extend(commonz.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    commonZ = np.array(commonZ)
    return labels_vector, commonZ

def valid(model, device, dataset, view, data_size, class_num, eval_h=False):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, commonZ = inference(test_loader, model, device, view, data_size)
    if eval_h:
        # print("Clustering results :")
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        y_pred = kmeans.fit_predict(commonZ)
        nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
        # print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))
    return acc, nmi, pur, ari, labels_vector, commonZ


