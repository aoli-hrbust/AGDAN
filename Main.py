import math

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from dataloader import load_data
from dataset import make_mask

from metric import evaluate
import random
import argparse
from Model import AGDAN
from ptsne_training import calculate_optimized_p_cond, make_joint, loss_function
from utils import calculate_c, make_qp
from torch_utils import convert_tensor
from ptsne_training import get_q_joint


parser = argparse.ArgumentParser(description='AGDAN')
parser.add_argument('--dataset', default='HW')
parser.add_argument("--eta", default=0.1)
parser.add_argument('--AnchorNum', type=int, default=40,
                        help='Initialize the number of anchors.')
parser.add_argument('--gamma', type=int, default=5)
parser.add_argument("--feature_dim", default=512)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.001)

args = parser.parse_args()

# device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cacluate_U(C):
    U, _, _ = torch.linalg.svd(C.T)
    U = U[:, 0:class_num]
    return U

def preprocess(eta: float):
    paired_rate: float = 1 - eta
    M = make_mask(
        paired_rate=paired_rate,
        sampleNum=data_size,
        viewNum=view,
        kind="partial",
    )
    perplexity: int = 10
    X = dataset.X
    X_view = [dataset.X[v][M[:, v]] for v in range(view)]
    scaler_view = [MinMaxScaler() for _ in range(view)]
    for v in range(view):
        X[v] = scaler_view[v].fit_transform(X[v])
        X_view[v] = scaler_view[v].fit_transform(X_view[v])
    X = convert_tensor(X, torch.float, device)
    X_view = convert_tensor(X_view, torch.float, device)

    X_gt = [None] * view
    scaler_view = [MinMaxScaler() for _ in range(view)]
    for v in range(view):
        X_gt[v] = scaler_view[v].fit_transform(dataset.X[v])
    X_gt = convert_tensor(X_gt, torch.float, device)

    S_view = [
        calculate_optimized_p_cond(x, math.log2(perplexity), dev=device)
        for x in X_view
    ]

    P_view = [make_joint(s) for s in S_view]

    res = dict(
        data=dataset,
        M=convert_tensor(M, torch.bool, device),
        S_view=S_view,
        P_view=P_view,
        X_view=X_view,
        X_gt=X_gt,
        X=X
    )

    return res

def pretrain(inputs):
    tot_loss = 0.
    print('Start Pretraining.')
    # loss_function = nn.KLDivLoss(reduction='mean')
    for epoch in tqdm.tqdm(range(200)):
        output, inputs = model(inputs)
        X_view = inputs["X_view"]
        X_hat = inputs["X_hat"]
        H_common = inputs["H_common"]
        M = inputs["M"]
        P_view = inputs["P_view"]
        loss_list = []
        for v in range(view):
            q_view = get_q_joint(H_common[M[:, v]])
            kl_loss = loss_function(p_joint=P_view[v], q_joint=q_view)
            loss_list.append(kl_loss)
            loss_list.append(F.mse_loss(X_hat[v][M[:, v]], X_view[v]))  # x_bar, x
        loss = sum(loss_list)/view
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        if epoch % 1000 == 0:
            print("mseloss loss", tot_loss)

    output, inputs = model(inputs)
    M = inputs["M"]
    H_common = inputs["H_common"]
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans_arch = KMeans(n_clusters=args.AnchorNum, n_init=100)
    for v in range(view):
        z_v = H_common[M[:, v]]
        kmeans.fit_predict(z_v.cpu().detach().data.numpy())
        model.aes[v].clusteringLayer.centroids.data = \
            torch.tensor(kmeans.cluster_centers_).to(device)

    kmeans_arch.fit_predict(H_common.cpu().detach().data.numpy())

    a_all = torch.tensor(kmeans_arch.cluster_centers_).to(device)
    c_allmin = calculate_c(H_common.T, a_all.T, gamma=args.gamma)
    model.cl_weight.data = c_allmin
    model.Al_weight.data = a_all.T

def fineTuning(inputs):
    loss_function = nn.KLDivLoss(reduction='mean')
    print('Start Self-supervised Learning.')
    for epoch in tqdm.tqdm(range(100)):
        mseloss = 0.
        kl_loss=0.
        output, inputs = model(inputs)
        X_hat = inputs["X_hat"]
        X_view = inputs["X_view"]
        M = inputs["M"]
        H_common = inputs["H_common"]
        H_view=inputs["H_view"]

        if epoch == 0:
            c_all = torch.clamp(model.cl_weight.data, 0, 1)
            U_common = cacluate_U(c_all)
            p_h = []
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            kmeans.fit_predict(U_common.cpu().detach().data.numpy())
            for v in range(view):
                U_common = U_common.to(device)
                q, p = make_qp(torch.tensor(U_common[M[:, v]]), kmeans.cluster_centers_)
                p = p.to(device)
                p_view = p @ p.T
                p_view = F.normalize(p_view)
                p_h.append(p_view)

        for v in range(view):
            mseloss = mseloss + torch.nn.functional.mse_loss(X_hat[v][M[:, v]], X_view[v])  # x_bar, x
            q_view = get_q_joint(H_common[M[:, v]])
            kl_loss = kl_loss + loss_function(q_view.log(), p_h[v])


        loss = 1 * mseloss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print('mse_loss: {:.4f}'.format(mseloss),
                  ',kl_loss: {:.4f}'.format(kl_loss))

    kmeans_arch = KMeans(n_clusters=args.AnchorNum, n_init=100)

    output, inputs =model(inputs)
    H_common = inputs["H_common"]

    kmeans_arch.fit_predict(H_common.cpu().detach().data.numpy())

    a_all = torch.tensor(kmeans_arch.cluster_centers_).to(device)
    c_allmin = calculate_c(H_common.T, a_all.T, gamma=args.gamma)
    model.cl_weight.data = c_allmin
    model.Al_weight.data = a_all.T


    c_all = torch.clamp(model.cl_weight.data, 0, 1)
    U_common = cacluate_U(c_all)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit_predict(U_common.cpu().detach().data.numpy())

    y_pred = kmeans.labels_
    nmi, ari, acc, pur, f1 = evaluate(y, y_pred)

    f1_mean = sum(f1) / len(f1)  # 计算平均 F1 分数


    print('Acc {:.4f}'.format(acc),', nmi {:.4f}'.format(nmi),
          ', pur {:.4f}'.format(pur),', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1_mean))


# datasets = ['MSRCv1', 'ORL-40', 'LandUse-21', 'Scene-15']
datasets = ['MSRCv1']
for data in datasets:
    args.dataset = data
    dataset, dims, view, data_size, class_num, labels = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # batch_size=args.batch_size,
        batch_size=data_size,
        shuffle=False
    )

    if args.dataset == "HW":
        args.AnchorNum = 50
        args.gamma = 0.1
        args.feature_dim = 10
        seed = 10
    if args.dataset == "COIL-20":
        args.AnchorNum = 20
        args.feature_dim = 256
        args.gamma = 1
        seed = 10
    if args.dataset == "Caltech101-20":
        args.AnchorNum = 20
        args.gamma = 1
        args.feature_dim = 512
        seed = 10
    if args.dataset == "MSRCv1":
        args.AnchorNum = 7
        args.gamma = 1
        args.feature_dim = 256
        seed = 10
    if args.dataset == "ORL-40":
        args.AnchorNum = 200
        args.gamma = 1
        seed = 10
    if args.dataset == "Cifar100":
        args.AnchorNum = 100
        args.gamma = 1
        seed = 10
    if args.dataset == "LandUse-21":
        args.AnchorNum = 84
        args.gamma = 100
        args.feature_dim = 256
        seed = 10
    if args.dataset == "Scene-15":
        args.AnchorNum = 30
        args.gamma = 0.1
        args.feature_dim = 256
        seed = 10
    if args.dataset == "Animal-50":
        args.AnchorNum = 100
        args.gamma = 100
        seed = 10
    if args.dataset == "USPS-MNIST":
        args.AnchorNum = 10
        args.gamma = 10
        seed = 10


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    setup_seed(seed)
    # etas = [0.1, 0.3, 0.5, 0.7, 0.9]
    etas = [0.1]
    for i in etas:
        args.eta = i
        T = 1
        for t in range(T):
            print(t + 1)
            print(args)
            inputs = preprocess(eta=args.eta)
            data = inputs["data"]
            y = data.Y
            X_view = inputs["X_view"]
            n_v = []
            for v in range(view):
                n_v.append(len(X_view[v]))
            model = AGDAN(data_size, args.AnchorNum, args.feature_dim, n_v, view, dims,
                             class_num, labels)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            pretrain(inputs)
            fineTuning(inputs)

