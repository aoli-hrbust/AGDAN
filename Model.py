from backbone import GCN_Encoder_SDIMC, Imputer
from torch.nn.parameter import Parameter
from typing import List
from torch_utils import (
    Tensor,
    nn,
    torch,
    F,
)

class SingleViewModel(nn.Module):

    def __init__(self, n_clusters, AnchorNum, feature_dim, n_v):
        super(SingleViewModel, self).__init__()
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clusteringLayer = ClusteringLayer(n_clusters, feature_dim)
        self.A_weight=torch.nn.Parameter(torch.Tensor( feature_dim,AnchorNum),requires_grad=False)
        self.c_weight = torch.nn.Parameter(torch.Tensor(AnchorNum, n_v), requires_grad=False)



    def computegcn(self, x, Af):
        D = torch.sum(Af, dim=1)
        D = 1 / D
        D = torch.sqrt(D)
        D = torch.diag(D)
        ATi = torch.matmul(torch.matmul(D, Af), D)
        z_refine = torch.matmul(ATi, torch.matmul(ATi, x))

        return z_refine, ATi

    def forward(self, H):
        q=self.clusteringLayer(H)

        return self.c_weight, q


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.enc_1 = nn.Linear(input_dim, 500)
        self.enc_2 = nn.Linear(500, 500)
        self.enc_3 = nn.Linear(500, 2000)
        self.enc_4 = nn.Linear(2000, feature_dim)
    def forward(self, x):
        enc_d1 = F.relu(self.enc_1(x))
        enc_d2 = F.relu(self.enc_2(enc_d1))
        enc_d3 = F.relu(self.enc_3(enc_d2))
        enc_d4 = self.enc_4(enc_d3)
        return enc_d4
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)
class AGDAN(torch.nn.Module):
    def __init__(self, data_size, AnchorNum, feature_dim, n_v, view, input_dim,  class_num, labels):
        super(AGDAN, self).__init__()
        # self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.n_v = n_v
        self.view = view
        self.class_num = class_num
        self.labels = labels
        self.data_size = data_size
        self.AnchorNum = AnchorNum
        self.Al_weight = torch.nn.Parameter(torch.Tensor(self.feature_dim, self.AnchorNum), requires_grad=False)
        self.cl_weight = torch.nn.Parameter(torch.Tensor(self.AnchorNum, self.data_size), requires_grad=False)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for v in range(view):
            self.encoders.append(GCN_Encoder_SDIMC(input_dim[v], feature_dim).to(self.device))
            self.decoders.append(Imputer(input_dim[v], feature_dim).to(self.device))
        aes = []

        for v in range(self.view):
            aes.append(SingleViewModel(
                n_clusters=self.class_num,
                AnchorNum=self.AnchorNum,
                feature_dim=self.feature_dim,
                n_v=self.n_v[v]))

        self.aes = nn.ModuleList(aes)
        self.fusion = AttentionLayer(self.feature_dim, self.view, self.device)

    def forward(self, inputs: dict):
        X_view: List[Tensor] = inputs["X_view"]
        M: Tensor = inputs["M"]
        S_view: List[Tensor] = inputs["S_view"]

        sampleNum, viewNum = M.shape
        H_view = [None] * viewNum
        for v in range(viewNum):
            h_tilde = self.encoders[v](X_view[v], S_view[v])
            H_view[v] = h_tilde
        inputs["H_view"] = H_view

        # Fusion
        H_common = self.fusion(inputs)

        outputs = []
        for v in range(viewNum):
            outputs.append(self.aes[v](H_common[M[:, v]]))

        outputs.append(self.Al_weight)
        outputs.append(self.cl_weight)

        Al_weight = self.Al_weight.to(self.device)
        cl_weight = self.cl_weight.to(self.device)
        z_bar = torch.matmul(Al_weight, cl_weight)
        z_bar = z_bar.T

        inputs["H_hat"] = z_bar
        inputs["H_common"] = H_common


        X_hat = [None] * viewNum
        for v in range(viewNum):
            X_hat[v] = self.decoders[v](H_common)
        inputs["X_hat"] = X_hat

        return outputs, inputs




class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, feature_dim):
        super(ClusteringLayer, self).__init__()
        self.centroids = Parameter(torch.Tensor(n_clusters, feature_dim),requires_grad=True)

    def forward(self, x):
        q = 1.0 / (1 + torch.sum(
            torch.pow(x.unsqueeze(1) - self.centroids, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()

        return q

class AttentionLayer(nn.Module):
    def __init__(self, latent_dim, view, device):
        super(AttentionLayer, self).__init__()
        self._latent_dim = latent_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(self._latent_dim * view, self._latent_dim * view),
            nn.BatchNorm1d(self._latent_dim * view),
            nn.ReLU(),
            nn.Linear(self._latent_dim * view, self._latent_dim * view),
            nn.BatchNorm1d(self._latent_dim * view),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self._latent_dim * view, view, bias=True)

    def forward(self, inputs, tau=10.0):
        X = inputs["X"]
        H_view = inputs["H_view"]
        M = inputs["M"]
        view = len(H_view)
        sampleNum, _ = X[0].shape
        _, feature_dim = H_view[0].shape
        H_common = torch.zeros(sampleNum, feature_dim).to(self.device)

        Hs = []
        for v in range (view):
            H = torch.zeros(sampleNum, feature_dim).to(self.device)
            H[M[:, v]] += H_view[v]
            Hs.append(H)
        h = torch.cat(Hs, dim=1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1).to(self.device)
        for v in range(view):
            H_common += e[:, v].unsqueeze(1) * Hs[v]
        H_common = H_common / torch.sum(M, 1, keepdim=True)
        H_common = F.normalize(H_common)
        return H_common
