import os

import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import stiefel_optimizer
from dataloader import *
from utils import (
    Lin_View,
    simplex_coordinates1,
    assign_soft_clusters,
    ams,
)
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from encoder_decoder import ClusterGAN_Dec, ClusterGAN_Enc, SimpleNet1, SimpleNet2


class RKM_Stiefel(nn.Module):
    """Defines the Stiefel RKM model and its loss functions"""

    def __init__(
        self,
        ipVec_dim,
        args,
        nChannels=1,
        recon_loss=nn.MSELoss(reduction="sum"),
        ngpus=1,
    ):
        super(RKM_Stiefel, self).__init__()
        self.ipVec_dim = ipVec_dim
        self.ngpus = ngpus
        self.args = args
        self.nChannels = nChannels
        self.recon_loss = recon_loss
        self._initialise_cluster_codes(self.args.k)
        self.centering_term: torch.Tensor | None = None
        self.eval_centering_term = None
        self.Dinv = None

        # Initialize Manifold parameter NOTE: parameter needs to be defined as s x df (transpose of defined as in paper) in order to work with CayleyAdam
        self.Ut = nn.Parameter(
            nn.init.orthogonal_(torch.Tensor(self.args.h_dim, self.args.x_fdim2))
        )
        # self.lambda_param = nn.Parameter(nn.init.orthogonal_(torch.Tensor(self.args.h_dim, 1)))

        # self.rot_parameter = self._init_rot_parameter() #alpha trainable
        self.rot_parameter = torch.Tensor([0])  # alpha fixed

       

        # self.encoder = Net1(self.nChannels, self.args, self.cnn_kwargs)
        # self.decoder = Net3(self.nChannels, self.args, self.cnn_kwargs)

        #self.encoder = ClusterGAN_Enc(self.nChannels, self.args)
        #self.decoder = ClusterGAN_Dec(self.nChannels, self.args)

        self.encoder = SimpleNet1(self.ipVec_dim, self.args)
        self.decoder = SimpleNet2(self.ipVec_dim, self.args)

    def compute_eval_centering_term(self, xtrain):
        N = xtrain.size(0)

        phi = self.encoder(xtrain)  # features

        # Calculate degree matrix
        D = phi @ (phi.t() @ torch.ones((N, 1)).to(self.args.proc))
        Dinv = torch.pow(D.flatten(), -1)
        # Calculate centering term

        self.eval_centering_term = 1 / Dinv.sum() * (phi.t() * Dinv).sum(dim=1)


    def forward(self, x: torch.Tensor, trained_phi: torch.Tensor | None =None):
        N = x.size(0)

        x
        # Find feature vectors
        phi = self.encoder(x)

        if trained_phi is None:
            phi_train = phi
        else:
            phi_train = trained_phi

        N_train = phi_train.size(0)

        # Calculate degree matrix
        D = phi @ (phi_train.t() @ torch.ones((N_train, 1)).to(self.args.proc))
        Dinv = torch.pow(D.flatten(), -1)

        # Calculate centering term
        if self.training:
            batch_centering_term = 1 / Dinv.sum() * (phi.t() * Dinv).sum(dim=1)
            self.centering_term = batch_centering_term if self.centering_term is None else (
            self.args.alpha * self.centering_term + (1 - self.args.alpha) * batch_centering_term
            )

        # weighted feature centering
        phi = phi - self.centering_term.detach()

        # Calculate score and latent variables
        e = phi @ self.Ut.t()
        h = torch.t(e.t() * Dinv)

        # Reconstruction
        x_tilde = self.decoder(phi @ self.Ut.t() @ self.Ut)

        # Calculate distances to prototypes
        codes = self.cluster_codes()

        if self.args.k == 2:
            # Calculate score values for clustering
            z = e[:, : self.args.k]
            # Calculate Euclidean distance dcos between score values Z and prototypes codes
            dcos = torch.norm(z[:, None] - codes, dim=2)
        else:
            # Calculate score values for clustering
            z = e[:, : self.args.k - 1]
            # Calculate Cosine distances to prototypes
            dcos = torch.ones(
                (z.shape[0], z.shape[1] + 1), device=self.args.proc
            ) - z @ torch.t(codes) / torch.outer(
                torch.sqrt(torch.diag(z @ torch.t(z))),
                torch.sqrt(torch.diag(codes @ torch.t(codes))),
            )
            # Normalise by maximum cosine distance
            dcos = dcos / (1 - 1 / (self.args.k - 1))

        return phi, e, h, x_tilde, dcos, Dinv

    def svd(self, x):
        phi = self.encoder(x)
        u, _, _ = torch.svd(phi.t() @ phi)
        u = u[:, : self.args.h_dim]
        self.Ut = nn.Parameter(u.t())
        return self.Ut

    def encode(self, x):

        # Find feature vectors
        phi = self.encoder(x)
        phi = phi - self.centering_term

        e = phi @ self.Ut.t()

        return phi, e

  
    def _initialise_cluster_codes(self, k):
        if k == 2:
            self.initial_cluster_codes = torch.Tensor([[1, 0], [-1, 0]]).to(
                self.args.proc
            )
        else:
            self.initial_cluster_codes = (
                torch.from_numpy(simplex_coordinates1(k - 1)).t().to(self.args.proc)
            )
        self._cluster_codes = self.initial_cluster_codes


    def cluster_codes(self):
        # self._cluster_codes = self.initial_cluster_codes @ self._rotation_matrix().t()
        return self._cluster_codes

    def update_cluster_codes(self, Phi):

        # calculate score variable
        E = Phi @ self.Ut.t()
        Z = E[:, : self.args.k - 1]
        # Assign clusters
        cluster_labels, _ = assign_soft_clusters(E)
        for i in range(self.args.k):
            self._cluster_codes[i] = torch.mean(Z[cluster_labels == i], dim=0)

        # return self._cluster_codes

    def _init_rot_parameter(self):
        # TODO generalise for self.k not 3
        return nn.Parameter(nn.init.orthogonal_(torch.Tensor(1, 1)))


# Accumulate trainable parameters in 2 groups:
# 1. Manifold_params 2. Network param
def param_state(model):
    param_g, param_e1 = [], []
    for name, param in model.named_parameters():
        if param.requires_grad and name != "manifold_param":
            param_e1.append(param)
        elif name == "manifold_param":
            param_g.append(param)
    return param_g, param_e1


def stiefel_opti(stief_param, lrg=1e-4):
    dict_g = {
        "params": stief_param,
        "lr": lrg,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "stiefel": True,
    }
    return stiefel_optimizer.AdamG([dict_g])  # CayleyAdam


def final_compute(model, args, ct, device=torch.device("cuda")):
    """Utility to re-compute Ut. Since some datasets could exceed the GPU memory limits, some intermediate
    variables are saved  on HDD, and retrieved later"""
    if not os.path.exists("oti/"):
        os.makedirs("oti/")

    args.shuffle = False
    x, _, _, _ = get_dataloader(args)

    # Compute feature-vectors
    for i, sample_batch in enumerate(tqdm(x)):
        torch.save(
            {"oti": model.encoder(sample_batch[0].to(device))},
            "oti/oti{}_checkpoint.pth_{}.tar".format(i, ct),
        )

    # Load feature-vectors
    phi = torch.Tensor([]).to(device)
    for i in range(0, len(x)):
        phi = torch.cat(
            (phi, torch.load("oti/oti{}_checkpoint.pth_{}.tar".format(i, ct))["oti"]),
            dim=0,
        )
    os.system("rm -rf oti/")

    N = phi.size(0)
    K = phi @ phi.t()
    D = torch.sum(K, 1)
    Dinv = torch.pow(d, -1)
    print(torch.min(Dinv))
    # oneN = torch.ones(N, 1).to(device)
    # phi = phi - 1/(oneN.t() @ Dinv @ oneN) * oneN.t() @ Dinv @ phi  # weighted feature centering

    dphi = torch.empty(phi.shape, device=device)
    for i in range(N):
        dphi[i, :] = phi[i, :] * Dinv[i]

    phic = phi - 1 / torch.sum(Dinv) * torch.sum(dphi, dim=0)

    # TODO: do SVD or not in final compute?

    # if YES
    # u, _, _ = torch.svd(phi.t() @ dphi)
    # u = u[:, :args.h_dim]
    # with torch.no_grad():
    #     model.manifold_param.masked_scatter_(model.manifold_param != u.t(), u.t())
    # if No
    u = model.manifold_param.t()

    # e_codes = model.cluster_codes()
    # h_codes = u[:,:args.k-1]
    # h = torch.mm(torch.t(phi.t()*dinv), u.to(device)), e, u
    e = phic @ u
    h = torch.t(e.t() * Dinv)
    return h, e, u, phi
