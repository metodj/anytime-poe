import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class BaseDUN(nn.Module):
    def __init__(self, net, regression=True):
        super().__init__()

        self.net = net
        self.regression = regression
        self.num_layers = len(self.net.layers) + 1

        if self.regression:
            self.logσ = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, return_depth_preds=False):
        raise NotImplementedError


class DUN(BaseDUN):
    def __init__(self, net, regression=True):
        super().__init__(net=net, regression=regression)

        self.prior_probs = nn.Parameter(1/self.num_layers * torch.ones(self.num_layers), requires_grad=False)
        self.q_logits = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)

    def forward(self, x, return_depth_preds=False):
        depth_preds = self.net(x)  # [D, B, O]
        probs = F.softmax(self.q_logits, dim=0).view(-1, 1, 1)  # [D, 1, 1]

        if self.regression:
            μ = torch.sum(depth_preds * probs, dim=0)  # [B, O]
            σ2_m = torch.sum(depth_preds**2 * probs, dim=0) - μ**2  # [B, O]
            σ2_d = torch.exp(self.logσ * 2)
            σ2 = σ2_m + σ2_d  # [B, O]

            if not return_depth_preds:
                return μ, σ2, σ2_d
            else:
                return μ, σ2, σ2_d, depth_preds
        else:
            depth_probs = F.softmax(depth_preds, dim=-1)
            probs = torch.sum(depth_probs * probs, dim=0)  # [B, O]

            if not return_depth_preds:
                return probs
            else:
                return probs, depth_probs

    def ELBO(self, x, y, N):
        preds = self.net(x)  # [D, B, O]
        D, B, O = preds.shape

        if self.regression:
            σ = torch.exp(self.logσ)
            likelihood = Normal(preds.view(-1, O), σ)
            lls_per_depth = likelihood.log_prob(y.repeat(D, 1))  # [D * B]
        else:
            likelihood = Categorical(logits=preds.view(-1, O))
            lls_per_depth = likelihood.log_prob(y.repeat(D,))  # [D * B]

        lls_per_depth = lls_per_depth.view(D, -1)  # [D, B]
        probs = F.softmax(self.q_logits, dim=0).view(-1, 1)  # [D, 1]
        ll = (lls_per_depth * probs).sum() * N / B

        kl = (probs * torch.log(probs / self.prior_probs)).sum()

        return ll - kl


class ProductDUN(BaseDUN):
    def __init__(self, net, regression=True):
        super().__init__(net=net, regression=regression)

        self.ens_weights = nn.Parameter(torch.ones(self.num_layers), requires_grad=False)

    def forward(self, x, return_depth_preds=False):
        depth_preds = self.net(x)  # [D, B, O]
        probs = F.softmax(self.ens_weights, dim=0).view(-1, 1, 1)  # [D, 1, 1]

        if self.regression:
            μ = depth_preds.mean(axis=0) # [B, O]
            σ2 = torch.exp(self.logσ * 2)

            if not return_depth_preds:
                return μ, σ2
            else:
                return μ, σ2, depth_preds
        else:
            raise NotImplementedError()

    def product_loss(self, x, y, N):
        preds = self.net(x)  # [D, B, O]
        D, B, O = preds.shape

        if self.regression:
            mu = preds.mean(axis=0)
            σ = torch.exp(self.logσ)
            likelihood = Normal(mu.view(-1, O), σ)
            lls_per_depth = likelihood.log_prob(y)  # B
        else:
            raise NotImplementedError()

        ll = lls_per_depth.sum() * N / B
        return ll
