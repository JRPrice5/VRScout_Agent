import math
from typing import Optional

import torch
from torch.distributions import Distribution
import torch.nn.functional as F


class TanhDiagNormal(Distribution):
    """
    Diagonal Gaussian in pre-tanh space with tanh squashing to [-1, 1].
    Parameterized by mean (mu) and log_std.
    Exposes: rsample(), sample(), log_prob(), entropy(), mean, mode.
    Shapes: params [..., D] -> outputs [..., D]
    """
    has_rsample = True
    arg_constraints = {}

    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor, min_log_std: float = -5.0, max_log_std: float = 2.0):
        assert mu.shape == log_std.shape
        self.mu = mu
        self.log_std = log_std.clamp(min=min_log_std, max=max_log_std)
        self.std = torch.exp(self.log_std)
        super().__init__(batch_shape=mu.shape[:-1], event_shape=mu.shape[-1:], validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.mu)

    @property
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mu)

    def _pre_tanh_sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.std * eps

    def rsample(self) -> torch.Tensor:
        z = self._pre_tanh_sample()
        return torch.tanh(z)

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        clipped = torch.clamp(value, -1 + eps, 1 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        log_unnorm = -0.5 * (((pre_tanh - self.mu) / (self.std + 1e-8)) ** 2)
        log_norm = -self.log_std - 0.5 * math.log(2 * math.pi)
        log_prob_gauss = (log_unnorm + log_norm).sum(dim=-1)
        log_det = torch.log(1 - clipped * clipped + 1e-6).sum(dim=-1)
        return log_prob_gauss - log_det

    def entropy(self) -> torch.Tensor:
        H_gauss = 0.5 * (1.0 + math.log(2 * math.pi)) + self.log_std
        return H_gauss.sum(dim=-1)


class BernoulliMulti(Distribution): #TODO check whether it fit our boolean states (24-27)
    """
    Vectorized independent Bernoulli across last dimension.
    Parameterized by logits [..., D].
    """
    has_rsample = False
    arg_constraints = {}

    def __init__(self, logits: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None):
        assert (logits is None) ^ (probs is None), "Specify exactly one of logits or probs"
        if logits is not None:
            self.logits = logits
            self.probs = torch.sigmoid(logits)
        else:
            self.probs = probs
            self.logits = torch.log(torch.clamp(probs, 1e-6, 1 - 1e-6) / (1 - torch.clamp(probs, 1e-6, 1 - 1e-6)))
        super().__init__(batch_shape=self.probs.shape[:-1], event_shape=self.probs.shape[-1:], validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        return self.probs

    @property
    def mode(self) -> torch.Tensor:
        return (self.probs > 0.5).float()

    def sample(self) -> torch.Tensor:
        return torch.bernoulli(self.probs)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        logp = -F.binary_cross_entropy_with_logits(self.logits, value, reduction='none')
        return logp.sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        p = torch.clamp(self.probs, 1e-6, 1 - 1e-6)
        return (-p * torch.log(p) - (1 - p) * torch.log(1 - p)).sum(dim=-1)


class DiagNormal(Distribution):
    """
    Unsquashed diagonal Gaussian N(mu, diag(std^2)).
    Parameterized by mean (mu) and log_std. Shapes: [..., D].
    """
    has_rsample = True

    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor, min_log_std: float = -5.0, max_log_std: float = 2.0):
        assert mu.shape == log_std.shape
        self.mu = mu
        self.log_std = log_std.clamp(min=min_log_std, max=max_log_std)
        self.std = torch.exp(self.log_std)
        super().__init__(batch_shape=mu.shape[:-1], event_shape=mu.shape[-1:], validate_args=False)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def mode(self) -> torch.Tensor:
        return self.mu

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.std * eps

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        z = (value - self.mu) / (self.std + 1e-8)
        log_unnorm = -0.5 * (z * z)
        log_norm = -self.log_std - 0.5 * math.log(2 * math.pi)
        return (log_unnorm + log_norm).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 * (1.0 + math.log(2 * math.pi)) + self.log_std).sum(dim=-1)

