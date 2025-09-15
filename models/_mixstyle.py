import random
import torch
import torch.nn as nn
from ._styleextraction import StyleStatistics


"""
Code taken and slightly adapted from:
 https://github.com/KaiyangZhou/mixstyle-release/blob/master/reid/models/mixstyle.py
"""


class MixStyle(nn.Module):
    def __init__(
            self, 
            p=0.5, 
            alpha=0.5, 
            eps=1e-5, 
            mix='random',
            num_layers: int = 3,
            **kwargs
    ):
        """
        Args:
        :param p: probability of using MixStyle.
        :param alpha: parameter of the Beta distribution.
        :param eps: scaling parameter to avoid numerical issues.
        :param mix: how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._active = True
        self.num_layers = num_layers
        
        self.num_domains = kwargs.get('num_domains', None)
        self.domain_names = kwargs.get('domain_names', None)


    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'


    def set_activation_status(self, status=True):
        self._active = status


    def update_mix_method(self, mix='random'):
        self.mix = mix


    def forward(self, x: torch.Tensor, domain_labels=None, layer_idx=0):
        if not self.training or not self._active:
            return x
        
        # input validation
        if domain_labels is not None:
            assert x.shape[0] == domain_labels.shape[0], \
                f"Batch size mismatch: x={x.shape}, domains={domain_labels.shape}"
    
        B, C, H, W = x.shape
        if random.random() > self.p or B < 2:  # skip if batch too small
            return x

        mu = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        var = x.view(B, C, -1).var(dim=2).view(B, C, 1, 1)
        sig = (var + self.eps).sqrt()
    
        x_normed = (x - mu.detach()) / sig.detach()
    
        # Mixing mit shape checks
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)    # random mixing
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
    
        return x_normed * sig_mix + mu_mix


def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('random')

