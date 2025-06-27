import random
import torch
import torch.nn as nn
# from contextlib import contextmanager
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
            mix='random',   # TODO wird das überhaupt noch gebraucht?
            num_domains: int = 4, 
            num_layers: int = 3,
            style_mode="average",  # "average", "selective", "paired", "attention"
            layer_config=None,     # Für "selective" oder "paired"
            store_stats=True      # Ob Statistiken gespeichert werden sollen
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
        self.store_stats = store_stats

        # StyleStatistics integration
        if num_domains is not None and store_stats:
            self.style_stats = StyleStatistics(
                num_domains=num_domains,
                num_layers=num_layers,
                mode=style_mode,
                layer_config=layer_config
            )
        else:
            self.style_stats = None

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._active = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    """
    def forward(self, x: torch.Tensor, domain_labels: torch.Tensor = None, layer_idx: int = 0):
        if domain_labels is not None and domain_labels.shape[0] != x.shape[0]:
            raise RuntimeError(
                f"Critical shape mismatch: x={x.shape}, domains={domain_labels.shape}\n"
                f"Layer: {layer_idx}, Training: {self.training}\n"
                f"Possible data loader issue or model architecture problem."
            )
        
        if not self.training or not self._active:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # save stats for the current layer (if active)
        if domain_labels is not None and self.style_stats is not None:
            for domain in torch.unique(domain_labels):
                mask = domain_labels == domain
                self.style_stats._update(
                    domain_idx=domain_labels,
                    layer_idx=layer_idx,
                    mu=mu[mask],
                    sig=sig[mask]
                )

        lam = self.beta.sample((B, 1, 1, 1))
        lam = lam.to(x.device)

        if self.mix == 'random':        
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1)  # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)

        return x_normed * sig_mix + mu_mix
    """

    def forward(self, x: torch.Tensor, domain_labels=None, layer_idx=0):
        if not self.training or not self._active:
            return x
        
        # Input validation
        if domain_labels is not None:
            assert x.shape[0] == domain_labels.shape[0], \
                f"Batch size mismatch: x={x.shape}, domains={domain_labels.shape}"
    
        B, C, H, W = x.shape
        if random.random() > self.p or B < 2:  # Skip if batch too small
            return x

        # Original MixStyle Logic mit sicherer Implementierung
        mu = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        var = x.view(B, C, -1).var(dim=2).view(B, C, 1, 1)
        sig = (var + self.eps).sqrt()
    
        # Sicherstellen dass wir nicht versehentlich B ändern
        x_normed = (x - mu.detach()) / sig.detach()
    
        # Stats tracking nur wenn shapes passen
        if domain_labels is not None and self.style_stats is not None:
            if domain_labels.shape[0] == B:  # Double-check
                self.style_stats._update(domain_labels, layer_idx, mu, sig)
    
        # Mixing mit shape checks
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B)    # random mixing!
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


def crossdomain_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('crossdomain')


"""
@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)
"""