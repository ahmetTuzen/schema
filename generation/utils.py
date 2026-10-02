import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.seed import seed_everything

import time
import resource
import logging

from contextlib import contextmanager

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed all RNGs via PyG's seed_everything (covers random, numpy, torch, torch.cuda). 
    Optional deterministic mode enables determinism at a significant performance cost.

    Same with ..clustering.utils.set_seed
    """

    seed_everything(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def _masked_reduce(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Shared masked-mean reducer."""
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp(min=1)


def masked_mse_loss(pred, target, mask):
    loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1) # [B, max_N]
    return _masked_reduce(loss, mask)


def masked_mae_loss(pred, target, mask):
    loss = F.l1_loss(pred, target, reduction='none').mean(dim=-1) # [B, max_N]
    return _masked_reduce(loss, mask)


def masked_huber_loss(pred, target, mask, delta: float = 1.0):
    loss = F.huber_loss(pred, target, reduction='none', delta=delta).mean(dim=-1)
    return _masked_reduce(loss, mask)


def masked_cosine_loss(pred, target, mask):
    cos_sim = F.cosine_similarity(pred, target, dim=-1) # [B, max_N]
    loss = 1.0 - cos_sim
    return _masked_reduce(loss, mask)


def masked_combined_loss(pred, target, mask, delta: float = 1.0, cosine_weight: float = 0.1):
    huber = masked_huber_loss(pred, target, mask, delta=delta)
    cosine = masked_cosine_loss(pred, target, mask)
    return huber + cosine_weight * cosine


def masked_bce_loss(pred, target, mask):
    """Binary features: 'pred' are logits, 'target' is 0/1."""
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none').mean(dim=-1)
    return _masked_reduce(loss, mask)


def build_loss_fn(args):
    """
    Build a masked-loss callable during training. 
    
    args.node_loss : 'mse' | 'mae' | 'huber' | 'cosine' | 'combined'
    args.huber_delta : delta for huber and combined
    args.cosine_weight : weight of the cosine term in combined

    """
    losses = {
        'mse': masked_mse_loss,
        'mae': masked_mae_loss,
        'huber': masked_huber_loss,
        'cosine': masked_cosine_loss,
        'combined': masked_combined_loss,
    } 

    # Binary features are modelled as independent Bernoulli variables.
    if getattr(args, 'feature_type', 'auto') == 'binary':
        return masked_bce_loss

    loss_name = getattr(args, 'node_loss', 'huber').lower()
    if loss_name not in losses:
        raise ValueError(f"Unknown loss '{loss_name}'. Choose from: {list(losses)}")

    delta = getattr(args, 'huber_delta', 1.0)
    cosine_weight = getattr(args, 'cosine_weight', 0.1)

    if loss_name == 'huber':
        return partial(masked_huber_loss, delta=delta)
    if loss_name == 'combined':
        return partial(masked_combined_loss, delta=delta, cosine_weight=cosine_weight)
    return losses[loss_name]

def log_peak_rss(tag: str = ""):
    """Log rss in MB. On Linux ru_maxrss is in kilobytes. Only tested in linux."""
    import resource, logging
    logger = logging.getLogger(__name__)
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(f"[mem]{' ' + tag if tag else ''} peak RSS: {kb / 1024:.1f} MB")


_stage_logger = logging.getLogger('stage')

@contextmanager
def stage_metrics(name: str):
    """
    Log wall time, peak host RSS and peak CUDA allocation for a stage.
    
    Use it with a with statement, e.g.:
    with stage_metrics('xx'):
        yy
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        yield
    finally:
        secs = time.time() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            _stage_logger.info(f"[metrics] {name}: {secs:.1f}s | peak RSS {rss_mb:.0f} MB | peak CUDA {gpu_mb:.0f} MB")
        else:
            _stage_logger.info(f"[metrics] {name}: {secs:.1f}s | peak RSS {rss_mb:.0f} MB")


def _xavier_linear_init(module: nn.Module) -> None:
    """Xavier-init every nn.Linear in nn.module and biases to 0"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _xavier_init_param(p: torch.Tensor) -> None:
    """Xavier-init 2D parameter tensor (weight matrix)"""
    assert p.dim() == 2, f"expected 2-D parameter, got shape {tuple(p.shape)}"
    nn.init.xavier_uniform_(p)

class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)