import argparse
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import build_loss_fn

def _read_shapes(dataloader) -> Tuple[int, int]:
    """
    Read feat_dim and pooled_dim from the first dataset sample.

    TODO: move this to NodeDataLoader, so we dont use disk operations in the model builder.
    But should be fine as it is now.
    """
    sample = dataloader.dataset[0]
    assert sample['x'].dim() == 2
    feat_dim = sample['x'].shape[1]
    assert sample['x_pool'].ndim == 1, "Expected x_pool to be 1-D [pooled_dim]"
    pooled_dim = sample['x_pool'].shape[0] 
    return feat_dim, pooled_dim


def _xavier_linear_init(module: nn.Module) -> None:
    """Xavier-init every nn.Linear in nn.module"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)

            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

            nn.init.xavier_uniform_(m.out_proj.weight)

            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

        if hasattr(m, "bias_k") and m.bias_k is not None:
            nn.init.zeros_(m.bias_k)
        if hasattr(m, "bias_v") and m.bias_v is not None:
            nn.init.zeros_(m.bias_v)


class BaseNodeGenerator(nn.Module):
    """
    Abstract base. 
    
    If you want to implement a new node generator, subclass this and implement the methods below.

    forward(x_pool, S, mask=None, ...) -> prediction [, aux]
    generate(x_pool, S, mask=None, noise=1.0) -> prediction
    loss(pred, target, mask, **kwargs) -> scalar loss

    But ofc, you can change as you like.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    @torch.no_grad()
    def init_output_bias(self, prior: torch.Tensor) -> None:
        """
        Set the output layer's bias to logit(prior) for binary features.
        """
        feat_dim = prior.shape[0]
        last = None
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.out_features == feat_dim:
                last = m
        if last is None or last.bias is None:
            return
        p = prior.detach().float().clamp(1e-4, 1 - 1e-4)
        last.bias.copy_(torch.log(p / (1 - p)))



class _SDPATransformerLayer(nn.Module):
    """
    Replacement for nn.TransformerEncoderLayer.
    Uses torch.nn.functional.scaled_dot_product_attention for self-attention.
    This is faster than nn.MultiheadAttention and supports Flash Attention.
    Should be O(n) in memory and O(n^2) in time, but Flash Attention is faster than the naive O(n^2) implementation.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, norm_first=True):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model
        self.norm_first = norm_first

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _sa(self, x, key_padding_mask):
        B, N, D = x.shape
        qkv = self.qkv_proj(x).view(B, N, 3, self.nhead, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Flash Attention requires bfloat16 or float16, 
        # so we cast to bfloat16 for speed and memory efficiency.
        orig_dtype = q.dtype
        q_h = q.to(torch.bfloat16)
        k_h = k.to(torch.bfloat16)
        v_h = v.to(torch.bfloat16)

        attn_mask = None

        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)

        out = F.scaled_dot_product_attention(
            q_h, k_h, v_h,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )        
        out = out.to(orig_dtype)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)

    def forward(self, x, src_key_padding_mask=None):
        if self.norm_first:
            x = x + self.drop(self._sa(self.norm1(x), src_key_padding_mask))
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop(self._sa(x, src_key_padding_mask)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class _SDPATransformer(nn.Module):
    def __init__(self, layer_factory, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(num_layers)])

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return x


class TransformerNodeGenerator(BaseNodeGenerator):
    """
    Node tokens from S rows -> self-attention -> cross-attention to x_pool.

    Padding mask support: if 'mask' is passed, it is propagated into both self- and cross-attention
    so padded positions dont distort real tokens' representations.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        dataloader,
        max_k: int,
        noise_dim: int = 16,
    ):
        super().__init__(args)

        self.max_k = max_k
        self.noise_dim = noise_dim

        self.latent_dim = getattr(args, 'latent_dim', 128)
        self.num_layers = getattr(args, 'num_layers', 2)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.nhead = getattr(args, 'nhead', 4)
        self.noise_during_training = getattr(args, 'noise_during_training', False)
        self.train_noise = getattr(args, 'train_noise', 1.0)

        self.feat_dim, self.pooled_dim = _read_shapes(dataloader)

        assert self.latent_dim % self.nhead == 0, f"latent_dim ({self.latent_dim}) must be divisible by nhead ({self.nhead})"

        # tokenization
        self.s_proj = nn.Linear(max_k, self.latent_dim)
        self.xpool_proj = nn.Linear(self.pooled_dim, self.latent_dim)

        # self-attention stack
        def make_layer():
            return _SDPATransformerLayer(
                d_model=self.latent_dim,
                nhead=self.nhead,
                dim_feedforward=self.latent_dim * 2,
                dropout=self.dropout,
                norm_first=True,
            )
        
        self.self_attn = _SDPATransformer(make_layer, num_layers=self.num_layers)

        # cross-attention.
        # We use nn.MultiheadAttention here because it supports a single memory token (x_pool).
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = self.latent_dim,
            num_heads = self.nhead,
            dropout = self.dropout,
            batch_first = True,
            )
        
        self.cross_norm = nn.LayerNorm(self.latent_dim)
        
        self.post_cross = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # output head
        self.out_proj = nn.Linear(self.latent_dim, self.feat_dim)

        if noise_dim > 0:
            self.noise_proj = nn.Linear(noise_dim, self.latent_dim)
        else:
            self.noise_proj = None

        self.loss_fn = build_loss_fn(args)


    def _encode(self, x_pool: torch.Tensor, S: torch.Tensor, mask: Optional[torch.Tensor], noise: float,) -> torch.Tensor:
        """
        mask: [B, max_N] bool, True = real node, false = padded node..
        """
        B, max_N, _ = S.shape

        # node tokens
        tokens = self.s_proj(S) # [B, max_N, latent_dim]

        # optional noise
        if self.noise_proj is not None and noise > 0.0:
            noise = torch.randn(B, max_N, self.noise_dim, device=S.device) * noise
            tokens = tokens + self.noise_proj(noise)

        # key_padding_mask True -> ignores
        key_padding_mask = (~mask) if mask is not None else None

        # self-attention across nodes (with padding mask -> BUGFIX)
        tokens = self.self_attn(tokens, src_key_padding_mask=key_padding_mask)

        # cross-attention: nodes attend to x_pool (single memory token)
        memory = self.xpool_proj(x_pool).unsqueeze(1) # [B, 1, latent_dim]
        attn_out, _ = self.cross_attn(
            query = tokens,
            key = memory,
            value = memory,
        )
      
        tokens = self.cross_norm(tokens + attn_out) # residual + norm
        tokens = tokens + self.post_cross(tokens) # post-attn FFN + residual

        pred = self.out_proj(tokens) # [B, max_N, feat_dim]

        if mask is not None:
            pred = pred * mask.unsqueeze(-1)

        return pred


    def forward(self, x_pool: torch.Tensor, S: torch.Tensor, mask: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        noise = self.train_noise if self.noise_during_training else 0.0
        return self._encode(x_pool, S, mask=mask, noise=noise)

    @torch.no_grad()
    def generate(self, x_pool: torch.Tensor, S: torch.Tensor, mask: Optional[torch.Tensor] = None, noise: float = 1.0, ) -> torch.Tensor:
        self.eval()
        return self._encode(x_pool, S, mask=mask, noise=noise)

    def loss(self, pred, target, mask):
        return self.loss_fn(pred, target, mask)

    def init_weights(self):
        _xavier_linear_init(self)

