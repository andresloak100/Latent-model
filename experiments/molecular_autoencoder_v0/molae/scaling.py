"""Linear-cost attention and unbounded positions — the two hard caps on scale.

Both blockers named in ROADMAP 6.2 live in the codec that actually works
(0.79 A held-out), not in the Perceiver branch that already solved them and
reconstructs at 9.3 A. So the fix is to give the working architecture the two
properties, not to switch architectures.

  1. ``nn.Embedding(max_res_pos)`` caps group indices at 1024 -- about 8k
     atoms. 1M atoms is ~125k groups, a 125x shortfall, and above the cap
     indices do not error, they CLAMP: group 1024 and group 90,000 receive an
     identical position. ``sinusoidal_index`` has no table and no cap.

  2. Dense self-attention over group tokens is O(R^2). At R = 125k that is
     1.56e10 pairs = 31 GB per head per layer in fp16, ~250 GB for the four
     heads and two layers currently configured. Dead on one GPU, which
     objective 4 makes binding rather than merely unfortunate.

The attention pattern here is local windows plus a few global tokens. Pure
sequence-local attention would be cheaper still and is wrong for this data:
sequence-adjacent residues are usually spatially adjacent, but the contacts
that determine a fold or an interface are exactly the ones that are not, and
a window cannot see them at any width worth having. Pooled global tokens
restore a long-range path at O(R.g), so cost stays linear while every group
keeps a route to every other group in two hops.

KNOWN LIMIT -- the long-range path is weak, and this is measured, not
suspected. Each global token mean-pools a contiguous segment, so a single
group's contribution is diluted by the segment length: at R = 125k and
g = 4, one group is 1/31250 of its summary. `tests/test_scaling.py` asserts
only that the path EXISTS and beats window-only, and that it is under 1e-2 --
the honest version of the claim.

The window's own receptive field grows by 2w per layer, which at the
configured two layers and w = 16 reaches 64 groups. That is ample at R = 8k
and not ample at R = 125k. Three ways out, in increasing order of cost:
raise g with R (it is linear, so g ~ sqrt(R) is affordable); dilate the
window between layers; or use spatial k-NN neighbourhoods instead of sequence
windows, which the ENCODER can do since it sees coordinates and the decoder
cannot since coordinates are what it predicts. Do not treat the current
setting as sufficient at 125k without measuring it there.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_index(idx: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal encoding of an integer index -> (..., d_model).

    Table-free, so neither the group count nor the latent count is bounded by
    a size chosen at construction time.
    """
    half = d_model // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=idx.device, dtype=torch.float32)
        / max(half, 1)
    )
    ang = idx.float().unsqueeze(-1) * freqs
    enc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if enc.shape[-1] < d_model:                       # odd d_model
        enc = torch.cat([enc, enc[..., :1]], dim=-1)
    return enc


class PositionEncoding(nn.Module):
    """Learned table or sinusoidal, chosen by config.

    Default stays learned so every existing checkpoint loads and every prior
    result reproduces. ``unbounded=True`` drops the table entirely.
    """

    def __init__(self, d_model: int, max_positions: int, unbounded: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_positions = max_positions
        self.unbounded = unbounded
        self.table = None if unbounded else nn.Embedding(max_positions, d_model)
        if unbounded:
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self.unbounded:
            return self.proj(sinusoidal_index(idx, self.d_model))
        return self.table(idx.clamp(max=self.max_positions - 1))


def _pad_to_multiple(x, window, value=0.0):
    """Right-pad the sequence axis so it splits into whole windows."""
    R = x.shape[1]
    rem = (-R) % window
    if rem == 0:
        return x, 0
    pad = x.new_full((x.shape[0], rem) + tuple(x.shape[2:]), value)
    return torch.cat([x, pad], dim=1), rem


class WindowedSelfAttention(nn.Module):
    """Local-window + global-token attention. O(R.(3w + g)), never O(R^2).

    Each query attends to its own window and the two adjacent ones (so the
    receptive field grows by 2w per layer and information still crosses the
    whole sequence in depth), plus ``n_global`` tokens pooled from the entire
    sequence, which give every group a two-hop path to every other group
    regardless of distance.

    Set ``window=0`` for exact dense attention -- used as the correctness
    oracle in tests and as the default so nothing existing changes.
    """

    def __init__(self, d_model, n_heads, ff_mult, dropout,
                 window: int = 0, n_global: int = 4):
        super().__init__()
        self.window = int(window)
        self.n_global = int(n_global)
        self.n_heads = n_heads
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )
        if self.window > 0 and self.n_global > 0:
            self.global_q = nn.Parameter(torch.randn(n_global, d_model) * 0.02)

    def _global_tokens(self, xn, key_padding_mask):
        """Mean-pool the sequence into n_global summary tokens.

        Pooling rather than attending keeps this O(R.g): a learned cross
        attention would be the same cost, but pooling cannot be starved by a
        bad initialisation and needs no extra parameters per head.
        """
        B, R, D = xn.shape
        g = self.n_global
        keep = (~key_padding_mask).to(xn.dtype) if key_padding_mask is not None \
            else xn.new_ones(B, R)
        # Contiguous segments, so each summary token owns a region of the
        # sequence instead of every token summarising the same thing.
        edges = torch.linspace(0, R, g + 1, device=xn.device).long()
        outs = []
        for i in range(g):
            a, b = int(edges[i]), int(max(edges[i + 1], edges[i] + 1))
            w = keep[:, a:b].unsqueeze(-1)
            outs.append((xn[:, a:b] * w).sum(1) / w.sum(1).clamp_min(1.0))
        return torch.stack(outs, dim=1) + self.global_q.unsqueeze(0)

    def forward(self, x, key_padding_mask=None):
        xn = self.n1(x)
        B, R, D = xn.shape
        w = self.window

        if w <= 0 or R <= w:                       # dense path (exact)
            a, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask,
                             need_weights=False)
            x = x + a
            return x + self.ff(self.n2(x))

        gtok = self._global_tokens(xn, key_padding_mask) if self.n_global > 0 else None

        xp, pad = _pad_to_multiple(xn, w)
        if key_padding_mask is None:
            kpm = xn.new_zeros(B, R, dtype=torch.bool)
        else:
            kpm = key_padding_mask
        kpm_p, _ = _pad_to_multiple(kpm.unsqueeze(-1).float(), w, value=1.0)
        kpm_p = kpm_p.squeeze(-1).bool()

        nw = xp.shape[1] // w
        q = xp.view(B, nw, w, D)
        # Keys are this window plus its two neighbours -> 3w keys per query.
        left = torch.roll(q, shifts=1, dims=1)
        right = torch.roll(q, shifts=-1, dims=1)
        left[:, 0] = 0.0                            # no wraparound across ends
        right[:, -1] = 0.0
        kv = torch.cat([left, q, right], dim=2)     # (B, nw, 3w, D)

        km = kpm_p.view(B, nw, w)
        kml = torch.roll(km, 1, dims=1); kml[:, 0] = True
        kmr = torch.roll(km, -1, dims=1); kmr[:, -1] = True
        kvm = torch.cat([kml, km, kmr], dim=2)      # (B, nw, 3w)

        if gtok is not None:
            gexp = gtok.unsqueeze(1).expand(B, nw, self.n_global, D)
            kv = torch.cat([kv, gexp], dim=2)
            kvm = torch.cat([kvm, kvm.new_zeros(B, nw, self.n_global)], dim=2)

        qf = q.reshape(B * nw, w, D)
        kf = kv.reshape(B * nw, kv.shape[2], D)
        mf = kvm.reshape(B * nw, kv.shape[2])
        # A window whose keys are ALL padding would produce NaN; unmask its own
        # slot so softmax has something to normalise over. Those rows are
        # padding themselves and are discarded below.
        mf[mf.all(dim=1), 0] = False

        a, _ = self.attn(qf, kf, kf, key_padding_mask=mf, need_weights=False)
        a = a.reshape(B, nw * w, D)[:, :R]

        x = x + a
        return x + self.ff(self.n2(x))


def make_group_attention(cfg, n_layers: int):
    """Attention stack for GROUP tokens, honouring the scaling config."""
    window = int(getattr(cfg, "attn_window", 0))
    n_global = int(getattr(cfg, "attn_global", 4))
    return nn.ModuleList([
        WindowedSelfAttention(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout,
                              window=window, n_global=n_global)
        for _ in range(n_layers)
    ])
