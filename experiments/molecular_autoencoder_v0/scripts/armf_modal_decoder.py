"""Modal decoder: displacement_i = B_i(structure) @ z.

PROPOSED ARM, not a replacement. Drops into armf_atlas_dm.py in place of `Codec`
(same constructor signature, same encode/decode/z0/code/forward contract) so it
can run as an ablation against the current decoder on identical data, splits and
guards.

WHY THIS SHAPE
--------------
The current decoder reaches the atoms like this:

    q = q_tok(stat)                        # per-atom structure embedding
    a = dec(q, lat, lat)                   # cross-attention to the latent
    g, b = film(lat.mean(1))               # global scale/shift
    h = LN(q + a) * (1 + g) + b
    disp = out(h + dff(h))

At L=1 the attention has a single key, so `a` is the SAME vector for every atom.
The latent therefore reaches the output through exactly three global objects --
`a`, `g`, `b` -- modulating a shared nonlinear function of `q_i`. Every
per-atom distinction has to be recovered from `q_i` alone, and `a` is added
immediately before a LayerNorm, which compresses precisely the component the
latent contributes.

Three measured facts say that shape is the problem:

1. 86-91% of latent variance at DM=256 encodes WHICH SYSTEM, not how it moves
   (INBOX 012b). Nothing stops it: `a`, `g` and `b` are free to carry identity,
   and identity is useful to a decoder that must infer per-atom behaviour from a
   generic nonlinearity.
2. The code uses ~15 effective dimensions regardless of width -- 10.8 at DM=16,
   ~15-17 at DM=256. Extra width is not being converted into extra
   conformational content.
3. FVE ~0.155 at DM=256 against per-system PCA-16 at ~0.53. The codec is far
   below a SIXTEEN-mode linear fit, and PCA's advantage is not depth -- it is
   that a displacement field is a linear combination of modes, which PCA
   represents exactly and this decoder must discover through a nonlinearity.

THE CHANGE
----------
Predict a per-atom basis from structure and let the latent be coefficients on it:

    B_i = basis(q_i)  in  R^{3 x d}          # from static features only
    disp_i = B_i @ z                          # linear in z

Consequences, in order of how much they matter here:

* IDENTITY CANNOT OCCUPY THE CODE. Everything system-specific lives in B, which
  is computed from the reference structure the decoder already has. `z` is
  dimensionally unable to say which protein this is -- it can only say how far
  along each mode the system currently sits. This is the architectural version
  of the INBOX 012b subtraction, and unlike `encode(x) - z0` it is exact rather
  than first-order.
* IT GENERALISES THE BASELINES RATHER THAN COMPETING WITH THEM. ANM fixes B from
  the Hessian; per-system PCA fits B to the target's own trajectory. This LEARNS
  B from structure: strictly more expressive than ANM, and zero-shot unlike PCA.
* L=1 IS NATIVE. No attention degeneracy to work around -- one token of width d
  is exactly d coefficients on a d-dimensional learned basis.

THE RISK, STATED BEFORE THE RESULT
----------------------------------
Collective modes are NONLOCAL. `q_tok` is a per-atom map over static features,
so a basis built from it alone can only express modes that are functions of an
atom's own attributes. If the bilinear arm underperforms the current decoder,
the first hypothesis is NOT that bilinearity is wrong -- it is that the basis
network cannot see enough structure. `ctx_layers > 0` adds k-NN message passing
so B_i depends on a neighbourhood; the repo already caches k-NN pairs
(precompute_graph_features.py). Escalate that before abandoning the form.

Linearity in z is a real restriction, and it is the point: PCA is linear and
reaches 0.53 where this decoder reaches 0.155.

UNTESTED. Written without cluster access; dry-run the __main__ path before
submitting, per the standing rule that a smoke test calling functions directly
never executes __main__.
"""

import torch
import torch.nn as nn


class ModalCodec(nn.Module):
    """Interface-compatible with armf_atlas_dm.Codec.

    Args mirror Codec: (Fs, L, dm, heads, dlat, sub_z0). Two additions:
      ctx_layers -- rounds of k-NN message passing before the basis head, so B_i
                    can depend on a neighbourhood rather than on atom i alone.
      tie_encoder -- encode by projecting displacement onto the SAME basis
                     (analysis/synthesis pair) instead of via attention.
    """

    def __init__(self, Fs, L, dm, heads=4, dlat=None, sub_z0=False,
                 ctx_layers=0, tie_encoder=False):
        super().__init__()
        self.sub_z0 = sub_z0
        self.dm = dm
        self.dlat = dlat if (dlat and dlat < dm) else dm
        self.L = L
        self.tie_encoder = tie_encoder
        self.ctx_layers = ctx_layers
        # coefficients seen by the basis: the whole code, flattened over tokens
        self.ncoef = L * self.dlat

        # ---- structure trunk: per-atom embedding of static features ----
        self.q_tok = nn.Linear(Fs, dm)
        self.q_ff = nn.Sequential(nn.Linear(dm, dm * 2), nn.GELU(), nn.Linear(dm * 2, dm))
        self.q_ln = nn.LayerNorm(dm)
        self.ctx = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dm, dm), nn.GELU(), nn.Linear(dm, dm))
            for _ in range(ctx_layers)
        ])
        self.ctx_ln = nn.ModuleList([nn.LayerNorm(dm) for _ in range(ctx_layers)])

        # ---- basis head: B_i in R^{3 x ncoef} ----
        # small init so early training starts near zero displacement rather than
        # at a large random field, which otherwise dominates the first epochs
        self.basis = nn.Linear(dm, 3 * self.ncoef)
        nn.init.normal_(self.basis.weight, std=0.02)
        nn.init.zeros_(self.basis.bias)

        # ---- encoder ----
        if not tie_encoder:
            self.lat = nn.Parameter(torch.randn(L, dm) * 0.02)
            self.in_tok = nn.Linear(Fs + 3, dm)
            self.enc = nn.MultiheadAttention(dm, heads, batch_first=True)
            self.sa = nn.MultiheadAttention(dm, heads, batch_first=True)
            self.l1, self.l2, self.l3 = (nn.LayerNorm(dm) for _ in range(3))
            self.ff = nn.Sequential(nn.Linear(dm, dm * 2), nn.GELU(), nn.Linear(dm * 2, dm))
            self.down = nn.Linear(dm, self.dlat) if self.dlat < dm else None
        else:
            # analysis is the adjoint of synthesis; one learned scale per coefficient
            self.coef_scale = nn.Parameter(torch.ones(self.ncoef))

    # ---------- structure trunk ----------

    def _struct(self, stat, knn=None):
        """(B, N, Fs) -> (B, N, dm). knn: optional (B, N, K) neighbour indices."""
        q = self.q_ln(self.q_tok(stat))
        q = q + self.q_ff(q)
        for layer, ln in zip(self.ctx, self.ctx_ln):
            if knn is None:
                break
            nb = torch.gather(
                q.unsqueeze(1).expand(-1, knn.shape[1], -1, -1), 2,
                knn.unsqueeze(-1).expand(-1, -1, -1, q.shape[-1]),
            ).mean(2)
            q = ln(q + layer(torch.cat([q, nb], -1)))
        return q

    def basis_of(self, stat, knn=None):
        """(B, N, Fs) -> (B, N, 3, ncoef). The learned modes, from structure alone."""
        q = self._struct(stat, knn)
        return self.basis(q).view(*stat.shape[:2], 3, self.ncoef)

    # ---------- codec contract ----------

    def encode(self, stat, disp, knn=None):
        """Returns the (B, L, dlat) code -- the object a generator would model."""
        if self.tie_encoder:
            Bm = self.basis_of(stat, knn)                       # (B,N,3,C)
            z = torch.einsum('bnic,bni->bc', Bm, disp)          # analysis
            z = z * self.coef_scale / max(stat.shape[1], 1) ** 0.5
            return z.view(stat.shape[0], self.L, self.dlat)
        tok = self.in_tok(torch.cat([stat, disp], -1))
        lat = self.lat.unsqueeze(0).expand(stat.shape[0], -1, -1)
        lat = self.l1(lat + self.enc(lat, tok, tok)[0])
        lat = self.l2(lat + self.sa(lat, lat, lat)[0])
        lat = self.l3(lat + self.ff(lat))
        return self.down(lat) if self.down is not None else lat

    def decode(self, stat, lat, knn=None):
        Bm = self.basis_of(stat, knn)                            # (B,N,3,C)
        z = lat.reshape(lat.shape[0], self.ncoef)                # (B,C)
        return torch.einsum('bnic,bc->bni', Bm, z)               # (B,N,3)

    def z0(self, stat, knn=None):
        """Code from ZERO displacement. Exactly zero by construction when
        tie_encoder=True -- analysis is linear in disp -- which is the point:
        the identity offset cannot exist. Kept for interface parity and as a
        regression test on the untied variant."""
        return self.encode(stat, torch.zeros(stat.shape[0], stat.shape[1], 3,
                                             device=stat.device, dtype=stat.dtype), knn)

    def code(self, stat, disp, knn=None):
        z = self.encode(stat, disp, knn)
        return z - self.z0(stat, knn) if self.sub_z0 else z

    def forward(self, stat, disp, knn=None):
        return self.decode(stat, self.code(stat, disp, knn), knn)


# ---------------------------------------------------------------------------
# Diagnostics this arm makes available that the attention decoder does not.
# ---------------------------------------------------------------------------

@torch.no_grad()
def basis_orthogonality(model, stat, knn=None):
    """Gram matrix of the learned modes, mass-weighted over atoms.

    A well-conditioned basis has near-diagonal Gram. Heavy off-diagonal mass
    means modes are redundant -- the code has fewer usable directions than its
    width, which is the participation-ratio finding (PR ~15 at every DM) read
    directly off the decoder instead of inferred from latent statistics.

    Returns (G, offdiag_frac) with G normalised to unit diagonal.
    """
    Bm = model.basis_of(stat, knn)                                # (B,N,3,C)
    B_ = Bm.reshape(Bm.shape[0], -1, Bm.shape[-1])                # (B,3N,C)
    G = torch.einsum('bpc,bpd->bcd', B_, B_)
    d = torch.diagonal(G, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    G = G / d.unsqueeze(-1) / d.unsqueeze(-2)
    eye = torch.eye(G.shape[-1], device=G.device).expand_as(G)
    off = (G - eye * G).abs().sum((-2, -1)) / (G.shape[-1] * (G.shape[-1] - 1))
    return G, off


@torch.no_grad()
def effective_modes(model, stat, knn=None, thresh=0.9):
    """How many learned modes carry `thresh` of the basis's total mass.

    The decoder-side analogue of rank90, and unlike rank90 it is a property of
    the ARCHITECTURE rather than of a per-system PCA fit -- so it is not subject
    to the in-sample bias or the n_eff limit that made rank90 a floor. If this
    saturates well below dlat, the token is over-provisioned regardless of what
    the training curve does.
    """
    Bm = model.basis_of(stat, knn)
    B_ = Bm.reshape(Bm.shape[0], -1, Bm.shape[-1])
    s = torch.linalg.svdvals(B_) ** 2
    c = torch.cumsum(s, -1) / s.sum(-1, keepdim=True).clamp_min(1e-12)
    return (c < thresh).sum(-1) + 1
