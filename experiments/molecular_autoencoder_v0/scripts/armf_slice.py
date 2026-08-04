"""Arm F -- the thinnest end-to-end viability slice for the deployed design.

Compress ONLY the per-timestep deviation; keep reference geometry + element +
graph identity as STATIC CONDITIONING to encoder AND decoder, never through the
bottleneck. Tests the load-bearing uncertainty the arm A-E cross never touched:
does compress->decode of DISPLACEMENT beat the zero-displacement null.

VIABILITY TEST, not progress on any objective: ~4k atoms not 1e6, ns not ms,
zero reactions.

Validity requirements (self-supervised objectives that shape WHAT, not HOW):
- CONDITIONING DROPOUT: mask contiguous regions of the static conditioning during
  training (reuse molae.masking._region_mask over canonical order) so the decoder
  cannot just echo reference geometry -- it must route dropped regions through the
  latent. drop_rate 0 is the control that shows the effect is real.
- LATENT NOISE: Gaussian perturbation of the latent in training; eval reports
  reconstruction vs perturbation magnitude -- the decoder's robustness budget,
  which tells step 2 how accurate the diffusion model must be.

Baseline: ZERO-DISPLACEMENT NULL (predict reference unchanged) -- the headline
number to beat. Control: zeroed latent at eval proves the latent does the work.
Local-frame output (--local-frame): displacement predicted in a per-atom frame
built from the REFERENCE bond geometry (equivariant); toggle for a global-aligned
fallback (data is CA-aligned, so equivariance is not exercised here).
"""
import argparse, glob, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from molae.graph_identity import graph_atom_features       # noqa: E402
from molae.masking import _region_mask                      # noqa: E402

COORD_SCALE = 10.0
EVAL_NOISE = [0.0, 0.5, 1.0, 2.0, 4.0]     # x per-slot latent std, robustness curve


def sinusoidal(idx, d):
    idx = idx.float().unsqueeze(-1)
    div = torch.exp(torch.arange(0, d, 2, device=idx.device).float() * (-math.log(10000.0) / d))
    out = torch.zeros(*idx.shape[:-1], d, device=idx.device)
    out[..., 0::2] = torch.sin(idx * div); out[..., 1::2] = torch.cos(idx * div)
    return out


def build_frames(ref, bonds, N):
    """Per-atom local frame from REFERENCE bond geometry (static). Identity where
    an atom has <2 usable neighbours or they are degenerate."""
    nbr = [[] for _ in range(N)]
    for i, j in bonds:
        if 0 <= i < N and 0 <= j < N:
            nbr[i].append(j); nbr[j].append(i)
    R = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    for i in range(N):
        if len(nbr[i]) < 2:
            continue
        v1 = ref[nbr[i][0]] - ref[i]; v2 = ref[nbr[i][1]] - ref[i]
        n1 = np.linalg.norm(v1)
        if n1 < 1e-4:
            continue
        e1 = v1 / n1
        v2p = v2 - (v2 @ e1) * e1; n2 = np.linalg.norm(v2p)
        if n2 < 1e-4:
            continue
        e2 = v2p / n2; e3 = np.cross(e1, e2)
        R[i] = np.stack([e1, e2, e3], axis=1)     # columns = frame axes
    return R


class ArmF(nn.Module):
    def __init__(self, n_elem=64, d_model=128, L=64, latent_dim=8, n_heads=4):
        super().__init__()
        self.d = d_model; self.L = L
        self.elem = nn.Embedding(n_elem, d_model)
        self.ref_proj = nn.Linear(3, d_model)
        self.rank_proj = nn.Linear(d_model, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)   # conditioning dropped
        self.disp_proj = nn.Linear(3, d_model)
        self.enc_mlp = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.query = nn.Parameter(torch.randn(L, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Linear(d_model, 3))

    def static_feat(self, batch, cond_drop=None):
        h = self.elem(batch["element_idx"]) + self.ref_proj(batch["ref"] / COORD_SCALE)
        h = h + self.rank_proj(sinusoidal(batch["canonical_rank"], self.d))
        if cond_drop is not None:                       # replace dropped conditioning with mask token
            h = torch.where(cond_drop.unsqueeze(-1), self.mask_token.expand_as(h), h)
        return h

    def encode(self, batch, cond_drop=None):
        disp = (batch["target"] - batch["ref"]) / COORD_SCALE
        h = self.enc_mlp(self.static_feat(batch, cond_drop) + self.disp_proj(disp))
        q = self.query.unsqueeze(0).expand(h.shape[0], -1, -1)
        z, _ = self.pool(q, h, h)
        return self.to_latent(z)

    def decode(self, z, batch, cond_drop=None, zero_latent=False):
        lat = self.from_latent(torch.zeros_like(z) if zero_latent else z)
        q = self.static_feat(batch, cond_drop)
        ctx, _ = self.read(q, lat, lat)
        local = self.head(q + ctx) * COORD_SCALE                        # (B,N,3) local-frame disp
        if "frames" in batch:
            local = torch.einsum("nij,bnj->bni", batch["frames"], local)  # -> global
        return batch["ref"] + local

    def forward(self, batch, cond_drop=None, latent_noise=0.0, zero_latent=False):
        z = self.encode(batch, cond_drop)
        if latent_noise > 0:
            z = z + latent_noise * z.detach().std() * torch.randn_like(z)
        return self.decode(z, batch, cond_drop, zero_latent)


def load_systems(data_dir, local_frame, dev):
    syss = []
    for p in sorted(glob.glob(f"{data_dir}/*.npz")):
        d = np.load(p, allow_pickle=True)
        el = np.asarray(d["element_idx"], np.int64)
        bonds = np.asarray(d["bonds"], np.int64).reshape(-1, 2)
        g = graph_atom_features(el, None, bonds, None)
        coords = np.asarray(d["coords"], np.float32)
        s = dict(name=Path(p).stem, coords=torch.from_numpy(coords).to(dev), N=el.shape[0],
                 element_idx=torch.from_numpy(el).to(dev),
                 canonical_rank=torch.from_numpy(np.asarray(g["canonical_rank"], np.int64)).to(dev))
        if local_frame:
            s["frames"] = torch.from_numpy(build_frames(coords[0], bonds, el.shape[0])).to(dev)
        syss.append(s)
    return syss


def rmsd(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1).mean(-1)).mean()


def make_batch(s, frames):
    n = len(frames)
    b = dict(ref=s["coords"][0:1].expand(n, -1, -1), target=s["coords"][frames],
             element_idx=s["element_idx"].unsqueeze(0).expand(n, -1),
             canonical_rank=s["canonical_rank"].unsqueeze(0).expand(n, -1))
    if "frames" in s:
        b["frames"] = s["frames"]
    return b


def region_drop(s, n, rate, span, gen):
    if rate <= 0:
        return None
    rp = s["canonical_rank"].unsqueeze(0).expand(n, -1)      # contiguous in canonical order
    m = torch.ones(n, s["N"], device=s["coords"].device)
    return _region_mask(rp, m, rate, span, gen).bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--drop-rate", type=float, default=0.25)
    ap.add_argument("--drop-span", type=int, default=8)
    ap.add_argument("--train-noise", type=float, default=0.3)     # latent noise during training
    ap.add_argument("--local-frame", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    syss = load_systems(args.data_dir, bool(args.local_frame), dev)
    T = syss[0]["coords"].shape[0]
    tr = list(range(1, int(T * 0.8))); va = list(range(int(T * 0.8), T))
    print(f"[armF] {len(syss)} sys T={T} L={args.L} d={args.latent_dim} drop={args.drop_rate} "
          f"train_noise={args.train_noise} local_frame={bool(args.local_frame)} dev={dev}")

    m = ArmF(L=args.L, latent_dim=args.latent_dim).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    print(f"[armF] params {sum(p.numel() for p in m.parameters()):,}")

    for ep in range(args.epochs):
        m.train(); tot = 0.0
        for si in torch.randperm(len(syss)).tolist():
            s = syss[si]
            fr = [tr[i] for i in torch.randperm(len(tr))[:8].tolist()]
            b = make_batch(s, fr)
            cd = region_drop(s, len(fr), args.drop_rate, args.drop_span, gen)
            pred = m(b, cond_drop=cd, latent_noise=args.train_noise)
            loss = ((pred - b["target"]) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep % 60 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:4d} train_loss {tot/len(syss):.3f}")

    m.eval()
    print(f"\n[armF] VAL per system (L={args.L}) -- improvement over zero-displacement null:")
    print(f"  {'system':10s}{'N':>6}{'null':>8}{'armF':>8}{'redux%':>8}{'zeroLat':>9}")
    rows = []
    with torch.no_grad():
        for s in syss:
            b = make_batch(s, va)
            null = float(rmsd(b["ref"], b["target"]))
            r = float(rmsd(m(b), b["target"]))
            rz = float(rmsd(m(b, zero_latent=True), b["target"]))
            redux = 100 * (null - r) / null
            rows.append((s["name"], s["N"], null, r, redux, rz))
            print(f"  {s['name']:10s}{s['N']:>6}{null:>8.3f}{r:>8.3f}{redux:>7.1f}%{rz:>9.3f}")
        import statistics as st
        print(f"\n  MEAN reduction vs null: {st.mean([x[4] for x in rows]):.1f}%  "
              f"(rule: >=50% at L=64 across N -> step 2)")
        print(f"  latent test: armF {st.mean([x[3] for x in rows]):.3f} vs zero-latent "
              f"{st.mean([x[5] for x in rows]):.3f} (latent earns keep if armF much lower)")

        # LATENT-NOISE robustness curve (what step 2's diffusion accuracy must meet)
        print(f"\n  latent-noise robustness (mean val RMSD vs eval perturbation x slot-std):")
        for ns in EVAL_NOISE:
            rs = []
            for s in syss:
                b = make_batch(s, va); z = m.encode(b)
                if ns > 0:
                    z = z + ns * z.std() * torch.randn_like(z)
                rs.append(float(rmsd(m.decode(z, b), b["target"])))
            print(f"    noise {ns:>4.1f}x -> RMSD {st.mean(rs):.3f}")

        # PER-REGION conditioning reliance: mask a conditioning region, localise degradation
        print(f"\n  per-region conditioning reliance (RMSD rise IN masked region vs baseline):")
        rel = []
        for s in syss:
            b = make_batch(s, va); base = m(b)
            cd = region_drop(s, len(va), 0.25, args.drop_span, gen)
            if cd is None: continue
            pred = m(b, cond_drop=cd)
            din = float(rmsd(pred[cd], b["target"][cd]))
            dbase = float(rmsd(base[cd], b["target"][cd]))
            rel.append(din - dbase)
        if rel:
            print(f"    mean in-region RMSD rise when conditioning dropped: {st.mean(rel):.3f} A "
                  f"(low => latent already carries it; high => decoder leans on conditioning)")


if __name__ == "__main__":
    main()
