"""Perceiver vs spatial-segment bottleneck on the DEVIATION task, matched SCALARS.

The intended architecture (atom-native Perceiver, generic slots) has never been
tested on the deviation task. This cell tests it against a deterministic spatial-
segment bottleneck (E1-file generalised via a space-filling curve) and against ANM,
AT EQUAL SCALARS = L*d (NOT equal tokens: a Perceiver L*d latent must be compared
to ANM with L*d coefficients, else it gets d-fold more budget).

Arms (only the bottleneck differs; static features -> both enc & dec, never through
the bottleneck):
  P      Perceiver: L learned slot queries cross-attend to N atoms. KEY = static
         (identity + ref position), VALUE = displacement projection. The key/value
         split is the fix for the zero-mean bug (a uniform pool averages aligned
         zero-mean displacement to ~0 -> frame-invariant latent). Permutation-
         equivariant; no ordering.
  P+SFC  P plus a sinusoidal encoding of the atom's Morton space-filling-curve rank
         appended to static features (spatial locality without forced assignment).
  S+SFC  deterministic segment: atom at SFC rank i -> slot floor(i*L/N); slot value
         = mean of its atoms' displacement projections; decoder reads own slot + 2
         neighbours.

Data/split identical to the generalisation study: mdCATH 320K, CA, frozen-7 test
(held-out rep0 second half), training pool = the rest. Target = Kabsch-aligned
displacement, mean-centred by the first-half mean (matches null/ANM/cross/within).
"""
import argparse, glob, os, math, numpy as np, torch, torch.nn as nn
CACHE = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace/perceiver_cache"   # numpy-only (no h5py on the node)
FROZEN = "/network/scratch/j/jacob-junqi.tian/mae_provisional/latent-model/experiments/molecular_autoencoder_v0/outputs/cluster/armf_frozen_test.txt"
TEMP, R0, R1 = "320", "0", "1"
BUDGETS = [64, 128, 256, 512]                                      # scalars = L*d, d=8 fixed
DVAL = 8
DM = 64                                                            # hidden width


def kabsch(traj, mask):
    ref = traj[0]; rcen = ref[mask].mean(0); Q = ref[mask] - rcen
    out = np.empty_like(traj)
    for t in range(len(traj)):
        P = traj[t]; pcen = P[mask].mean(0); Pc = P[mask] - pcen
        U, S, Vt = np.linalg.svd(Pc.T @ Q); dd = np.sign(np.linalg.det(Vt.T @ U.T))
        out[t] = (P - pcen) @ (Vt.T @ np.diag([1, 1, dd]) @ U.T).T + rcen
    return out


def anm_modes(ca, cutoff=13.0):                                   # ALL non-trivial CA modes, ascending
    n = len(ca); H = np.zeros((3 * n, 3 * n))
    for i in range(n):
        dv = ca - ca[i]; d = np.linalg.norm(dv, axis=1)
        for j in np.where((d < cutoff) & (d > 1e-6))[0]:
            if j <= i: continue
            u = dv[j] / d[j]; b = np.outer(u, u)
            H[3*i:3*i+3, 3*j:3*j+3] -= b; H[3*j:3*j+3, 3*i:3*i+3] -= b
            H[3*i:3*i+3, 3*i:3*i+3] += b; H[3*j:3*j+3, 3*j:3*j+3] += b
    w, V = np.linalg.eigh(H)
    return V[:, 6:]                                               # (3n, 3n-6)


def pca_modes(dc, L):
    _, _, Vt = np.linalg.svd(dc, full_matrices=False)
    return Vt[:L].T


def resid(ev, M):
    rec = ev @ M @ M.T
    return float(np.sqrt(((rec - ev) ** 2).sum() / (ev.shape[0] * (ev.shape[1] // 3))))


def morton_rank(ref):                                            # frame-stable SFC rank from reference coords
    x = np.empty_like(ref)
    for a in range(3):
        lo, hi = ref[:, a].min(), ref[:, a].max()
        x[:, a] = np.round((ref[:, a] - lo) / (hi - lo + 1e-9) * 1023)
    xi = x.astype(np.int64); code = np.zeros(len(ref), np.int64)
    for b in range(10):
        code |= ((xi[:, 0] >> b) & 1) << (3 * b)
        code |= ((xi[:, 1] >> b) & 1) << (3 * b + 1)
        code |= ((xi[:, 2] >> b) & 1) << (3 * b + 2)
    order = np.argsort(code, kind="stable"); rank = np.empty(len(ref), np.int64)
    rank[order] = np.arange(len(ref))
    return rank


def sfc_sin(rank, N, nf=8):                                      # sinusoidal encoding of SFC rank/N
    p = rank[:, None] / N * (2 ** np.arange(nf))[None] * math.pi
    return np.concatenate([np.sin(p), np.cos(p)], 1).astype(np.float32)   # (N, 2*nf)


# ---------------- models ----------------
class Perceiver(nn.Module):
    def __init__(self, sdim, L, use_sfc):
        super().__init__()
        self.L = L
        self.slots = nn.Parameter(torch.randn(L, DM) * 0.02)
        self.key = nn.Sequential(nn.Linear(sdim, DM), nn.GELU(), nn.Linear(DM, DM))
        self.val = nn.Sequential(nn.Linear(3, DM), nn.GELU(), nn.Linear(DM, DM))
        self.to_lat = nn.Linear(DM, DVAL)
        self.q = nn.Sequential(nn.Linear(sdim, DM), nn.GELU(), nn.Linear(DM, DM))
        self.lk = nn.Linear(DVAL, DM); self.lv = nn.Linear(DVAL, DM)
        self.out = nn.Sequential(nn.Linear(DM, DM), nn.GELU(), nn.Linear(DM, 3))

    def encode(self, static, disp):                             # static (N,sdim), disp (B,N,3) -> z (B,L,DVAL)
        K = self.key(static); attn = torch.softmax(self.slots @ K.T / math.sqrt(DM), -1)   # (L,N), frame-stable
        V = self.val(disp)                                     # (B,N,DM)
        z = torch.einsum("ln,bnd->bld", attn, V)
        return self.to_lat(z)

    def decode(self, static, z):                               # z (B,L,DVAL) -> disp (B,N,3)
        Qd = self.q(static); Kd = self.lk(z); Vd = self.lv(z)
        a = torch.softmax(torch.einsum("nd,bld->bnl", Qd, Kd) / math.sqrt(DM), -1)
        return self.out(torch.einsum("bnl,bld->bnd", a, Vd))


class Segment(nn.Module):
    def __init__(self, sdim, L):
        super().__init__()
        self.L = L
        self.val = nn.Sequential(nn.Linear(3, DM), nn.GELU(), nn.Linear(DM, DVAL))
        self.dec = nn.Sequential(nn.Linear(sdim + 3 * DVAL, DM), nn.GELU(),
                                 nn.Linear(DM, DM), nn.GELU(), nn.Linear(DM, 3))

    def encode(self, static, disp, assign):                    # deterministic segment mean
        B = disp.shape[0]; v = self.val(disp)                  # (B,N,DVAL)
        z = torch.zeros(B, self.L, DVAL, device=disp.device)
        z.index_add_(1, assign, v)
        cnt = torch.bincount(assign, minlength=self.L).clamp(min=1).float()[None, :, None]
        return z / cnt

    def decode(self, static, z, assign):
        own = z[:, assign]                                     # (B,N,DVAL)
        left = z[:, (assign - 1).clamp(min=0)]; right = z[:, (assign + 1).clamp(max=self.L - 1)]
        st = static[None].expand(z.shape[0], -1, -1)
        return self.dec(torch.cat([st, own, left, right], -1))


# ---------------- data ----------------
def load_system(dom, eval_split):
    fp = f"{CACHE}/{dom}.npz"
    if not os.path.exists(fp): return None
    dd = np.load(fp)
    c0 = dd["c0"].astype(np.float64)
    if c0.shape[1] < 20: return None
    c1 = dd["c1"].astype(np.float64) if (eval_split and "c1" in dd) else None
    n = c0.shape[1]; a0 = kabsch(c0, np.ones(n, bool)); d0 = (a0 - a0[0]).reshape(len(a0), -1)
    ref = a0[0]; h = len(d0) // 2
    mean0 = (d0[:h] if eval_split else d0).mean(0)
    stat = ((ref - ref.mean(0)) / (ref.std(0) + 1e-6)).astype(np.float32)
    rank = morton_rank(ref)
    out = dict(dom=dom, n=n, ref=ref, stat=stat, rank=rank, mean0=mean0,
               d=d0, h=h, sfc=sfc_sin(rank, n))
    if eval_split:
        ev = d0[h:] - mean0; out["ev"] = ev
        out["Mpca"] = pca_modes(d0[:h] - mean0, 64)
        if c1 is not None:
            a1 = kabsch(c1, np.ones(n, bool)); d1 = (a1 - a1[0]).reshape(len(a1), -1)
            out["Mcross"] = pca_modes(d1 - d1.mean(0), 64)
        out["Manm_all"] = anm_modes(ref)
    return out


def static_feat(s, use_sfc):
    st = s["stat"]
    if use_sfc: st = np.concatenate([st, s["sfc"]], 1)
    return torch.tensor(st, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    frozen = [d for d in open(FROZEN).read().split() if d]
    alldoms = sorted(os.path.basename(x)[:-4] for x in glob.glob(f"{CACHE}/*.npz"))
    train_doms = [d for d in alldoms if d not in frozen]
    if args.smoke:
        train_doms = train_doms[:4]; frozen = frozen[:2]; BUD = [64, 128]
    else:
        BUD = BUDGETS
    print(f"[perceiver-seg] device={dev}  train {len(train_doms)}  test(frozen) {len(frozen)}  epochs={args.epochs}")

    train = [s for s in (load_system(d, False) for d in train_doms) if s is not None]
    test = [s for s in (load_system(d, True) for d in frozen) if s is not None]

    # ---- G3: references recomputed here, on the frozen 7, same held-out frames ----
    null = np.mean([np.sqrt((s["ev"] ** 2).sum() / (s["ev"].shape[0] * s["n"])) for s in test])
    anm_b = {}
    for b in BUD:
        vals = []
        for s in test:
            La = min(b, s["Manm_all"].shape[1]); vals.append(resid(s["ev"], s["Manm_all"][:, :La]))
        anm_b[b] = np.mean(vals)
    cross = np.mean([resid(s["ev"], s["Mcross"]) for s in test if "Mcross" in s])
    within = np.mean([resid(s["ev"], s["Mpca"]) for s in test])

    print("\n=== GUARD BLOCK (read first) ===")
    print("  G3 references on frozen 7 (absolute A, held-out rep0 2nd half):")
    print(f"     null {null:.2f}   cross(L64) {cross:.2f}   within(L64) {within:.2f}")
    print("     ANM at matched scalars: " + "  ".join(f"{b}s={anm_b[b]:.2f}" for b in BUD))
    caps = [f"{b}(cap@{min(b, min(s['Manm_all'].shape[1] for s in test))})" for b in BUD
            if b > min(s['Manm_all'].shape[1] for s in test)]
    if caps: print(f"     NOTE ANM capped by 3N-6 on small systems: {caps}")

    def train_cell(arm, L, use_sfc):
        sdim = 3 + (16 if use_sfc else 0)
        model = (Segment(sdim, L) if arm == "S+SFC" else Perceiver(sdim, L, use_sfc)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        cache = {}
        for s in train:
            st = static_feat(s, use_sfc).to(dev)
            asg = torch.tensor((s["rank"] * L // s["n"]).clip(0, L - 1), device=dev) if arm == "S+SFC" else None
            cache[s["dom"]] = (st, asg)
        for ep in range(args.epochs):
            for s in np.random.permutation(train):
                st, asg = cache[s["dom"]]
                dfull = s["d"] - s["mean0"]; T = dfull.shape[0]
                idx = np.random.choice(T, min(args.batch, T), replace=False)
                disp = torch.tensor(dfull[idx].reshape(len(idx), s["n"], 3), dtype=torch.float32, device=dev)
                opt.zero_grad()
                if arm == "S+SFC":
                    z = model.encode(st, disp, asg); pred = model.decode(st, z, asg)
                else:
                    z = model.encode(st, disp); pred = model.decode(st, z)
                (((pred - disp) ** 2).mean()).backward(); opt.step()
        return model

    def eval_cell(model, arm, L, use_sfc):
        rmsd, zero, cos_list, slotstd = [], [], [], []
        for s in test:
            st = static_feat(s, use_sfc).to(dev)
            asg = torch.tensor((s["rank"] * L // s["n"]).clip(0, L - 1), device=dev) if arm == "S+SFC" else None
            ev = torch.tensor(s["ev"].reshape(s["ev"].shape[0], s["n"], 3), dtype=torch.float32, device=dev)
            with torch.no_grad():
                if arm == "S+SFC":
                    z = model.encode(st, ev, asg); pred = model.decode(st, z, asg)
                    zpred = model.decode(st, torch.zeros_like(z), asg)
                else:
                    z = model.encode(st, ev); pred = model.decode(st, z)
                    zpred = model.decode(st, torch.zeros_like(z))
            rmsd.append(float(torch.sqrt(((pred - ev) ** 2).sum() / (ev.shape[0] * s["n"]))))
            zero.append(float(torch.sqrt(((zpred - ev) ** 2).sum() / (ev.shape[0] * s["n"]))))
        # G1 frame-variance on ONE test system: 5 frames
        s0 = test[0]; st = static_feat(s0, use_sfc).to(dev)
        fr = np.linspace(0, s0["ev"].shape[0] - 1, 5).astype(int)
        evc = torch.tensor((s0["ev"][fr]).reshape(5, s0["n"], 3), dtype=torch.float32, device=dev)
        asg = torch.tensor((s0["rank"] * L // s0["n"]).clip(0, L - 1), device=dev) if arm == "S+SFC" else None
        with torch.no_grad():
            zc = model.encode(st, evc, asg) if arm == "S+SFC" else model.encode(st, evc)
        zf = zc.reshape(5, -1); zf = zf / (zf.norm(dim=1, keepdim=True) + 1e-9)
        cos = float((zf @ zf.T).masked_select(~torch.eye(5, dtype=bool, device=dev)).mean())
        sstd = float((zc.std(0).mean() / (zc.std() + 1e-9)))
        return np.mean(rmsd), np.mean(zero), cos, sstd

    print("\n=== RESULTS ===")
    print(f"  {'arm':7s}{'L':>4}{'d':>3}{'scalars':>8}{'heldA':>7}{'vsANM':>7}{'zeroA':>7}{'G1cos':>8}{'slotStd':>8}  VOID")
    best = None
    for b in BUD:
        L = b // DVAL
        for arm, use_sfc in [("P", False), ("P+SFC", True), ("S+SFC", True)]:
            model = train_cell(arm, L, use_sfc)
            r, z, cos, sstd = eval_cell(model, arm, L, use_sfc)
            void = cos > 0.9999 or abs(r - z) < 0.02
            vs = r - anm_b[b]
            print(f"  {arm:7s}{L:>4}{DVAL:>3}{b:>8}{r:>7.2f}{vs:>+7.2f}{z:>7.2f}{cos:>8.4f}{sstd:>8.3f}  "
                  f"{'VOID' if void else ''}")
            if not void and (best is None or r < best[0]): best = (r, arm, b)
            if not void and r < anm_b[b]:
                print(f"     ** {arm}@{b}s BEATS ANM ({r:.2f} < {anm_b[b]:.2f}) -- learned general codec beats the physics prior **")
    print(f"\n  reference: null {null:.2f} | ANM " + " ".join(f"{b}s:{anm_b[b]:.2f}" for b in BUD) +
          f" | cross {cross:.2f} | within {within:.2f}")
    if best: print(f"  best non-void learned cell: {best[1]}@{best[2]}s = {best[0]:.2f} A (ANM@{best[2]}s = {anm_b[best[2]]:.2f})")


if __name__ == "__main__":
    main()
