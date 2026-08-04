"""SECONDARY B: scaling & timing curve for the search economics.

Report vs N: sparse ANM eigensolve (Lanczos, leading ~64 modes -- NO dense
diagonalisation), encode, decode, one diffusion step, peak memory. Confirm Lanczos
matches dense for small N. Extrapolate to 1e6 atoms and give a candidates-per-hour
figure for the evolutionary-search design.

Timing is CPU here (no local GPU); the neural steps (diffusion, decode) would be
GPU-accelerated, the eigensolve is sparse-solver bound. Assumptions stated inline.
"""
import time, resource, numpy as np, torch, torch.nn as nn, math
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
CUT, L, Tdiff = 13.0, 64, 100
NS = [200, 500, 1000, 2000, 5000, 10000, 20000]


def make_structure(N):                                            # jittered cubic lattice ~ CA density
    side = int(np.ceil(N ** (1 / 3))) + 2
    pts = np.array([(x, y, z) for x in range(side) for y in range(side) for z in range(side)])[:N] * 3.8
    return (pts + np.random.RandomState(0).randn(*pts.shape) * 0.5).astype(np.float64)


def sparse_hessian(xyz):
    n = len(xyz); pairs = np.array(list(cKDTree(xyz).query_pairs(CUT)))
    rows, cols, vals = [], [], []
    for i, j in pairs:
        u = (xyz[j] - xyz[i]); u /= np.linalg.norm(u); b = np.outer(u, u)
        for (a, c, s) in [(i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)]:
            for x in range(3):
                for y in range(3):
                    rows.append(3*a+x); cols.append(3*c+y); vals.append(s*b[x, y])
    return sp.csc_matrix((vals, (rows, cols)), shape=(3*n, 3*n)), len(pairs)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Denoiser(nn.Module):
    def __init__(self, L, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(2*L+16, h), nn.GELU(),
                          nn.Linear(h, h), nn.GELU(), nn.Linear(h, L))
    def forward(self, zt, cond, t):
        f = torch.exp(torch.arange(8) * (-math.log(1e4) / 8)); a = t[:, None].float() * f[None]
        return self.net(torch.cat([zt, cond, torch.sin(a), torch.cos(a)], -1))


m = Denoiser(L); m.eval()
print(f"[scaling] leading {L} ANM modes via Lanczos (sparse), CUT={CUT}. CPU timing.")
print(f"  {'N':>7}{'contacts':>9}{'eigsh_s':>9}{'dense_s':>9}{'match':>8}{'enc_ms':>8}{'dec_ms':>8}{'1step_ms':>9}{'peakMB':>8}")
data = []
for N in NS:
    xyz = make_structure(N); H, nc = sparse_hessian(xyz)
    t0 = time.perf_counter()
    w, V = eigsh(H, k=L + 8, sigma=1e-6, which="LM"); idx = np.argsort(w); modes = V[:, idx[6:6 + L]]
    t_eig = time.perf_counter() - t0
    t_dense = match = np.nan
    if N <= 1500:
        t0 = time.perf_counter(); wd, Vd = np.linalg.eigh(H.toarray()); t_dense = time.perf_counter() - t0
        match = float(np.max(np.abs(np.sort(w)[6:6+L] - wd[6:6+L]) / (np.abs(wd[6:6+L]) + 1e-9)))
    disp = np.random.RandomState(1).randn(100, 3 * N)
    t0 = time.perf_counter(); Z = disp @ modes; t_enc = (time.perf_counter() - t0) / 100 * 1000
    t0 = time.perf_counter(); _ = Z @ modes.T; t_dec = (time.perf_counter() - t0) / 100 * 1000
    zt = torch.zeros(1, L); cond = torch.zeros(1, L)
    with torch.no_grad():
        t0 = time.perf_counter()
        for k in range(Tdiff): m(zt, cond, torch.tensor([k]))
        t_step = (time.perf_counter() - t0) * 1000
    data.append((N, t_eig, t_enc, t_dec, t_step))
    print(f"  {N:>7}{nc:>9}{t_eig:>9.2f}{t_dense:>9.2f}{match:>8.1e}{t_enc:>8.2f}{t_dec:>8.2f}{t_step:>9.1f}{rss_mb():>8.0f}")

# extrapolate eigsh: fit t ~ a * N^p (log-log), then project to 1e6
Ns = np.array([d[0] for d in data]); te = np.array([d[1] for d in data])
p, loga = np.polyfit(np.log(Ns), np.log(te), 1)
t_eig_1e6 = np.exp(loga) * 1e6 ** p
# per candidate: eigsh (once/molecule) + rollout (1000 diffusion sub-steps once decoded) + decode(1000 frames)
tdec_per_frame_1e6 = data[-1][3] * (1e6 / data[-1][0])           # decode scales O(N*L)
t_roll = data[-1][4] * 1000 / 1000                               # 1000 rollout steps, step ~ O(1) in N
per_cand = t_eig_1e6 + t_roll + tdec_per_frame_1e6 * 1000 / 1000
print(f"\n  eigsh scaling: t ~ N^{p:.2f}; projected to 1e6 atoms = {t_eig_1e6:.0f} s")
print(f"  1e6-atom per-candidate estimate (eigsh once + 1000-step rollout + decode): ~{per_cand:.0f} s CPU")
print(f"  -> ~{3600/max(per_cand,1e-6):.1f} candidates / CPU-hour (GPU would cut the NN + decode terms;")
print(f"     eigsh dominates and is sparse-solver bound). Assumes constant contact density and")
print(f"     that the DDPM transition operator transfers across sizes.")
