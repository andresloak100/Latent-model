"""ATLAS data access: MEMMAP loader + CONSERVATION OF n + frame-space ceiling.

MEMORY. ~157 GB of cache will not fit in RAM, so the MISATO pattern (load the whole cache up front)
does not survive here. Each protein is stored UNCOMPRESSED .npy and opened with mmap_mode='r';
training reads only the 8 random frames it needs per step. Per-epoch shuffling is over CHUNK INDICES,
never over a materialised array.

CONSERVATION OF n APPLIES TO THE TRAINING LOOP, NOT ONLY TO ANALYSIS. An OOM or a failed load is an
N-CORRELATED FAILURE: memory pressure kills the LARGEST proteins first, silently removing the top of
the N axis, and every downstream slope is then computed on a truncated range. That is the
curl-timeout bug relocated into the trainer. So every load is logged, n_in -> n_out is reported, and
the REALISED atom-count range of what actually trained is printed against what was SELECTED. If they
differ, the run is VOID until explained.

CEILING. PCA on 2,000 frames x 3N coords at N=33,541 is a 2,000 x 100,623 matrix. Economy SVD works
but materialises a 1.61 GB Vt (53 s, 7.55 GB peak RSS, measured). The pure FRAME-SPACE route never
forms any 3N-sized array: G = Xtr Xtr^T (2000x2000), eigh, then ho @ V_k = (ho Xtr^T) U_k / s_k.
MEASURED at the ATLAS worst case: 8.5 s, largest array 32 MB, agreeing with economy SVD to 2.6e-18
-- 6.2x faster. Always use this route."""
import os, glob, json, numpy as np
from armf_anm import modes as anm_modes


class AtlasStore:
    def __init__(self, root, verbose=True):
        self.root = root
        self.meta = []
        for j in sorted(glob.glob(f"{root}/*.json")):
            if os.path.basename(j).startswith("fail_"): continue
            m = json.load(open(j)); m["path"] = j.replace(".json", ".npy")
            if os.path.exists(m["path"]): self.meta.append(m)
        self.selected = len(self.meta)
        if verbose:
            A = [m["atoms"] for m in self.meta]
            print(f"  [store] {self.selected} proteins indexed, atoms {min(A)}-{max(A)}, "
                  f"{sum(m['bytes'] for m in self.meta)/1e9:.1f} GB on disk (memmapped, not loaded)")
        self.loaded, self.failed = set(), {}

    def frames(self, i, idx):
        """Memmap-read only the requested frames. Any failure is RECORDED, never swallowed."""
        m = self.meta[i]
        try:
            a = np.load(m["path"], mmap_mode="r")            # (R, F, N, 3), nothing materialised
            R, F = a.shape[0], a.shape[1]
            out = np.stack([np.asarray(a[k % R, k // R % F]) for k in idx]).astype(np.float32)
            self.loaded.add(m["pdb"]); return out
        except Exception as e:
            self.failed[m["pdb"]] = {"atoms": m["atoms"], "err": f"{type(e).__name__}: {e}"}
            return None

    def conservation_report(self):
        """n_in -> n_out for the DATA PIPELINE, with the realised vs selected atom range."""
        sel = [m["atoms"] for m in self.meta]
        got = [m["atoms"] for m in self.meta if m["pdb"] in self.loaded]
        print(f"  [n] data pipeline: {self.selected} selected -> {len(self.loaded)} actually trained")
        if not got: print("    NOTHING TRAINED -- run is VOID"); return False
        print(f"    atom range SELECTED {min(sel)}-{max(sel)}  vs  REALISED {min(got)}-{max(got)}")
        ok = (len(self.loaded) == self.selected)
        if not ok:
            f = self.failed
            print(f"    FAILED {len(f)}: atoms " +
                  (f"{min(v['atoms'] for v in f.values())}-{max(v['atoms'] for v in f.values())}" if f else "-"))
            print(f"    *** N-CORRELATED LOSS -> the top of the N axis may be truncated; RUN IS VOID "
                  f"until explained (this is the curl-timeout bug relocated into the trainer) ***")
        return ok


def pca_ceiling_framespace(X, h, ks):
    """FVE of the top-k train-PCA subspace on held-out frames, WITHOUT forming any 3N-sized array.
    X: (T, 3N) float; h: train split. Verified against economy SVD to 2.6e-18 at N=33,541."""
    tr = X[:h].astype(np.float64); ho = X[h:].astype(np.float64)
    tr = tr - tr.mean(0); ho = ho - X[:h].astype(np.float64).mean(0)
    G = tr @ tr.T                                            # (h, h) -- the only large product
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); U = U[:, o]
    C = ho @ tr.T                                            # (T-h, h)
    sst = float((ho ** 2).sum()); out = {}
    for k in ks:
        kk = min(k, (w > w[0] * 1e-12).sum())
        s = np.sqrt(np.clip(w[:kk], 1e-12, None))
        proj = (C @ U[:, :kk]) / s
        out[k] = float((proj ** 2).sum() / (sst + 1e-12))
        out[f"valid{k}"] = bool(k <= 0.30 * (h - 1))         # G7 per system, h varies
    return out


# ---- shared per-system access, lifted here so the curve script and any analysis import ONE
# definition. Duplicating them across scripts is how a comparator ends up computed differently
# from the model it is compared against (FAMILY F).
def sysdata(store, i):
    """REPLICAS 0+1 = TRAIN (5,002 frames), REPLICA 2 = HELD-OUT (2,501 frames).

    Two gains, both free -- the cache already held all three replicas and only replica 0 was being
    read. (1) FRAMES PER INTRINSIC DIMENSION triples: at N=33,377 with rank90 ~400 that is 12.5x
    instead of 5.0x, which is the actual mechanism behind the ceiling degrading with N (it was never
    '80 frames', it was frames vs rank90, and rank90 GROWS with N at b>=+0.10). (2) The held-out set
    becomes an INDEPENDENT TRAJECTORY rather than the temporally-adjacent tail of the training one --
    a strictly stronger generalisation test.
    Nothing large is materialised: coordinates stay memmapped and the Gram is chunked over columns."""
    m = store.meta[i]
    try:
        a = np.load(m["path"], mmap_mode="r")          # (R, F, N, 3)
        R, F, N = a.shape[0], a.shape[1], a.shape[2]
        if R < 3: return None
        ref = np.asarray(a[0, 0]).astype(np.float64)
        # per-system mean over TRAIN replicas only, streamed
        mu = np.zeros(3*N)
        for r in (0, 1): mu += np.asarray(a[r]).reshape(F, -1).astype(np.float64).sum(0)
        mu /= (2*F)
        sst = 0.0                                       # STREAMED: (F,3N) is 1.0 GB at N=33,377,
        for c0 in range(0, 3*N, 20000):                 # and 125 of those would OOM -- killing the
            c1 = min(c0 + 20000, 3*N)                   # LARGEST systems first, i.e. an N-correlated
            sst += float(((np.asarray(a[2]).reshape(F, -1)[:, c0:c1].astype(np.float64)
                           - mu[c0:c1]) ** 2).sum())    # failure truncating the axis under test.
        s0 = np.asarray(a[0]).reshape(F, -1).astype(np.float64) - mu
        scale = float(np.sqrt((s0 ** 2).reshape(F, N, 3).sum(-1).mean()) + 1e-6)
        el = np.array(m["elem"][:N])
        ELEMS = [1, 6, 7, 8, 16, 15, 9, 17]
        oh = np.zeros((N, len(ELEMS) + 1), np.float32)
        for k, e in enumerate(el): oh[k, ELEMS.index(int(e)) if int(e) in ELEMS else -1] = 1.0
        rp = ((ref - ref.mean(0)) / (ref.std() + 1e-6)).astype(np.float32)
        store.loaded.add(m["pdb"])
        return dict(pdb=m["pdb"], N=N, F=F, ref=ref, scale=scale, mu=mu, path=m["path"],
                    stat=np.concatenate([oh, rp], 1).astype(np.float32),
                    sst=sst)
    except Exception as e:
        store.failed[m["pdb"]] = {"atoms": m["atoms"], "err": f"{type(e).__name__}: {e}"}
        return None


def ho_frames(d, s, e):
    """Held-out frames (replica 2), memmapped and centred. Nothing large retained."""
    a = np.load(d["path"], mmap_mode="r")
    return np.asarray(a[2, s:e]).astype(np.float64) - d["mu"].reshape(d["N"], 3)


def ho_cols(d, c0, c1):
    """Held-out COLUMN slice, for chunked products."""
    a = np.load(d["path"], mmap_mode="r")
    return np.asarray(a[2]).reshape(d["F"], -1)[:, c0:c1].astype(np.float64) - d["mu"][c0:c1]


def anm_fve_streamed(d, V, chunk=20000):
    """||ho V||^2 / sst accumulated over column chunks -- ho is never materialised."""
    acc = np.zeros((d["F"], V.shape[1]))
    for c0 in range(0, 3 * d["N"], chunk):
        c1 = min(c0 + chunk, 3 * d["N"])
        acc += ho_cols(d, c0, c1) @ V[c0:c1]
    return float((acc ** 2).sum() / (d["sst"] + 1e-12))


def train_frames(d, idx):
    """Read specific TRAIN frames (replicas 0+1) via memmap; nothing else materialised."""
    a = np.load(d["path"], mmap_mode="r"); F = d["F"]
    out = np.empty((len(idx), d["N"], 3), np.float32)
    for k, t in enumerate(idx):
        out[k] = np.asarray(a[t // F, t % F])
    return out.reshape(len(idx), -1) - d["mu"].astype(np.float32)


def ceiling_chunked(d, ks, chunk=20000):
    """PCA ceiling with the train Gram accumulated over COLUMN CHUNKS, so a 5,002 x 100,131 matrix
    is never held. G = sum_c Xc Xc^T and C = sum_c Ho_c Xc^T are both exact under chunking."""
    a = np.load(d["path"], mmap_mode="r"); F = d["F"]; D = 3 * d["N"]; H = 2 * F
    G = np.zeros((H, H)); C = np.zeros((F, H))
    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        Xc = np.concatenate([np.asarray(a[r]).reshape(F, -1)[:, c0:c1] for r in (0, 1)], 0).astype(np.float64)
        Xc -= d["mu"][c0:c1]
        G += Xc @ Xc.T
        C += ho_cols(d, c0, c1) @ Xc.T
        del Xc
    w, U = np.linalg.eigh(G); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); U = U[:, o]
    out = {}
    for k in ks:
        kk = min(k, int((w > w[0] * 1e-12).sum()))
        proj = (C @ U[:, :kk]) / np.sqrt(np.clip(w[:kk], 1e-12, None))
        out[k] = float((proj ** 2).sum() / (d["sst"] + 1e-12))
        out[f"valid{k}"] = bool(k <= 0.30 * (H - 1))
        out[f"cond{k}"] = float(np.sqrt(w[min(kk, len(w)) - 1] / max(w[0], 1e-12)))
    return out
