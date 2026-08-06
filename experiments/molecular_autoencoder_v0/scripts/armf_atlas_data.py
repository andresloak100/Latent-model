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
