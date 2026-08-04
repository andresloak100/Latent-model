"""Fetch a BOUNDED, non-redundant-ish set of X-ray PDB structures for B-factor
pretraining: per-residue CA coords + crystallographic B-factors. Report-scale, ~2500
structures (scale only if the prototype beats ANM). CC0 data. Saves per-structure
npz (ca_xyz, bfac, resid) to $WR/bfactor_cache/."""
import os, json, subprocess, numpy as np
WR = "/network/scratch/j/jacob-junqi.tian/latent-model-workspace"
OUT = f"{WR}/bfactor_cache"; os.makedirs(OUT, exist_ok=True)
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
TARGET = 2500


def get_ids(n_pool=10000):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less", "value": 2.0}}]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": n_pool}, "results_content_type": ["experimental"]}}
    out = subprocess.run(["curl", "-s", "https://search.rcsb.org/rcsbsearch/v2/query",
                          "-H", "Content-Type: application/json", "-d", json.dumps(q)],
                         capture_output=True, text=True).stdout
    ids = [r["identifier"] for r in json.loads(out)["result_set"]]
    stride = max(1, len(ids) // TARGET)
    return ids[::stride][:TARGET]                                 # spanning sample across the alphabet


def parse_pdb(txt):                                              # first protein chain: CA coords + B-factor
    ca_xyz, bfac, resid, chain0 = [], [], [], None
    for ln in txt.splitlines():
        if not ln.startswith("ATOM"): continue
        if ln[12:16].strip() != "CA": continue
        alt = ln[16]
        if alt not in (" ", "A"): continue
        ch = ln[21]
        if chain0 is None: chain0 = ch
        if ch != chain0: break                                   # stop at second chain
        try:
            x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54]); b = float(ln[60:66])
        except ValueError:
            continue
        ca_xyz.append((x, y, z)); bfac.append(b); resid.append(int(ln[22:26]))
    return np.array(ca_xyz), np.array(bfac), np.array(resid)


ids = get_ids(); print(f"[bfactor-fetch] {len(ids)} candidate IDs; downloading .pdb, parsing CA+B")
ok = 0
for i, pid in enumerate(ids):
    fp = f"{OUT}/{pid}.npz"
    if os.path.exists(fp): ok += 1; continue
    r = subprocess.run(["curl", "-sf", f"https://files.rcsb.org/download/{pid}.pdb"], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout: continue
    xyz, b, resid = parse_pdb(r.stdout)
    if len(xyz) < 40 or len(xyz) > 500: continue                 # bounded protein length
    if b.std() < 1e-6: continue                                  # degenerate B-factors
    np.savez(fp, ca_xyz=xyz.astype(np.float32), bfac=b.astype(np.float32), resid=resid)
    ok += 1
    if ok % 200 == 0: print(f"  {ok} saved (of {i+1} tried)")
print(f"=== bfactor-fetch DONE: {ok} structures cached -> {OUT} ===")
