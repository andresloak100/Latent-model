# Step 2: first end-to-end compress -> diffuse -> decode skeleton (RUNS)

The largest unexamined risk in the project: nothing has ever run compress ->
diffuse -> decode on MD. This is a SKELETON that RUNS, not a good model.

**Design** (per system, `scripts/armf_step2_diffusion.py`):
- compress: Kabsch-align backbone trajectory, PCA-64 of the displacement (the only
  arm-F signal that cleared the gate: backbone L64 ~0.97A / 75% on mdCATH). -> z_t.
- diffuse: small CONDITIONAL DDPM p(z_{t+1}|z_t) (Tdiff=100) on the latent's
  temporal transitions.
- decode: z -> displacement -> backbone coords.
- validation WITHOUT side chains: CA-CA distances, backbone bond geometry,
  backbone clashes, and DRIFT over a 50-step rollout (not RMSD).

**Result (2 mdCATH domains, 320K):**

| metric | 2cndA01 | 2e2dC02 | native |
|---|---|---|---|
| finite / runs end-to-end | yes | yes | -- |
| CA-CA consecutive (A) | 3.67 | 3.66 | ~3.80 |
| adjacent backbone-atom (A) | 1.49 | 1.48 | ~1.3-1.5 |
| CA drift 1->50 steps (A) | 1.17->1.54 | 1.02->0.99 | ~1-3, no blow-up |
| min CA-CA / clash (A) | 2.63 | 1.37 | >~3.7 |

**Load-bearing risk CLEARED:** the pipeline runs end-to-end, produces finite
structures, and the rollout is STABLE (drift ~1-1.5A over 50 steps, no
divergence). Geometry is roughly plausible (CA-CA ~3.66, bonds ~1.48) but has
CLASHES (min CA-CA 1.4-2.6A << 3.7) -- a quality issue expected of a skeleton, not
a viability blocker. The mechanism works; making it good is the next arc.

**Next (quality, not viability):** reduce clashes (a geometry/energy term or a
better codec), more systems + longer horizon, and the PCA-initialised-AE result
(whether a nonlinear codec beats PCA-64) feeds back into the compress stage.
