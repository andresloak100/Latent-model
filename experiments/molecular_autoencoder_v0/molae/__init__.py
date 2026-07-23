"""molae -- minimal molecular structure autoencoder (research prototype).

Modules:
  constants   chemical vocabularies, covalent radii
  parsing     mmCIF/PDB -> cleaned atom arrays (+ filtering record)
  dataset     torch Dataset + collate for variable-size proteins
  alignment   differentiable Kabsch superposition
  losses      reconstruction losses (coord/distance/bond/clash/chirality)
  metrics     rotation/translation-invariant evaluation metrics
  model       Perceiver-style encoder / bottleneck / decoder
  baselines   PCA / mean-shape / identity comparison
  pdb_io      write reconstructed structures to PDB
  config      YAML experiment configuration
  utils       seeding, environment capture, checkpoints
"""

__version__ = "0.0.1"
