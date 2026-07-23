"""Chemical constants and vocabularies for the molecular autoencoder.

Everything here is small, hard-coded, and reproducible so that the data
pipeline has no hidden external dependency (no monomer library, no network
lookups). Topology is *perceived* from coordinates using covalent radii
(see ``parsing.perceive_bonds``) rather than read from a template library;
this keeps the prototype self-contained. See README for the rationale and
limitations.
"""

from __future__ import annotations

# --- Elements -------------------------------------------------------------
# Fixed element vocabulary. Index 0 is the padding element, index 1 is UNK.
ELEMENTS = [
    "PAD", "UNK", "C", "N", "O", "S", "P", "SE", "H",
    "NA", "MG", "K", "CA_ION", "ZN", "FE", "CL",
]
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
PAD_ELEMENT_IDX = 0
UNK_ELEMENT_IDX = 1

# Covalent radii in angstrom (Cordero et al. 2008), used for distance-based
# bond perception. Keys are element symbols as they appear in mmCIF.
COVALENT_RADII = {
    "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "SE": 1.20, "H": 0.31, "F": 0.57, "CL": 1.02, "BR": 1.20,
}
DEFAULT_COVALENT_RADIUS = 0.77
# A bond is perceived when d(i, j) < r_cov(i) + r_cov(j) + tolerance.
BOND_TOLERANCE = 0.45  # angstrom

# --- Residues -------------------------------------------------------------
# 20 standard amino acids (3-letter). Index 0 pad, index 1 UNK/non-standard.
STANDARD_AA = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
RESIDUES = ["PAD", "UNK"] + STANDARD_AA
RESIDUE_TO_IDX = {r: i for i, r in enumerate(RESIDUES)}
PAD_RESIDUE_IDX = 0
UNK_RESIDUE_IDX = 1
STANDARD_AA_SET = set(STANDARD_AA)

# Three-letter -> one-letter, used for FASTA export / similarity splitting.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# --- Atom names -----------------------------------------------------------
# Backbone heavy atoms (order matters for backbone RMSD & chirality).
BACKBONE_ATOMS = ["N", "CA", "C", "O"]

# Union of standard heavy-atom names across the 20 amino acids. Used to build
# a fixed atom-name vocabulary so the decoder can be conditioned on atom
# identity. Hydrogens are excluded (usually absent in X-ray structures).
SIDECHAIN_ATOMS = [
    "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2", "CE", "CE1", "CE2", "CE3",
    "CZ", "CZ2", "CZ3", "CH2", "ND1", "ND2", "NE", "NE1", "NE2", "NH1",
    "NH2", "NZ", "OG", "OG1", "OD1", "OD2", "OE1", "OE2", "OH", "OXT",
    "SD", "SG",
]
ATOM_NAMES = ["PAD", "UNK"] + BACKBONE_ATOMS + SIDECHAIN_ATOMS
ATOM_NAME_TO_IDX = {a: i for i, a in enumerate(ATOM_NAMES)}
PAD_ATOM_IDX = 0
UNK_ATOM_IDX = 1

N_ELEMENTS = len(ELEMENTS)
N_RESIDUES = len(RESIDUES)
N_ATOM_NAMES = len(ATOM_NAMES)
