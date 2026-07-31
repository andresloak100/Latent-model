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
#
# APPEND-ONLY. Every index below is baked into saved checkpoints' embedding
# rows, so new symbols go at the END and existing ones never move. Growing the
# list changes N_ELEMENTS and therefore the embedding shape; utils.load_checkpoint
# grows old checkpoints' tables to match (see _grow_embeddings there).
ELEMENTS = [
    "PAD", "UNK", "C", "N", "O", "S", "P", "SE", "H",
    "NA", "MG", "K", "CA_ION", "ZN", "FE", "CL",
    # --- added for ligands/cofactors (halogens + boron/silicon are common in
    # drug-like molecules and absent from the protein-only vocabulary) ---
    "F", "BR", "I", "B", "SI",
]
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENTS)}
PAD_ELEMENT_IDX = 0
UNK_ELEMENT_IDX = 1

# Covalent radii in angstrom (Cordero et al. 2008), used for distance-based
# bond perception. Keys are element symbols as they appear in mmCIF.
COVALENT_RADII = {
    "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07,
    "SE": 1.20, "H": 0.31, "F": 0.57, "CL": 1.02, "BR": 1.20,
    "I": 1.39, "B": 0.84, "SI": 1.11,
    # Metal ions: coordination bonds are NOT covalent bonds and perceiving them
    # would create spurious topology, so these are given a small radius that
    # keeps them unbonded at physiological coordination distances (2.0-2.4 A).
    "NA": 0.30, "MG": 0.30, "K": 0.30, "CA": 0.30, "ZN": 0.30, "FE": 0.30,
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
# "LIG" marks an atom belonging to a non-polymer group (ligand, cofactor, ion).
# APPEND-ONLY for the same checkpoint reason as ELEMENTS. Distinct from UNK,
# which means "a residue we could not identify" -- LIG is a positive statement
# that this atom is not part of the protein chain.
RESIDUES = ["PAD", "UNK"] + STANDARD_AA + ["LIG"]
RESIDUE_TO_IDX = {r: i for i, r in enumerate(RESIDUES)}
PAD_RESIDUE_IDX = 0
UNK_RESIDUE_IDX = 1
STANDARD_AA_SET = set(STANDARD_AA)
LIGAND_RESIDUE_IDX = RESIDUE_TO_IDX["LIG"]

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

# --- Ligand filtering ------------------------------------------------------
# Crystallisation additives, cryoprotectants and buffer components. These are
# artefacts of how the crystal was grown, not biology, and keeping them would
# teach the model to reconstruct laboratory reagents. Excluded by default;
# pass ``exclude_ligands=()`` to parse_structure to keep everything.
CRYSTALLIZATION_ADDITIVES = frozenset({
    "HOH", "DOD", "GOL", "EDO", "PEG", "PGE", "PG4", "1PE", "2PE", "P6G",
    "SO4", "PO4", "NO3", "ACT", "ACY", "FMT", "MES", "TRS", "EPE", "IMD",
    "DMS", "MPD", "BME", "IPA", "MOH", "CIT", "TLA", "SCN", "AZI", "BCT",
})

N_ELEMENTS = len(ELEMENTS)
N_RESIDUES = len(RESIDUES)
N_ATOM_NAMES = len(ATOM_NAMES)

# --- Decoder output slots --------------------------------------------------
# The direct decoder emits N_SLOTS coordinate slots per GROUP and each atom
# gathers slot (group_idx * N_SLOTS + slot_idx).
#
# For a protein residue a group is the residue and slot_idx IS atom_name_idx,
# so the addressing is exactly the historical `res_pos * N_ATOM_NAMES +
# atom_name_idx` -- unchanged, bit for bit.
#
# A ligand has no residue and no fixed atom-name vocabulary (its atoms are
# C1/C2/N3/...), so its atoms are addressed by ORDINAL within the group. That
# is why N_SLOTS must stay >= N_ATOM_NAMES, and why a ligand with more than
# N_SLOTS atoms is split across consecutive groups rather than overflowing into
# the next group's slot bank.
N_SLOTS = N_ATOM_NAMES
