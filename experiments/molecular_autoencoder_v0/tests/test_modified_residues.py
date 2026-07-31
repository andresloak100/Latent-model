"""Non-standard residues must stay IN their polymer chain.

This exists because of a real failure found on real data. Selenomethionine,
hydroxyproline, D-amino acids and other non-canonical residues are deposited
with het_flag "H" *while sitting inside the backbone*. The polymer loop
dropped them for not being standard amino acids, and the ligand pass then
re-added them as free-floating groups with their own synthetic chain_idx --
at which point the chain guard in perceive_bonds correctly, and uselessly,
refused to bond them to the protein they are covalently part of.

Two things broke at once: the residue lost both peptide bonds, and it was
renumbered to a group AFTER every real residue, destroying its position in
the sequence. On a 741-structure complex set this hit 58% of entries, and the
top "ligands" by frequency were HYP/DPR/AIB/DLE/MSE -- modified residues, not
compounds.

Non-canonical residues are also a primary tool for engineering designed
proteins, so this is on the critical path rather than an edge case.
"""

import numpy as np
import pytest

from molae import constants as C
from molae.parsing import parse_structure

from test_ligands import _atom_line          # column-exact PDB writer


def _pdb_with_modified_residue(tmp_path, mod_name="MSE", position=5, n_res=10,
                               mod_atoms=(("N", "N"), ("CA", "C"), ("C", "C"),
                                          ("O", "O"), ("CB", "C"), ("SE", "SE"))):
    """Poly-ALA chain whose residue at `position` is a HETATM modified residue.

    Geometry is a straight 3.8 A backbone so consecutive C->N pairs fall inside
    the covalent cutoff and peptide bonds are perceived.
    """
    lines, serial = [], 1
    for r in range(1, n_res + 1):
        base = np.array([3.8 * r, 0.0, 0.0])
        if r == position:
            for k, (aname, elem) in enumerate(mod_atoms):
                off = [(0., 0., 0.), (1.4, .2, 0.), (2.5, -.4, 0.),
                       (2.6, -1.6, 0.), (1.5, 1.7, 0.), (1.6, 3.2, 0.)][k % 6]
                lines.append(_atom_line("HETATM", serial, f" {aname:<3s}"[:4],
                                        mod_name, "A", r, base + np.array(off), elem))
                serial += 1
            continue
        for aname, elem, off in [("N", "N", (0., 0., 0.)), ("CA", "C", (1.4, .2, 0.)),
                                 ("C", "C", (2.5, -.4, 0.)), ("O", "O", (2.6, -1.6, 0.)),
                                 ("CB", "C", (1.5, 1.7, 0.))]:
            lines.append(_atom_line("ATOM", serial, f" {aname:<3s}"[:4], "ALA",
                                    "A", r, base + np.array(off), elem))
            serial += 1
    lines.append("END")
    p = tmp_path / "mod.pdb"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _cross_bonds(ps, group):
    s = set(np.where(ps.res_pos == group)[0].tolist())
    return sum(1 for i, j in ps.bonds if (int(i) in s) != (int(j) in s))


def test_dropped_by_default(tmp_path):
    """Default behaviour is unchanged: non-standard residues are dropped."""
    ps = parse_structure(_pdb_with_modified_residue(tmp_path), pdb_id="T")
    assert ps.record["n_modified_residues_kept"] == 0
    assert "MSE" in ps.record["dropped_nonstandard_residues"]


def test_kept_in_chain_at_its_sequence_position(tmp_path):
    ps = parse_structure(_pdb_with_modified_residue(tmp_path, position=5),
                         pdb_id="T", keep_modified_residues=True)
    mod = np.where(ps.residue_idx == C.UNK_RESIDUE_IDX)[0]
    assert len(mod) == 6
    assert ps.record["n_modified_residues_kept"] == 1
    # Position in the chain is preserved, NOT pushed after every real residue.
    assert sorted(set(ps.res_pos[mod].tolist())) == [4]        # 0-based
    assert ps.n_residues == 10


def test_keeps_its_peptide_bonds(tmp_path):
    """The bug: a modified residue re-added as a ligand had zero bonds to the
    chain it is covalently part of."""
    ps = parse_structure(_pdb_with_modified_residue(tmp_path, position=5),
                         pdb_id="T", keep_modified_residues=True)
    mod_group = 4
    # Same as any other internal residue: one peptide bond on each side.
    standard = [_cross_bonds(ps, g) for g in range(1, 9) if g != mod_group]
    assert set(standard) == {2}, standard
    assert _cross_bonds(ps, mod_group) == 2


def test_shares_the_polymer_chain_idx(tmp_path):
    """A synthetic chain_idx is what made perceive_bonds refuse the bonds."""
    ps = parse_structure(_pdb_with_modified_residue(tmp_path), pdb_id="T",
                         keep_ligands=True, keep_modified_residues=True)
    assert set(ps.chain_idx.tolist()) == {0}


def test_not_double_counted_as_a_ligand(tmp_path):
    """It carries het_flag H, so the ligand pass must skip polymer residues."""
    ps = parse_structure(_pdb_with_modified_residue(tmp_path), pdb_id="T",
                         keep_ligands=True, keep_modified_residues=True)
    assert not (ps.residue_idx == C.LIGAND_RESIDUE_IDX).any()
    assert ps.record["n_ligand_atoms"] == 0
    assert ps.n_atoms == 9 * 5 + 6          # 9 ALA + one 6-atom modified residue


def test_non_vocabulary_atoms_are_kept_not_truncated(tmp_path):
    """SE is not a protein atom name; dropping it would silently shorten the
    residue. The group falls back to ordinal slots and keeps every atom."""
    ps = parse_structure(_pdb_with_modified_residue(tmp_path), pdb_id="T",
                         keep_modified_residues=True)
    mod = np.where(ps.residue_idx == C.UNK_RESIDUE_IDX)[0]
    assert "SE" in [ps.atom_name[i] for i in mod]
    assert np.array_equal(np.sort(ps.slot_idx[mod]), np.arange(len(mod)))


def test_decoder_addresses_stay_unique(tmp_path):
    ps = parse_structure(_pdb_with_modified_residue(tmp_path), pdb_id="T",
                         keep_ligands=True, keep_modified_residues=True)
    addr = ps.res_pos * C.N_SLOTS + ps.slot_idx
    assert len(set(addr.tolist())) == len(addr)


def test_oversized_modified_residue_is_dropped(tmp_path):
    """A group cannot address more atoms than it has slots."""
    big = tuple((f"X{i}", "C") for i in range(C.N_SLOTS + 3))
    ps = parse_structure(
        _pdb_with_modified_residue(tmp_path, mod_atoms=big), pdb_id="T",
        keep_modified_residues=True)
    assert ps.record["dropped_oversized_residues"] == 1
    assert ps.record["n_modified_residues_kept"] == 1   # counted, then rejected
    addr = ps.res_pos * C.N_SLOTS + ps.slot_idx
    assert len(set(addr.tolist())) == len(addr)


def test_sequence_marks_modified_residues_as_X(tmp_path):
    ps = parse_structure(_pdb_with_modified_residue(tmp_path, position=5),
                         pdb_id="T", keep_modified_residues=True)
    assert ps.sequence == "AAAA" + "X" + "AAAAA"
