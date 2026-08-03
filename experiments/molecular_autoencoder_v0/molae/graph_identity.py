"""Molecule-general atom addressing: no residues, no atom-name vocabulary.

CONTRACT CHANGE PENDING (ROADMAP §8.5): when the reaction channel lands, identity
must anchor to (element, persistent t=0 index); the graph features here (WL class,
canonical rank) demote from identity KEY to time-varying CONDITIONING, because
bond changes re-index atoms mid-trajectory -- the traceability failure this module
guards against. Not implemented; recorded here as the contract that must change.


What `res_pos` was actually providing
-------------------------------------
Not biology. ADDRESSABILITY. `(res_pos, slot_idx)` is a unique key per atom,
and uniqueness is what lets the decoder emit a distinct query -- and therefore
a distinct position -- for every atom. Drop it naively and two atoms get the
same query and the same predicted coordinates.

That is the hard part of going molecule-general, because chemistry-derived
identity is *inherently non-unique*. The two methyls of isopropyl, the six
carbons of benzene: same element, same charge, same neighbourhood to any
depth. Any purely chemical identity assigns them the same address.

So identity and addressability are separated here:

    identity        Weisfeiler-Lehman refinement over (element, charge, bond
                    types, degree). Chemically meaningful, permutation
                    -equivariant, and deliberately NOT unique -- symmetric
                    atoms share a class, which is correct.

    addressability  a canonical rank, made unique by iterative tie-breaking.
                    Two symmetric atoms get different ranks, so they get
                    different decoder queries and can be placed apart.

Symmetric atoms are genuinely interchangeable, so which one gets which rank is
arbitrary. That is a property of the molecule, not a defect here, and it is
why evaluation has to be symmetry-aware: permuting the input can swap the
positions of two automorphic atoms without the structure changing at all.
"""

from __future__ import annotations

import numpy as np

MAX_WL_CLASSES = 4096       # hashed class vocabulary for the embedding table


def _neighbour_lists(n_atoms: int, bonds: np.ndarray, bond_types: np.ndarray | None):
    nbr = [[] for _ in range(n_atoms)]
    if bonds is None or len(bonds) == 0:
        return nbr
    bt = (np.ones(len(bonds), dtype=np.int64) if bond_types is None
          else np.asarray(bond_types, dtype=np.int64))
    for (a, b), t in zip(np.asarray(bonds, dtype=np.int64), bt):
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            nbr[a].append((int(b), int(t)))
            nbr[b].append((int(a), int(t)))
    return nbr


def wl_classes(elements, charges=None, bonds=None, bond_types=None, n_iter=3):
    """Weisfeiler-Lehman refinement -> one integer class per atom.

    Permutation-equivariant: permuting the atoms permutes the classes with
    them, and the multiset of classes is unchanged. Symmetric atoms converge
    to the SAME class, which is the intended behaviour -- distinguishing them
    is the job of the canonical rank, not of chemistry.
    """
    elements = np.asarray(elements, dtype=np.int64)
    n = len(elements)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    charges = (np.zeros(n, dtype=np.int64) if charges is None
               else np.asarray(charges, dtype=np.int64))
    nbr = _neighbour_lists(n, bonds, bond_types)

    labels = [(int(e), int(c), len(nbr[i])) for i, (e, c) in enumerate(zip(elements, charges))]
    cur = _compact(labels)
    for _ in range(n_iter):
        nxt = []
        for i in range(n):
            # Sorted, so the label does not depend on neighbour ordering.
            ring = tuple(sorted((cur[j], t) for j, t in nbr[i]))
            nxt.append((cur[i], ring))
        new = _compact(nxt)
        if np.array_equal(new, cur):
            break                       # refinement has converged
        cur = new
    return cur % MAX_WL_CLASSES


def _compact(labels):
    """Map arbitrary hashable labels to 0..k-1, ordered by the label itself.

    Ordering by the label (not by first appearance) is what makes the result
    independent of input atom order.
    """
    uniq = sorted(set(labels), key=repr)
    idx = {u: i for i, u in enumerate(uniq)}
    return np.array([idx[l] for l in labels], dtype=np.int64)


def canonical_order(elements, charges=None, bonds=None, bond_types=None, n_iter=3):
    """A canonical permutation of the atoms, plus a unique rank per atom.

    Returns ``(order, rank, classes)`` where ``order[k]`` is the input index of
    the k-th canonical atom and ``rank[i]`` is atom i's position in that order.

    Ties inside a symmetry class are broken one at a time, re-refining after
    each break -- the standard canonical-labelling loop. When atoms are truly
    automorphic any choice yields an isomorphic labelling, so the canonical
    SEQUENCE OF FEATURES is reproducible even though which physical atom lands
    at which rank is not. That distinction is the whole reason evaluation is
    symmetry-aware.
    """
    elements = np.asarray(elements, dtype=np.int64)
    n = len(elements)
    if n == 0:
        return (np.zeros(0, dtype=np.int64),) * 3
    charges = (np.zeros(n, dtype=np.int64) if charges is None
               else np.asarray(charges, dtype=np.int64))
    nbr = _neighbour_lists(n, bonds, bond_types)
    classes = wl_classes(elements, charges, bonds, bond_types, n_iter)

    cur = classes.copy()
    individual = np.zeros(n, dtype=np.int64)     # tie-break tokens
    for _ in range(n):
        key = list(zip(cur.tolist(), individual.tolist()))
        packed = _compact(key)
        sizes = np.bincount(packed)
        tied = np.where(sizes[packed] > 1)[0]
        if len(tied) == 0:
            break
        # Smallest ambiguous class first; within it, the lowest input index.
        smallest = min(np.unique(packed[tied]).tolist(),
                       key=lambda c: (int(sizes[c]), int(c)))
        pick = int(np.where(packed == smallest)[0][0])
        individual[pick] = 1 + int(individual.max())
        # Re-refine so the break propagates through the graph.
        for _ in range(n_iter):
            cur = _compact([(int(packed[i]), int(individual[i]),
                             tuple(sorted((int(packed[j]), t) for j, t in nbr[i])))
                            for i in range(n)])
            packed = cur

    final = _compact(list(zip(cur.tolist(), individual.tolist(),
                              np.arange(n).tolist())))
    order = np.argsort(final, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    return order, rank, classes


def graph_atom_features(elements, charges=None, bonds=None, bond_types=None,
                        n_iter=3):
    """Everything the decoder needs to address an atom, from the graph alone.

    No residues, no chains, no atom-name vocabulary -- only what an .sdf
    actually contains.
    """
    elements = np.asarray(elements, dtype=np.int64)
    n = len(elements)
    charges = (np.zeros(n, dtype=np.int64) if charges is None
               else np.asarray(charges, dtype=np.int64))
    nbr = _neighbour_lists(n, bonds, bond_types)
    order, rank, classes = canonical_order(elements, charges, bonds, bond_types, n_iter)

    degree = np.array([len(nbr[i]) for i in range(n)], dtype=np.int64)
    max_bt = np.array([max([t for _, t in nbr[i]], default=0) for i in range(n)],
                      dtype=np.int64)
    # Ordinal WITHIN the symmetry class: the deterministic tiebreak that gives
    # automorphic atoms distinct addresses.
    ordinal = np.zeros(n, dtype=np.int64)
    seen: dict[int, int] = {}
    for i in order.tolist():
        c = int(classes[i])
        ordinal[i] = seen.get(c, 0)
        seen[c] = ordinal[i] + 1

    return {
        "element_idx": elements,
        "formal_charge": charges,
        "wl_class": classes,
        "canonical_rank": rank,
        "class_ordinal": ordinal,
        "degree": degree,
        "max_bond_type": max_bt,
    }
