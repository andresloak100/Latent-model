#!/usr/bin/env python3
"""Extract the one-letter sequence from an AFDB mmCIF. Handles BOTH encodings.

WHY THIS IS ITS OWN FILE. My first extractor matched only the multi-line `;`-delimited form, and
mmCIF writes SHORT sequences INLINE on the same line as the tag. The result was a parse failure on
every structure at or below ~80 residues -- 118 of 2,000 -- with KS D=1.0000 against the parsed set:
the two length distributions were completely DISJOINT, cutting at 80/81.

That is a perfect Family A. The exclusion was a deterministic function of length, length is the
regressor, and it removed exactly the short end -- the <=110-residue slice 68d identified as the 57x
scale-up of the regime that already works. At 1M it would have dropped ~59,000 structures, all short,
while the pipeline reported success."""
import re

_BLOCK = re.compile(r'_entity_poly\.pdbx_seq_one_letter_code_can\s*\n?;([^;]+);', re.S)
_BLOCK2 = re.compile(r'_entity_poly\.pdbx_seq_one_letter_code\s*\n?;([^;]+);', re.S)
_INLINE = re.compile(r'_entity_poly\.pdbx_seq_one_letter_code_can\s+([A-Za-z()]+)\s*\n')
_INLINE2 = re.compile(r'_entity_poly\.pdbx_seq_one_letter_code\s+([A-Za-z()]+)\s*\n')


def seq_from_cif(text):
    """Return the one-letter sequence, or None. Tries the _can form first (canonical, one letter per
    residue), then the raw form; block encoding before inline for each."""
    for rx in (_BLOCK, _INLINE, _BLOCK2, _INLINE2):
        m = rx.search(text)
        if m:
            s = re.sub(r'[^A-Za-z]', '', m.group(1)).upper()
            if s:
                return s
    return None
