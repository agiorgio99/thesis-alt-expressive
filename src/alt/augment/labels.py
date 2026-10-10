"""
labels.py — update GTSinger word annotations after an augmentation.
"""

from __future__ import annotations

import copy

from .io import TECH_CODE


def update_labels(meta: list, affected_word_indices: list, technique_key: str) -> list:
    """Deep-copy ``meta`` and flag the augmented technique on affected words.

    For every affected word, the per-phone flag list ``w[technique_key]`` is set
    to all "1" and the technique's code is added to the comma-separated ``tech``
    field (dropping the "0" = no-technique marker).

    Args:
        meta:                  Original word-level annotation.
        affected_word_indices: Indices of the words the transform modified.
        technique_key:         Technique flag name ("vibrato", "breathy",
            "glissando", "pharyngeal", "falsetto" or "mix").

    Returns:
        The updated copy of the annotation (``meta`` itself is not modified).
    """
    aug = copy.deepcopy(meta)
    for wi in affected_word_indices:
        w = aug[wi]
        n = len(w["ph"])
        w[technique_key] = ["1"] * n
        existing = set(w["tech"].split(",")) if w["tech"] != "0" else set()
        existing.discard("0")
        existing.add(TECH_CODE[technique_key])
        w["tech"] = ",".join(sorted(existing, key=int))
    return aug
