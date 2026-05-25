"""
text.py — text normalisation and grapheme-to-phoneme conversion.

These helpers are deliberately shared by *every* ASR model so that WER/PER are
computed identically across models — otherwise model comparisons are unfair.
(Mirrors the ``clean_text`` / ``text_to_phonemes`` logic from the MIR project.)
"""

from __future__ import annotations

import re

# ── Tokens that mark silence / breaths in singing annotations ────────────────
# Filtered out before scoring so they never count as words.
SKIP_LABELS: set[str] = {
    "", "sp", "sil", "SIL", "<eps>", "spn", "<SP>", "SP", "<AP>", "AP",
}

# Lazily created G2p instance — importing g2p_en is slow, so defer it.
_g2p = None


def clean_text(text: str) -> str:
    """Normalise a transcript line for WER/PER scoring.

    Steps: strip angle-bracket tokens (``<AP>``, ``<SP>``...), lowercase, keep
    only ``a-z`` and spaces, collapse whitespace.

    Args:
        text: Raw transcript or hypothesis string (any type tolerated).

    Returns:
        The normalised lowercase string, or "" if the input was not a string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)      # remove <AP>, <SP>, XML-style tokens
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)      # keep letters + spaces only
    return re.sub(r"\s+", " ", text).strip()


def _get_g2p():
    """Return a cached ``g2p_en.G2p`` instance, importing it on first use.

    Returns:
        The shared G2p callable used by ``text_to_phonemes``.
    """
    global _g2p
    if _g2p is None:
        from g2p_en import G2p          # imported lazily — heavy NLTK dependency
        _g2p = G2p()
    return _g2p


def text_to_phonemes(text: str) -> str:
    """Convert text to a space-separated ARPAbet phoneme string.

    Args:
        text: Raw text; it is ``clean_text``-normalised internally first.

    Returns:
        Space-separated ARPAbet phonemes, or "" if the input is empty.
    """
    text = clean_text(text)
    if not text:
        return ""
    phones = _get_g2p()(text)
    return " ".join(p for p in phones if p.strip() and p != " ")
