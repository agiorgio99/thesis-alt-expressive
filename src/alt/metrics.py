"""
metrics.py — evaluation metrics for ASR, alignment and pitch.

Provides:
* ASR        — WER, PER, substitution/deletion/insertion counts, hallucination
               rate, and stratified (per-technique / per-singer) breakdowns.
* Alignment  — Time Boundary Error (TBE) between predicted and ground-truth
               intervals, plus tolerance-band accuracy.
* Pitch      — F0 Frame Error (FFE) between two F0 contours.

All scoring uses the shared text normalisation in ``text.py`` so numbers are
comparable across every ASR model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .text import CHAR_LEVEL_LANGUAGES, clean_text, text_to_phonemes


# ─────────────────────────────────────────────────────────────────────────────
#  ASR metrics
# ─────────────────────────────────────────────────────────────────────────────
def score_pair(reference: str, hypothesis: str, language: str = "english") -> dict[str, Any]:
    """Score one (reference, hypothesis) transcript pair.

    Args:
        reference:  Ground-truth lyrics.
        hypothesis: ASR output.
        language:   Dataset language. PER (phoneme error rate) is only computed
            for English — ``text_to_phonemes`` uses an English-only g2p
            backend (``g2p_en``), so PER is left ``None`` for every other
            language rather than silently scoring garbage phonemes.

    Returns:
        A dict with ``wer``, ``cer``, ``per``, ``sub``/``del``/``ins`` counts
        and ``n_ref`` (reference word count). ``wer``/``cer``/``per`` are
        ``None`` when the reference is empty; ``per`` is also ``None`` for
        non-English languages. For character-level languages (see
        ``CHAR_LEVEL_LANGUAGES``) ``wer`` is still reported but is not
        meaningful — use ``cer`` as the headline metric there.
    """
    from jiwer import cer as jiwer_cer
    from jiwer import process_words
    from jiwer import wer as jiwer_wer

    ref = clean_text(reference, language)
    hyp = clean_text(hypothesis, language)

    if not ref:
        return {"wer": None, "cer": None, "per": None,
                "sub": 0, "del": 0, "ins": 0, "n_ref": 0}

    # CER: for character-level languages whitespace carries no information
    # (references are space-joined characters, hypotheses are unspaced), so it
    # is removed on both sides; elsewhere spaces are kept as characters, which
    # is the standard jiwer CER definition.
    if language.lower() in CHAR_LEVEL_LANGUAGES:
        ref_c, hyp_c = ref.replace(" ", ""), hyp.replace(" ", "")
    else:
        ref_c, hyp_c = ref, hyp
    cer = jiwer_cer(ref_c, hyp_c)

    measures = process_words(ref, hyp)
    per = None
    if language.lower() == "english":
        ref_ph = text_to_phonemes(ref)
        hyp_ph = text_to_phonemes(hyp)
        per = jiwer_wer(ref_ph, hyp_ph) if ref_ph else None
    return {
        "wer": jiwer_wer(ref, hyp),
        "cer": cer,
        "per": per,
        "sub": measures.substitutions,
        "del": measures.deletions,
        "ins": measures.insertions,
        "n_ref": len(ref.split()),
    }


def score_asr(df: pd.DataFrame, ref_col: str = "text",
              hyp_col: str = "hypothesis", language: str = "english") -> pd.DataFrame:
    """Add per-utterance WER/PER columns to an ASR results DataFrame.

    Args:
        df:       DataFrame with one row per utterance.
        ref_col:  Column holding ground-truth lyrics.
        hyp_col:  Column holding ASR hypotheses.
        language: Dataset language, forwarded to ``score_pair`` (gates PER to
            English-only and picks the right text-normalisation character set).

    Returns:
        A copy of ``df`` with added columns: ``wer``, ``cer``, ``per``,
        ``wer_sub``, ``wer_del``, ``wer_ins``, ``wer_n_ref``.
    """
    rows = [score_pair(str(r.get(ref_col, "")), str(r.get(hyp_col, "")), language)
            for _, r in df.iterrows()]
    out = df.copy()
    out["wer"] = [r["wer"] for r in rows]
    out["cer"] = [r["cer"] for r in rows]
    out["per"] = [r["per"] for r in rows]
    out["wer_sub"] = [r["sub"] for r in rows]
    out["wer_del"] = [r["del"] for r in rows]
    out["wer_ins"] = [r["ins"] for r in rows]
    out["wer_n_ref"] = [r["n_ref"] for r in rows]
    return out


def aggregate_asr(df: pd.DataFrame,
                  hallucination_threshold: float = 1.0,
                  metric: str = "wer") -> dict[str, Any]:
    """Aggregate scored ASR rows into headline metrics.

    Args:
        df:                      DataFrame already passed through ``score_asr``.
        hallucination_threshold: ``metric`` >= this value counts as a
            hallucination.
        metric:                  Headline error metric (``"wer"`` or
            ``"cer"``, see ``text.primary_metric``). Drives the hallucination
            flag and the ``<metric>_no_halluc`` field, so character-level
            languages are not judged on their meaningless WER.

    Returns:
        A dict with ``n``, ``metric``, ``wer``, ``cer``, ``per``,
        ``hallucination_rate`` and ``<metric>_no_halluc`` (mean headline
        metric excluding hallucinated utterances). ``per`` is ``None`` when no
        utterance has a PER (non-English). Rates are fractions in [0, 1];
        multiply by 100 for percentages.
    """
    def _mean(frame: pd.DataFrame, col: str) -> float | None:
        """Mean of ``col`` ignoring NaN/None; ``None`` if nothing to average."""
        if col not in frame.columns:
            return None
        vals = pd.to_numeric(frame[col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None

    scored = df.dropna(subset=[metric])
    if scored.empty:
        return {"n": 0, "metric": metric, "wer": None, "cer": None, "per": None,
                "hallucination_rate": None, f"{metric}_no_halluc": None}
    halluc = scored[metric] >= hallucination_threshold
    clean = scored[~halluc]
    return {
        "n": int(len(scored)),
        "metric": metric,
        "wer": _mean(scored, "wer"),
        "cer": _mean(scored, "cer"),
        "per": _mean(scored, "per"),
        "hallucination_rate": float(halluc.mean()),
        f"{metric}_no_halluc": _mean(clean, metric),
    }


def stratified_asr(df: pd.DataFrame, by: str,
                   hallucination_threshold: float = 1.0,
                   metric: str = "wer") -> pd.DataFrame:
    """Break ASR metrics down by a metadata column (technique, singer, ...).

    Args:
        df:                      DataFrame already passed through ``score_asr``.
        by:                      Column to group by (e.g. "technique").
        hallucination_threshold: ``metric`` >= this value counts as a
            hallucination.
        metric:                  Headline error metric, forwarded to
            ``aggregate_asr`` and used for sorting.

    Returns:
        One row per group with the same fields as ``aggregate_asr`` plus the
        grouping value, sorted by descending headline metric.
    """
    if by not in df.columns:
        return pd.DataFrame()
    rows = []
    for value, group in df.groupby(by):
        agg = aggregate_asr(group, hallucination_threshold, metric)
        agg[by] = value
        rows.append(agg)
    result = pd.DataFrame(rows)
    return result.sort_values(metric, ascending=False, na_position="last")


# ─────────────────────────────────────────────────────────────────────────────
#  Alignment metrics — Time Boundary Error
# ─────────────────────────────────────────────────────────────────────────────
def time_boundary_error(pred: list, gt: list) -> dict[str, Any]:
    """Compare predicted vs ground-truth intervals by positional matching.

    Predicted interval ``i`` is matched to ground-truth interval ``i`` (the
    same positional scheme used in the MIR project). The boundary error of a
    pair is the mean of its absolute start-error and end-error.

    Args:
        pred: Predicted ``Interval`` objects (need ``.start`` / ``.end``).
        gt:   Ground-truth ``Interval`` objects.

    Returns:
        A dict with ``n`` (matched pairs), ``mean_tbe`` and ``median_tbe`` in
        seconds, and ``within_20ms`` / ``within_50ms`` / ``within_100ms``
        accuracy fractions. All-zero / empty when no pairs match.
    """
    n = min(len(pred), len(gt))
    if n == 0:
        return {"n": 0, "mean_tbe": None, "median_tbe": None,
                "within_20ms": None, "within_50ms": None, "within_100ms": None}
    errors = np.array([
        (abs(pred[i].start - gt[i].start) + abs(pred[i].end - gt[i].end)) / 2
        for i in range(n)
    ])
    return {
        "n": int(n),
        "mean_tbe": float(errors.mean()),
        "median_tbe": float(np.median(errors)),
        "within_20ms": float(np.mean(errors <= 0.020)),
        "within_50ms": float(np.mean(errors <= 0.050)),
        "within_100ms": float(np.mean(errors <= 0.100)),
    }


def aggregate_tbe(per_utt: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool per-utterance TBE results into corpus-level numbers.

    Args:
        per_utt: A list of dicts as returned by ``time_boundary_error``.

    Returns:
        A dict with the same keys as ``time_boundary_error``, pooled by
        weighting each utterance by its matched-pair count.
    """
    rows = [r for r in per_utt if r.get("n")]
    if not rows:
        return {"n": 0, "mean_tbe": None, "median_tbe": None,
                "within_20ms": None, "within_50ms": None, "within_100ms": None}
    weights = np.array([r["n"] for r in rows], dtype=float)

    def _wmean(key: str) -> float:
        """Return the pair-count-weighted mean of ``key`` across ``rows``."""
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.average(vals, weights=weights))

    return {
        "n": int(weights.sum()),
        "mean_tbe": _wmean("mean_tbe"),
        "median_tbe": _wmean("median_tbe"),
        "within_20ms": _wmean("within_20ms"),
        "within_50ms": _wmean("within_50ms"),
        "within_100ms": _wmean("within_100ms"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Pitch metric — F0 Frame Error
# ─────────────────────────────────────────────────────────────────────────────
def f0_frame_error(pred_f0: np.ndarray, ref_f0: np.ndarray,
                   cents_threshold: float = 50.0) -> float | None:
    """Compute the F0 Frame Error (FFE) between two aligned F0 contours.

    FFE = fraction of frames that are either *voicing errors* (one contour
    voiced, the other not) or *gross pitch errors* (both voiced but differing
    by more than ``cents_threshold`` cents).

    Args:
        pred_f0:         Predicted F0 per frame in Hz; unvoiced frames are NaN
                         or <= 0.
        ref_f0:          Reference F0 per frame in Hz, same convention.
        cents_threshold: Pitch-difference tolerance in cents (default 50).

    Returns:
        The FFE as a fraction in [0, 1], or ``None`` if the contours have no
        overlapping frames.
    """
    n = min(len(pred_f0), len(ref_f0))
    if n == 0:
        return None
    p = np.asarray(pred_f0[:n], dtype=float)
    r = np.asarray(ref_f0[:n], dtype=float)

    p_voiced = np.isfinite(p) & (p > 0)
    r_voiced = np.isfinite(r) & (r > 0)

    voicing_error = p_voiced != r_voiced            # one voiced, other not
    both = p_voiced & r_voiced
    cents = np.zeros(n)
    cents[both] = 1200.0 * np.abs(np.log2(p[both] / r[both]))
    gross_error = both & (cents > cents_threshold)

    return float(np.mean(voicing_error | gross_error))
