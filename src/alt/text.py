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


# Per-language character classes kept during normalisation. The English/Latin
# default now keeps common accented letters too (é, ü, ñ...) — the old
# ``[^a-z ]`` filter silently deleted them, corrupting WER for French, German,
# Spanish and Italian even though they're Latin-script.
_KEEP_CHARS: dict[str, str] = {
    "chinese":  r"^一-鿿 ",
    "japanese": r"^぀-ヿ一-鿿 ",
    "korean":   r"^가-힣 ",
    "russian":  r"^а-яё ",
}
_DEFAULT_KEEP_CHARS = r"^a-zàâäåæçèéêëìíîïñòóôöùúûüÿœ "  # Latin-script default

# Languages whose GTSinger annotations are character/syllable-level: the JSON
# "word" field holds one hanzi / kana / hangul syllable, so the space-joined
# reference has no real word boundaries. For these, WER is meaningless (every
# character counts as a "word" in the reference but not in the hypothesis) and
# CER — computed with whitespace removed — is the headline metric instead.
CHAR_LEVEL_LANGUAGES: set[str] = {"chinese", "japanese", "korean"}

# Whisper ISO-639-1 codes → the lowercase language names used by the dataset
# adapter and ``clean_text``. Lets scripts that take a Whisper code (e.g.
# ``finetune_whisper.py --language zh``) reuse the language-aware helpers.
ISO_TO_LANGUAGE: dict[str, str] = {
    "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
    "ru": "russian", "es": "spanish", "fr": "french", "de": "german",
    "it": "italian",
}


def primary_metric(language: str = "english") -> str:
    """Return the headline error metric for a dataset language.

    Args:
        language: Lowercase language name (e.g. "english", "chinese").

    Returns:
        ``"cer"`` for character-level languages (Chinese/Japanese/Korean),
        ``"wer"`` otherwise.
    """
    return "cer" if language.lower() in CHAR_LEVEL_LANGUAGES else "wer"


def join_words(words: list[str], language: str = "english") -> str:
    """Join annotation words into a transcript string for ``language``.

    Character-level languages are joined without separators (natural script
    form, which is also what Whisper emits); all others are space-joined.

    Args:
        words:    Annotation words, already stripped of silence/breath tokens.
        language: Lowercase language name.

    Returns:
        The joined transcript.
    """
    sep = "" if language.lower() in CHAR_LEVEL_LANGUAGES else " "
    return sep.join(words)


# Lazily created OpenCC Traditional→Simplified converter (Chinese only).
_t2s = None


def _to_simplified(text: str) -> str:
    """Convert Traditional Chinese characters to Simplified with OpenCC.

    Args:
        text: Any string; non-Chinese characters pass through unchanged.

    Returns:
        The Simplified-Chinese form of ``text``.
    """
    global _t2s
    if _t2s is None:
        import opencc                    # imported lazily — Chinese-only need
        _t2s = opencc.OpenCC("t2s")
    return _t2s.convert(text)


def clean_text(text: str, language: str = "english") -> str:
    """Normalise a transcript line for WER/PER scoring.

    Steps: strip angle-bracket tokens (``<AP>``, ``<SP>``...), convert
    Traditional to Simplified characters (Chinese only), lowercase, keep only
    the script-appropriate characters for ``language`` plus spaces, collapse
    whitespace.

    Args:
        text: Raw transcript or hypothesis string (any type tolerated).
        language: Dataset language (lowercase name, e.g. "english", "chinese").
            Selects which character class survives normalisation; unrecognised
            languages fall back to the accented-Latin default.

    Returns:
        The normalised lowercase string, or "" if the input was not a string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)      # remove <AP>, <SP>, XML-style tokens
    if language.lower() == "chinese":
        # GTSinger Chinese lyrics are Simplified, but Whisper often decodes
        # Mandarin in Traditional characters (也許 vs 也许). Map everything to
        # Simplified so script choice is not counted as recognition error.
        text = _to_simplified(text)
    text = text.lower()
    keep = _KEEP_CHARS.get(language.lower(), _DEFAULT_KEEP_CHARS)
    text = re.sub(f"[{keep}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_g2p():
    """Return a cached ``g2p_en.G2p`` instance, importing it on first use.

    Returns:
        The shared G2p callable used by ``text_to_phonemes``.
    """
    global _g2p
    if _g2p is None:
        import nltk
        # NLTK ≥ 3.9 renamed the tagger to averaged_perceptron_tagger_eng;
        # older versions use averaged_perceptron_tagger. Download both silently.
        for resource in ("averaged_perceptron_tagger_eng",
                         "averaged_perceptron_tagger", "cmudict"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
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
