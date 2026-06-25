"""
dataset.py — dataset adapters and a registry to choose one by name.

Why an adapter pattern?
The pipeline must work with *different* singing corpora (GTSinger today;
VocalSet, others later). Each corpus has its own folder layout and annotation
format, but the rest of the pipeline only needs a flat list of ``Utterance``
objects. So every corpus gets a small ``DatasetAdapter`` subclass that knows
how to crawl its files and emit ``Utterance`` records.

Add a new dataset in three steps:
    1. Subclass ``DatasetAdapter`` and implement ``list_utterances()``.
    2. Register it with the ``@register_dataset("yourname")`` decorator.
    3. Set ``data.name: yourname`` in the YAML config.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .text import SKIP_LABELS, clean_text


# ─────────────────────────────────────────────────────────────────────────────
#  The common record type every adapter emits
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Utterance:
    """One audio clip plus its metadata — the unit the pipeline operates on.

    Attributes:
        utt_id:          Unique identifier (safe for use as a filename).
        audio_path:      Absolute path to the source audio file.
        text:            Ground-truth lyrics (normalised lowercase), may be "".
        singer_id:       Performer identifier (used for per-singer stratifying).
        technique:       Expressive technique label (vibrato, breathy, ...).
        group:           Coarse subset label ("technique" | "control" | "speech").
        textgrid_path:   Path to a phoneme/word TextGrid, or None.
        json_path:       Path to a JSON annotation (GT timestamps), or None.
        extra:           Any extra dataset-specific metadata.
    """
    utt_id: str
    audio_path: str
    text: str = ""
    singer_id: str = "unknown"
    technique: str = "none"
    group: str = "unknown"
    textgrid_path: str | None = None
    json_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base class
# ─────────────────────────────────────────────────────────────────────────────
class DatasetAdapter(ABC):
    """Base class: turns one corpus on disk into a list of ``Utterance``s.

    Args:
        root:     Root folder of the dataset on disk.
        language: Language subset filter (corpus-specific meaning).
        limit:    If set, keep only the first N utterances (smoke tests).
    """

    def __init__(self, root: str | Path, language: str = "english",
                 limit: int | None = None) -> None:
        self.root = Path(root)
        self.language = language
        self.limit = limit

    @abstractmethod
    def list_utterances(self) -> list[Utterance]:
        """Crawl the dataset and return all utterances.

        Returns:
            A list of ``Utterance`` records. Implementations should apply
            ``self.limit`` before returning.
        """
        raise NotImplementedError

    def _apply_limit(self, items: list[Utterance]) -> list[Utterance]:
        """Truncate a list of utterances to ``self.limit`` if one is set.

        Args:
            items: The full list of utterances.

        Returns:
            The list, truncated to the first ``self.limit`` items if a limit
            was given, otherwise unchanged.
        """
        return items[: self.limit] if self.limit else items


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
DATASET_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_dataset(name: str) -> Callable[[type[DatasetAdapter]], type[DatasetAdapter]]:
    """Class decorator that registers a ``DatasetAdapter`` under a name.

    Args:
        name: Registry key (matches ``data.name`` in the config).

    Returns:
        A decorator that records the class and returns it unchanged.
    """
    def _decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        DATASET_REGISTRY[name] = cls
        return cls
    return _decorator


def get_dataset(name: str, root: str | Path, language: str = "english",
                limit: int | None = None) -> DatasetAdapter:
    """Instantiate the dataset adapter registered under ``name``.

    Args:
        name:     Registry key (e.g. "gtsinger").
        root:     Dataset root folder.
        language: Language subset filter.
        limit:    Optional cap on the number of utterances.

    Returns:
        A ready-to-use ``DatasetAdapter`` instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](root=root, language=language, limit=limit)


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: read ground-truth word lyrics from a TextGrid
# ─────────────────────────────────────────────────────────────────────────────
def lyrics_from_json(json_path: str | Path) -> str:
    """Extract word-level transcript from a GTSinger JSON annotation file.

    Used as a fallback when no TextGrid is available (e.g. augmented files).

    Args:
        json_path: Path to a GTSinger ``.json`` annotation file.

    Returns:
        Space-joined non-silence words, or "" if the file is missing/unreadable.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return ""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        words = [
            w["word"] for w in data
            if isinstance(w, dict) and w.get("word") not in ("<SP>", None, "")
        ]
        return " ".join(words)
    except Exception:
        return ""


def lyrics_from_textgrid(tg_path: str | Path) -> str:
    """Extract the word-tier transcript from a TextGrid file.

    Args:
        tg_path: Path to a ``.TextGrid`` file.

    Returns:
        Space-joined words from the first "word" tier (silence markers removed),
        or "" if the file is missing or unreadable.
    """
    tg_path = Path(tg_path)
    if not tg_path.exists():
        return ""
    try:
        from praatio import textgrid as ptextgrid
        tg = ptextgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        for tier_name in tg.tierNames:
            if "word" in tier_name.lower():
                words = [e.label.strip() for e in tg.getTier(tier_name).entries
                         if e.label.strip() not in SKIP_LABELS]
                return " ".join(words)
    except Exception:
        pass
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  GTSinger adapter
# ─────────────────────────────────────────────────────────────────────────────
@register_dataset("gtsinger")
class GTSingerAdapter(DatasetAdapter):
    """Adapter for the GTSinger corpus (English subset).

    Expected on-disk layout (one WAV per leaf folder)::

        <root>/<singer>/<technique_folder>/<song>/<group_folder>/XXXX.wav
                                                                 XXXX.TextGrid
                                                                 XXXX.json

    The adapter is robust to an extra nesting level inside the root (zip
    archives sometimes add one).
    """

    # GTSinger folder names -> coarse pipeline group label.
    _GROUP_LABEL = {
        "Breathy_Group": "technique", "Glissando_Group": "technique",
        "Vibrato_Group": "technique", "Falsetto_Group": "technique",
        "Mixed_Voice_Group": "technique", "Pharyngeal_Group": "technique",
        "Control_Group": "control", "Paired_Speech_Group": "speech",
    }
    # GTSinger technique-folder names -> normalised technique label.
    _TECHNIQUE_LABEL = {
        "Breathy": "breathy", "Glissando": "glissando", "Vibrato": "vibrato",
        "Mixed_Voice_and_Falsetto": "mixed_falsetto", "Pharyngeal": "pharyngeal",
    }
    _SINGER_HINTS = ("EN-Alto-1", "EN-Alto-2", "EN-Tenor-1")

    def _find_english_root(self, base: Path) -> Path:
        """Locate the real corpus root, tolerating one extra nesting folder.

        Args:
            base: The configured dataset root.

        Returns:
            The folder that directly contains the per-singer directories.
        """
        for p in sorted(base.rglob("*")):
            if p.is_dir() and any(h in p.name for h in self._SINGER_HINTS):
                return p.parent
            if p.is_dir() and p.name.lower() == "english":
                return p
        return base

    def list_utterances(self) -> list[Utterance]:
        """Crawl the GTSinger folder tree and build one record per WAV.

        Returns:
            A list of ``Utterance`` objects (truncated to ``self.limit``).
        """
        english_root = self._find_english_root(self.root)
        records: list[Utterance] = []

        wav_files = [w for w in english_root.rglob("*.wav")
                     if not w.name.startswith(".")]
        for wav in sorted(wav_files):
            group_folder = wav.parent.name
            if group_folder not in self._GROUP_LABEL:
                continue                              # not a recognised subset

            song = wav.parent.parent.name
            technique_folder = wav.parent.parent.parent.name
            singer_id = wav.parent.parent.parent.parent.name

            technique = self._TECHNIQUE_LABEL.get(
                technique_folder, technique_folder.lower())
            group = self._GROUP_LABEL[group_folder]

            tg_path = wav.with_suffix(".TextGrid")
            json_path = wav.with_suffix(".json")
            lyrics = (clean_text(lyrics_from_textgrid(tg_path))
                      or clean_text(lyrics_from_json(json_path)))

            utt_id = (f"{singer_id}__{technique}__{group}"
                      f"__{song.replace(' ', '_')}__{wav.stem}")

            records.append(Utterance(
                utt_id=utt_id,
                audio_path=str(wav),
                text=lyrics,
                singer_id=singer_id,
                technique=technique,
                group=group,
                textgrid_path=str(tg_path) if tg_path.exists() else None,
                json_path=str(json_path) if json_path.exists() else None,
                extra={"song": song, "group_folder": group_folder},
            ))

        return self._apply_limit(records)


# ─────────────────────────────────────────────────────────────────────────────
#  VocalSet adapter — template / stub for a second corpus
# ─────────────────────────────────────────────────────────────────────────────
@register_dataset("vocalset")
class VocalSetAdapter(DatasetAdapter):
    """Adapter stub for the VocalSet corpus.

    VocalSet is organised as ``<singer>/<context>/<technique>/*.wav`` and is
    technique-rich but mostly *lyric-free* (vowels, scales, arpeggios). It is
    included here as a worked example of how to plug in a second dataset:
    fill in ``list_utterances()`` with VocalSet's real layout and supply lyric
    text from whatever transcript source you use.

    Until then it raises ``NotImplementedError`` so a misconfigured run fails
    loudly rather than silently producing nothing.
    """

    def list_utterances(self) -> list[Utterance]:
        """Not yet implemented — see the class docstring.

        Raises:
            NotImplementedError: Always, until the VocalSet crawl is written.
        """
        raise NotImplementedError(
            "VocalSetAdapter is a stub. Implement list_utterances() to crawl "
            "the VocalSet layout, then this dataset becomes selectable via "
            "data.name: vocalset in the config."
        )
