"""
pairing.py — paired (technique take, control take) iteration and split filters.

GTSinger records every phrase twice for each technique song: once with the
technique (``<Tech>_Group/NNNN.wav``) and once as a plain control
(``Control_Group/NNNN.wav``) — same singer, same lyrics, same score, same file
stem. That paired design lets us *measure* what each technique changes in the
real data (see ``features.py``) instead of guessing transform parameters.

This module also owns ``song_key`` — the split unit of the song-disjoint test
split — so that augmentation, calibration and fine-tuning all agree on which
songs are held out.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Technique folder → the real technique group folder(s) it contains. The
# Mixed_Voice_and_Falsetto folder holds two distinct techniques.
TECH_GROUPS: dict[str, list[str]] = {
    "Vibrato": ["Vibrato_Group"],
    "Breathy": ["Breathy_Group"],
    "Glissando": ["Glissando_Group"],
    "Pharyngeal": ["Pharyngeal_Group"],
    "Mixed_Voice_and_Falsetto": ["Falsetto_Group", "Mixed_Voice_Group"],
}

# Real technique group folder → the per-phone annotation flag that marks it.
GROUP_FLAG: dict[str, str] = {
    "Vibrato_Group": "vibrato",
    "Breathy_Group": "breathy",
    "Glissando_Group": "glissando",
    "Pharyngeal_Group": "pharyngeal",
    "Falsetto_Group": "falsetto",
    "Mixed_Voice_Group": "mix",
}

# Whisper ISO code ↔ GTSinger folder name.
LANGUAGE_ISO: dict[str, str] = {
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "russian": "ru", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it",
}


def song_key(wav: str | Path) -> str:
    """Language-local song identity used as the split unit.

    Both the original and the augmented trees share the layout
    ``<root>/<singer>/<technique>/<song>/<group>/<file>.wav``, so the song
    folder is two levels above the WAV. The title is NFKC-normalised (zip
    extraction can leave Hangul/kana decomposed), case-folded and
    whitespace-collapsed so that "Let It Go" and "let it go" — the same song
    under different singers/techniques — map to one key.

    Args:
        wav: Path to a WAV file inside a ``*_Group`` folder.

    Returns:
        The normalised song title.
    """
    title = unicodedata.normalize("NFKC", Path(wav).parent.parent.name)
    return re.sub(r"\s+", " ", title).strip().casefold()


def load_test_songs(language: str, repo: str | Path = ".") -> set[str]:
    """Load the held-out song keys of a language's song-disjoint test split.

    Args:
        language: Lowercase language name (e.g. "english").
        repo:     Repository root (manifests live in ``results/splits``).

    Returns:
        The set of ``song_key`` values in the test split.

    Raises:
        FileNotFoundError: If the split has not been generated yet
            (``python scripts/finetune_whisper.py --split-only --language <iso>``).
    """
    iso = LANGUAGE_ISO[language.lower()]
    path = Path(repo) / "results" / "splits" / f"test_manifest_{iso}.songs.json"
    with open(path, encoding="utf-8") as fh:
        return set(json.load(fh)["test_songs"])


def voice_type(singer_id: str) -> str:
    """Extract the voice type from a GTSinger singer ID ("EN-Alto-1" → "alto").

    Args:
        singer_id: Singer folder name ``<LANG>-<Range>-<n>``.

    Returns:
        Lowercase voice type ("soprano", "alto", "tenor", "bass"), or
        "unknown" if the ID does not follow the pattern.
    """
    parts = singer_id.split("-")
    return parts[1].lower() if len(parts) >= 3 else "unknown"


@dataclass(frozen=True)
class ClipPair:
    """A real technique take and its control take (same phrase, same singer).

    Attributes:
        language:     Lowercase language name.
        singer_id:    Singer folder name (e.g. "EN-Alto-1").
        tech_folder:  Technique folder ("Vibrato", "Mixed_Voice_and_Falsetto"...).
        group:        Real technique group ("Vibrato_Group", "Falsetto_Group"...).
        song:         Song folder name (raw, as on disk).
        stem:         File stem shared by both takes ("0003").
        tech_wav:     Path of the technique take.
        control_wav:  Path of the control take.
    """
    language: str
    singer_id: str
    tech_folder: str
    group: str
    song: str
    stem: str
    tech_wav: Path
    control_wav: Path

    @property
    def flag(self) -> str:
        """Annotation flag name of this pair's technique ("vibrato", "mix"...)."""
        return GROUP_FLAG[self.group]

    @property
    def song_key(self) -> str:
        """Split key of the pair's song."""
        return song_key(self.tech_wav)


def iter_pairs(root: str | Path, language: str,
               exclude_songs: set[str] | None = None,
               groups: list[str] | None = None) -> Iterator[ClipPair]:
    """Yield every (technique take, control take) pair of a language.

    Only pairs where both WAVs and both JSON annotations exist are yielded.

    Args:
        root:          Language root, e.g. ``data/GTSinger/English``.
        language:      Lowercase language name.
        exclude_songs: Song keys to skip — pass ``load_test_songs(...)`` so
            that calibration/augmentation never touches held-out songs.
        groups:        Restrict to these real technique groups (default: all).

    Yields:
        ``ClipPair`` records in deterministic (sorted) order.
    """
    root = Path(root)
    exclude_songs = exclude_songs or set()
    for singer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for tech_folder, tech_groups in TECH_GROUPS.items():
            tech_dir = singer_dir / tech_folder
            if not tech_dir.is_dir():
                continue
            for song_dir in sorted(p for p in tech_dir.iterdir() if p.is_dir()):
                ctrl_dir = song_dir / "Control_Group"
                if not ctrl_dir.is_dir():
                    continue
                for group in tech_groups:
                    if groups is not None and group not in groups:
                        continue
                    grp_dir = song_dir / group
                    if not grp_dir.is_dir():
                        continue
                    for tech_wav in sorted(grp_dir.glob("*.wav")):
                        ctrl_wav = ctrl_dir / tech_wav.name
                        if not (ctrl_wav.exists()
                                and tech_wav.with_suffix(".json").exists()
                                and ctrl_wav.with_suffix(".json").exists()):
                            continue
                        if song_key(tech_wav) in exclude_songs:
                            continue
                        yield ClipPair(language, singer_dir.name, tech_folder,
                                       group, song_dir.name, tech_wav.stem,
                                       tech_wav, ctrl_wav)
