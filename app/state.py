"""
state.py — shared data index and helpers for the Gradio app.

The GTSinger tree has ~100k files; walking it on every UI interaction would be
slow, so it is indexed once into ``app/cache/index.parquet`` (rebuild with
``python -m app.state --rebuild``). Audio is never copied or bundled: the app
reads it from ``data/`` at runtime (GTSinger is CC BY-NC-SA — see REPRODUCE.md).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.augment.pairing import LANGUAGE_ISO, load_test_songs, song_key  # noqa: E402

DATA_ROOT = REPO / "data" / "GTSinger"
AUG_ROOT = REPO / "data" / "GTSinger_Augmented"
CACHE = REPO / "app" / "cache"
INDEX_PATH = CACHE / "index.parquet"

# Order used in every dropdown.
GROUP_ORDER = ["Control_Group", "Vibrato_Group", "Breathy_Group", "Glissando_Group",
               "Pharyngeal_Group", "Falsetto_Group", "Mixed_Voice_Group",
               "Paired_Speech_Group"]


def _scan(root: Path, origin: str) -> list[dict]:
    """Walk ``<root>/<Lang>/<Singer>/<Tech>/<Song>/<Group>/*.wav`` into rows.

    Args:
        root:   Dataset root (original or augmented).
        origin: "real" or "legacy" (augmented v1 files, stems ``NNNN_v{k}``).

    Returns:
        One dict per WAV.
    """
    rows = []
    for wav in root.glob("*/*/*/*/*/*.wav"):
        group_dir = wav.parent
        song_dir = group_dir.parent
        rows.append({
            "origin": origin,
            "language": song_dir.parent.parent.parent.name.lower(),
            "singer": song_dir.parent.parent.name,
            "tech_folder": song_dir.parent.name,
            "song": song_dir.name,
            "group": group_dir.name,
            "stem": wav.stem,
            "wav": str(wav.relative_to(REPO)),
            "song_key": song_key(wav),
        })
    return rows


def build_index() -> pd.DataFrame:
    """Scan the original and augmented trees and tag the split of each song.

    Returns:
        The index DataFrame (also written to ``INDEX_PATH``).
    """
    rows = _scan(DATA_ROOT, "real")
    if AUG_ROOT.is_dir():
        # Symlinked Control/Speech groups in the augmented tree duplicate the
        # originals — keep only the generated technique groups.
        rows += [r for r in _scan(AUG_ROOT, "legacy")
                 if r["group"] not in ("Control_Group", "Paired_Speech_Group")]
    df = pd.DataFrame(rows)

    # Split tag per language: songs in the song-disjoint test manifest → "test".
    df["split"] = "train"
    for lang in df["language"].unique():
        if lang not in LANGUAGE_ISO:
            continue
        try:
            test = load_test_songs(lang, REPO)
        except FileNotFoundError:
            continue
        mask = (df["language"] == lang) & df["song_key"].isin(test)
        df.loc[mask, "split"] = "test"

    df = df.sort_values(["language", "singer", "tech_folder", "song", "group", "stem"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INDEX_PATH, index=False)
    return df.reset_index(drop=True)


@lru_cache(maxsize=1)
def index() -> pd.DataFrame:
    """Return the cached index, building it on first use."""
    if INDEX_PATH.exists():
        return pd.read_parquet(INDEX_PATH)
    return build_index()


def real() -> pd.DataFrame:
    """Only the original GTSinger recordings."""
    df = index()
    return df[df["origin"] == "real"]


def options(df: pd.DataFrame, column: str) -> list[str]:
    """Sorted unique values of a column (dropdown choices)."""
    vals = df[column].dropna().unique().tolist()
    if column == "group":
        return sorted(vals, key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99)
    return sorted(vals)


def take(language: str, singer: str, tech_folder: str, song: str, group: str,
         stem: str, origin: str = "real") -> Path | None:
    """Absolute WAV path of one take, or ``None`` if it does not exist.

    Args:
        language, singer, tech_folder, song, group, stem: Index coordinates.
        origin: "real" or "legacy".

    Returns:
        The WAV path.
    """
    df = index()
    hit = df[(df["origin"] == origin) & (df["language"] == language)
             & (df["singer"] == singer) & (df["tech_folder"] == tech_folder)
             & (df["song"] == song) & (df["group"] == group) & (df["stem"] == stem)]
    return REPO / hit.iloc[0]["wav"] if len(hit) else None


def lyrics(json_path: Path) -> str:
    """Readable lyrics of an annotation (silence/breath tokens dropped)."""
    from alt.augment.io import load_meta
    from alt.text import join_words
    if not json_path.exists():
        return ""
    words = [w["word"] for w in load_meta(json_path)
             if not str(w["word"]).startswith("<")]
    lang = json_path.parts[json_path.parts.index("GTSinger") + 1].lower() \
        if "GTSinger" in json_path.parts else "english"
    return join_words(words, lang)


if __name__ == "__main__":
    if "--rebuild" in sys.argv or not INDEX_PATH.exists():
        d = build_index()
        print(f"indexed {len(d)} files → {INDEX_PATH}")
        print(d.groupby(["origin", "language"]).size())
