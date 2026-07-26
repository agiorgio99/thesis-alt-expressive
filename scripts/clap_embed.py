#!/usr/bin/env python3
"""
clap_embed.py — extract CLAP audio embeddings for the original and the
WORLD-augmented GTSinger trees.

Why
───
WER tells you whether Whisper *understood* the augmented audio; it does not
tell you whether the augmented audio *sounds like* the technique it claims to
be. CLAP (Contrastive Language-Audio Pretraining) maps audio into a 512-d
joint audio/text space trained on large music+speech corpora. Two clips that
sit close together in that space are perceptually/semantically similar. So:

    "Is a WORLD-vibrato sample acoustically in the same region as a real
     vibrato recording, or is it its own artefact cluster?"

becomes a measurable geometric question. This script only produces the
embeddings; ``clap_analysis.py`` does the statistics and the plots.

Design notes
────────────
* Metadata comes from ``alt.dataset.get_dataset`` so technique / singer /
  group labels are identical to the ones used in every ASR results CSV — the
  embedding tables join 1:1 with ``results/*/asr_*.csv`` on ``utt_id``.
* The augmented tree symlinks Control_Group and Paired_Speech_Group back to
  the originals. Rows are de-duplicated on ``Path.resolve()``, so a control
  WAV is embedded exactly once no matter how many roots point at it.
* ``origin`` is inferred per file, not per root: a file inside the augmented
  tree whose stem ends in ``_v<k>`` and that lives in a technique group is
  ``aug``; everything else is ``real``.
* CLAP has a 10 s receptive field. Longer utterances are split into
  non-overlapping windows, each window is embedded and L2-normalised, and the
  mean is re-normalised. Deterministic (unlike CLAP's default random crop).

Usage
─────
    python scripts/clap_embed.py \
        --roots data/GTSinger/English data/GTSinger_Augmented/English \
        --out results/clap

    # quick smoke test on 200 utterances
    python scripts/clap_embed.py --roots data/GTSinger/English --limit 200

    # only the shared held-out test split used in Phase 3
    python scripts/clap_embed.py --roots ... --manifest results/shared_test_manifest.json

Outputs
───────
    <out>/embeddings.npz     emb (N, D) float32 + utt_id (N,) str
    <out>/metadata.csv       one row per utterance, joins to emb by row order
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the local ``src/`` package importable without installing the project.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.dataset import get_dataset  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:                                   # pragma: no cover
    def tqdm(it, **kw):
        return it


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLAP_SR = 48_000          # CLAP is trained at 48 kHz, NOT the 16 kHz the ASR uses
WINDOW_S = 10.0           # CLAP's receptive field
DEFAULT_MODEL = "laion/larger_clap_music_and_speech"

# Augmented files are written as "{orig_stem}_v{k}.wav" by
# scripts/build_augmented_dataset.py — this is how we tell them apart.
_VARIANT_RE = re.compile(r"^(?P<stem>.+)_v(?P<k>\d+)$")


# ─────────────────────────────────────────────────────────────────────────────
#  Metadata collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_rows(roots: list[str], language: str, limit: int | None,
                 manifest: str | None) -> pd.DataFrame:
    """Crawl every dataset root and build a de-duplicated metadata table.

    Args:
        roots:    One or more dataset root folders (original and/or augmented).
        language: Language subset passed to the dataset adapter.
        limit:    Optional cap on utterances *per root* (smoke tests).
        manifest: Optional JSON manifest of audio_path strings to keep.

    Returns:
        A DataFrame with one row per unique audio file and the columns
        ``utt_id, audio_path, real_path, text, singer_id, technique, group,
        song, origin, variant, source_stem, root``.
    """
    rows: list[dict] = []
    seen_path: set[str] = set()      # catches symlinked Control_Group
    seen_id: set[str] = set()        # catches --copy'd Control_Group

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset root not found: {root_path}")

        adapter = get_dataset("gtsinger", root=root_path, language=language,
                              limit=limit, manifest=manifest)
        utts = adapter.list_utterances()
        print(f"[collect] {root_path}: {len(utts)} utterances")

        for u in utts:
            # build_augmented_dataset.py symlinks Control_Group/Paired_Speech_Group
            # into the augmented tree by default, but copies them under --copy.
            # Resolving the path catches the first case; the utt_id catches the
            # second (an identical copy has an identical id).
            real = str(Path(u.audio_path).resolve())
            if real in seen_path or u.utt_id in seen_id:
                continue
            seen_path.add(real)
            seen_id.add(u.utt_id)

            stem = Path(u.audio_path).stem
            m = _VARIANT_RE.match(stem)
            is_variant = m is not None
            # A file is augmented iff it is a "_v<k>" variant sitting in a
            # technique group. Control/speech groups are always real audio.
            origin = "aug" if (is_variant and u.group == "technique") else "real"

            rows.append({
                "utt_id":      u.utt_id,
                "audio_path":  u.audio_path,
                "real_path":   real,
                "text":        u.text,
                "singer_id":   u.singer_id,
                "technique":   u.technique,
                "group":       u.group,
                "song":        u.extra.get("song", ""),
                "group_folder": u.extra.get("group_folder", ""),
                "origin":      origin,
                "variant":     int(m.group("k")) if is_variant else -1,
                # source_stem lets you pair an augmented clip with the exact
                # Control_Group clip it was synthesised from.
                "source_stem": m.group("stem") if is_variant else stem,
                "root":        str(root_path),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No utterances collected — check --roots paths.")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CLAP model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class ClapEmbedder:
    """Thin wrapper around HuggingFace ``ClapModel`` for audio + text features.

    Args:
        model_name: HF hub id (default: laion/larger_clap_music_and_speech).
        device:     "cuda" or "cpu".
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda") -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        print(f"[clap] loading {model_name} on {self.device} …")
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()

        # Fused checkpoints expect truncation="fusion"; unfused ones "rand_trunc".
        enable_fusion = bool(
            getattr(getattr(self.model.config, "audio_config", None),
                    "enable_fusion", False)
        )
        self.truncation = "fusion" if enable_fusion else "rand_trunc"
        print(f"[clap] truncation mode: {self.truncation}")

    # ── audio ────────────────────────────────────────────────────────────────
    def embed_waveforms(self, waves: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of 48 kHz mono waveforms.

        Args:
            waves: List of 1-D float32 arrays, each <= 10 s.

        Returns:
            An ``(B, D)`` float32 array of L2-normalised audio embeddings.
        """
        inputs = self.processor(
            audio=waves, sampling_rate=CLAP_SR, return_tensors="pt",
            padding=True, truncation=self.truncation,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.get_audio_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = self.torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().numpy()

    # ── text ─────────────────────────────────────────────────────────────────
    def embed_texts(self, prompts: list[str]) -> np.ndarray:
        """Embed a list of natural-language prompts into the same space.

        Args:
            prompts: Free-text descriptions, e.g. "a singer using vibrato".

        Returns:
            An ``(B, D)`` float32 array of L2-normalised text embeddings.
        """
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.get_text_features(**inputs)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = self.torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Windowing
# ─────────────────────────────────────────────────────────────────────────────
def load_windows(path: str, window_s: float = WINDOW_S) -> list[np.ndarray]:
    """Load an audio file at 48 kHz and cut it into <= ``window_s`` chunks.

    Args:
        path:     Path to the audio file.
        window_s: Window length in seconds (CLAP's receptive field).

    Returns:
        A list of 1-D float32 arrays. Files shorter than the window return a
        single (unpadded) chunk; the processor handles the padding.
    """
    import librosa
    y, _ = librosa.load(path, sr=CLAP_SR, mono=True)
    y = y.astype(np.float32)
    if y.size == 0:
        return [np.zeros(int(CLAP_SR * 0.5), dtype=np.float32)]

    n = int(CLAP_SR * window_s)
    if y.size <= n:
        return [y]
    chunks = [y[i:i + n] for i in range(0, y.size, n)]
    # Drop a trailing sliver shorter than 1 s — it is mostly padding artefact.
    if chunks[-1].size < CLAP_SR and len(chunks) > 1:
        chunks = chunks[:-1]
    return chunks


def embed_dataframe(df: pd.DataFrame, embedder: ClapEmbedder,
                    batch_size: int = 16, window_s: float = WINDOW_S) -> np.ndarray:
    """Embed every row of the metadata table.

    Long utterances are window-averaged: each window is embedded, the vectors
    are averaged, and the mean is re-normalised to unit length.

    Args:
        df:         Metadata table from ``collect_rows``.
        embedder:   A ready ``ClapEmbedder``.
        batch_size: Number of *windows* per forward pass.
        window_s:   Window length in seconds.

    Returns:
        An ``(N, D)`` float32 array aligned row-for-row with ``df``.
    """
    all_windows: list[np.ndarray] = []
    owner: list[int] = []              # window index -> dataframe row index

    print("[embed] loading audio …")
    for i, path in enumerate(tqdm(df["audio_path"].tolist(), desc="load")):
        try:
            chunks = load_windows(path, window_s)
        except Exception as exc:                       # pragma: no cover
            print(f"[warn] failed to load {path}: {exc}")
            chunks = [np.zeros(int(CLAP_SR * 0.5), dtype=np.float32)]
        for c in chunks:
            all_windows.append(c)
            owner.append(i)

    print(f"[embed] {len(all_windows)} windows from {len(df)} utterances")
    vecs: list[np.ndarray] = []
    for s in tqdm(range(0, len(all_windows), batch_size), desc="clap"):
        vecs.append(embedder.embed_waveforms(all_windows[s:s + batch_size]))
    win_emb = np.concatenate(vecs, axis=0)

    dim = win_emb.shape[1]
    out = np.zeros((len(df), dim), dtype=np.float32)
    counts = np.zeros(len(df), dtype=np.int32)
    for w, row in enumerate(owner):
        out[row] += win_emb[w]
        counts[row] += 1
    out /= np.maximum(counts, 1)[:, None]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Extract CLAP embeddings for original + augmented GTSinger.")
    p.add_argument("--roots", nargs="+", required=True,
                   help="Dataset roots, e.g. data/GTSinger/English "
                        "data/GTSinger_Augmented/English")
    p.add_argument("--out", default="results/clap",
                   help="Output folder (default: results/clap)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF CLAP checkpoint id")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16,
                   help="Windows per forward pass (default: 16)")
    p.add_argument("--window-s", type=float, default=WINDOW_S,
                   help="Window length in seconds (default: 10)")
    p.add_argument("--language", default="english")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap utterances per root (smoke test)")
    p.add_argument("--manifest", default=None,
                   help="Optional JSON manifest of audio_path strings to keep")
    return p.parse_args()


def main() -> None:
    """Collect metadata, embed every utterance, write npz + csv."""
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_rows(args.roots, args.language, args.limit, args.manifest)
    print("\n[collect] rows per origin/group:")
    print(df.groupby(["origin", "group", "technique"]).size().to_string())

    embedder = ClapEmbedder(args.model, args.device)
    emb = embed_dataframe(df, embedder, args.batch_size, args.window_s)

    np.savez_compressed(out_dir / "embeddings.npz",
                        emb=emb, utt_id=df["utt_id"].to_numpy())
    df.to_csv(out_dir / "metadata.csv", index=False)

    print(f"\n=== wrote {emb.shape[0]} x {emb.shape[1]} embeddings to "
          f"{out_dir / 'embeddings.npz'} ===")
    print(f"=== metadata: {out_dir / 'metadata.csv'} ===")


if __name__ == "__main__":
    main()
