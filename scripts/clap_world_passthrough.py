#!/usr/bin/env python3
"""
clap_world_passthrough.py — build and embed the WORLD "null augmentation"
control condition.

Why this exists
---------------
Every augmented clip carries two things at once: the technique transformation
we intended, and the WORLD analysis/resynthesis artefact we did not. Because
the artefact turns out to dominate the CLAP embedding, `fd_ratio` and
`magnitude_ratio` in the main analysis conflate the two, and the perfect
real-vs-augmented AUC cannot be attributed to either.

This script produces the missing reference: each Control_Group clip passed
through `pw.wav2world` -> `pw.synthesize` with **no parameter modified at
all**. Any distance between real audio and this population is pure vocoder
artefact. Subtracting it isolates what the technique transformations actually
contribute.

The rows are written with ``origin="resynth"`` and ``group="control_resynth"``
so that every existing section of clap_analysis.py — which filters on
``group in {technique, control}`` and ``origin in {real, aug}`` — ignores them
completely. The numbers already reported therefore do not change.

Usage
-----
    python scripts/clap_world_passthrough.py --device cuda

By default this merges the passthrough rows into ``results/clap`` in place.
Re-running is safe: rows from a previous run are replaced, not appended.
Then re-run the analysis over the enlarged embedding set:

    python scripts/clap_analysis.py --emb-dir results/clap --projection tsne
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

try:
    import pyworld as pw
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyworld"])
    import pyworld as pw

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# Must match build_augmented_dataset.py exactly, or the artefact measured here
# is not the artefact present in the augmented clips.
SR = 16_000
FRAME_MS = 5.0


def load_wav(path: Path) -> np.ndarray:
    """Load a wav as float64 mono at SR, resampling if needed."""
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float64)
    if sr != SR:
        if not _LIBROSA:
            raise RuntimeError(f"{path}: sr={sr} != {SR} and librosa missing")
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr,
                               target_sr=SR).astype(np.float64)
    return wav


def world_roundtrip(wav: np.ndarray) -> np.ndarray:
    """Decompose and immediately resynthesise, changing nothing."""
    f0, sp, ap = pw.wav2world(wav, SR, frame_period=FRAME_MS)
    return pw.synthesize(f0, sp, ap, SR, frame_period=FRAME_MS)


def resolve_audio(row: pd.Series) -> Path | None:
    """Find a control clip on disk, preferring the repo-relative path."""
    for key in ("audio_path", "real_path"):
        raw = row.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        p = Path(raw)
        for cand in (p, REPO / p):
            if cand.is_file():
                return cand
    return None


def build_passthrough(df: pd.DataFrame, audio_out: Path) -> pd.DataFrame:
    """Resynthesise every control clip and return the new metadata rows."""
    ctrl = df[(df["group"] == "control") & (df["origin"] == "real")]
    if ctrl.empty:
        sys.exit("[error] no control rows found in metadata.csv")
    print(f"[passthrough] {len(ctrl)} control clips -> {audio_out}")

    rows, missing, failed = [], 0, 0
    for _, r in tqdm(list(ctrl.iterrows()), desc="world roundtrip"):
        src = resolve_audio(r)
        if src is None:
            missing += 1
            continue
        rel = Path(str(r["audio_path"]))
        dst = audio_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            try:
                sf.write(dst, world_roundtrip(load_wav(src)).astype(np.float32), SR)
            except Exception as exc:
                print(f"[warn] {src}: {exc}")
                failed += 1
                continue
        new = r.copy()
        new["utt_id"] = f"{r['utt_id']}__resynth"
        new["audio_path"] = str(dst)
        new["real_path"] = str(dst.resolve())
        new["origin"] = "resynth"          # invisible to origin in {real, aug}
        new["group"] = "control_resynth"   # invisible to group in {technique, control}
        new["variant"] = -2
        rows.append(new)

    if missing:
        print(f"[warn] {missing} control clips not found on disk — check --emb-dir "
              f"was produced on this machine, or pass --audio-root")
    if failed:
        print(f"[warn] {failed} clips failed to resynthesise")
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb-dir", default="results/clap",
                   help="existing embedding folder (embeddings.npz + metadata.csv)")
    p.add_argument("--audio-out", default="data/GTSinger_Resynth",
                   help="where the resynthesised wavs are written")
    p.add_argument("--out", default="results/clap",
                   help="embedding folder to write; defaults to --emb-dir, "
                        "i.e. the passthrough rows are merged in place "
                        "(safe to re-run: old passthrough rows are replaced)")
    p.add_argument("--model", default="laion/larger_clap_music_and_speech")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--audio-only", action="store_true",
                   help="write the wavs but skip embedding")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir, out_dir = Path(args.emb_dir), Path(args.out)
    npz = np.load(emb_dir / "embeddings.npz", allow_pickle=True)
    emb = npz["emb"].astype(np.float32)
    ids = list(npz["utt_id"])
    df = pd.read_csv(emb_dir / "metadata.csv")
    assert len(df) == len(emb), "metadata.csv and embeddings.npz are out of sync"

    # Running in place (--out == --emb-dir) must stay idempotent: drop any
    # passthrough rows left by an earlier run before regenerating them.
    keep = (df["origin"] != "resynth").to_numpy()
    if not keep.all():
        print(f"[merge] replacing {int((~keep).sum())} passthrough rows "
              f"from a previous run")
        emb = emb[keep]
        ids = [i for i, k in zip(ids, keep) if k]
        df = df[keep].reset_index(drop=True)

    new_df = build_passthrough(df, Path(args.audio_out))
    if new_df.empty:
        sys.exit("[error] nothing resynthesised")
    if args.audio_only:
        print(f"[done] wrote {len(new_df)} wavs to {args.audio_out}")
        return

    # Reuse the exact embedder and windowing of the original run.
    from clap_embed import ClapEmbedder, embed_dataframe

    embedder = ClapEmbedder(args.model, args.device)
    new_emb = embed_dataframe(new_df.reset_index(drop=True), embedder,
                              batch_size=args.batch_size)
    new_emb = np.asarray(new_emb, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat([df, new_df], ignore_index=True)
    all_emb = np.vstack([emb, new_emb])
    all_ids = ids + list(new_df["utt_id"])
    assert len(all_df) == len(all_emb) == len(all_ids)

    np.savez(out_dir / "embeddings.npz",
             emb=all_emb, utt_id=np.array(all_ids, dtype=object))
    all_df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"\n[done] {len(all_df)} utterances written to {out_dir}")
    print(all_df.groupby(["origin", "group"]).size().to_string())
    print(f"\nNext:  python scripts/clap_analysis.py --emb-dir {out_dir} "
          f"--projection tsne")


if __name__ == "__main__":
    main()
