"""
pipeline.py — the orchestrator that runs the baseline end to end.

Given an ``ExperimentConfig`` it:
    1. loads the chosen dataset into ``Utterance`` records;
    2. runs the chosen ASR model and scores WER/PER (overall + stratified);
    3. runs the chosen forced aligner and scores Time Boundary Error;
    4. runs CREPE pitch extraction and saves per-utterance F0 statistics.

Each stage is independent and guarded by its ``enabled`` flag, so you can run
just the ASR baseline, just alignment, etc. Every stage writes a CSV into
``results/<experiment_name>/`` so partial runs are never lost.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alignment import get_aligner, parse_textgrid
from .asr import get_asr_model
from .config import ExperimentConfig
from .dataset import Utterance, get_dataset
from .metrics import (aggregate_asr, aggregate_tbe, score_asr,
                      stratified_asr, time_boundary_error)
from .pitch import CrepeExtractor
from .report import make_report


def _iter(items, desc: str):
    """Wrap an iterable in a tqdm progress bar when tqdm is installed.

    Args:
        items: Any iterable.
        desc:  Progress-bar description.

    Returns:
        ``items`` wrapped in tqdm, or ``items`` unchanged if tqdm is missing.
    """
    try:
        from tqdm import tqdm
        return tqdm(items, desc=desc)
    except Exception:
        print(f"  {desc}...")
        return items


class Pipeline:
    """Runs the configured baseline pipeline and writes results to disk.

    Args:
        config: A fully populated ``ExperimentConfig``.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.cfg = config
        # All outputs for this run live in results/<experiment_name>/.
        self.out_dir = Path(config.paths.results_dir) / config.experiment_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(config.paths.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.utterances: list[Utterance] = []

    # ── Stage 0: dataset ─────────────────────────────────────────────────────
    def load_dataset(self) -> list[Utterance]:
        """Load the configured dataset and cache its utterances.

        Returns:
            The list of ``Utterance`` records (also stored on ``self``).
        """
        cfg = self.cfg.data
        manifest = getattr(cfg, "manifest", None)
        print(f"[dataset] adapter={cfg.name}  root={self.cfg.paths.dataset_root}"
              + (f"  manifest={manifest}" if manifest else ""))
        adapter = get_dataset(cfg.name, root=self.cfg.paths.dataset_root,
                              language=cfg.language, limit=cfg.limit,
                              manifest=manifest)
        self.utterances = adapter.list_utterances()
        print(f"[dataset] loaded {len(self.utterances)} utterances")

        # Persist the inventory so later stages / inspection have a manifest.
        pd.DataFrame([u.__dict__ for u in self.utterances]).to_csv(
            self.out_dir / "inventory.csv", index=False)
        return self.utterances

    # ── Stage 1: ASR ─────────────────────────────────────────────────────────
    def run_asr(self) -> list[pd.DataFrame] | None:
        """Run every configured ASR model in sequence and score WER/PER.

        Returns:
            A list of per-utterance DataFrames (one per model), or ``None``
            if the ASR stage is disabled.
        """
        cfg = self.cfg.asr
        if not cfg.enabled:
            print("[asr] disabled — skipping")
            return None

        scored_utts = [u for u in self.utterances if u.text.strip()]
        paths = [u.audio_path for u in scored_utts]
        thr = self.cfg.evaluation.hallucination_threshold
        results = []

        for model_name in cfg.model_names:
            print(f"\n[asr] model={model_name}  device={cfg.device}")
            model = get_asr_model(model_name, device=cfg.device,
                                  batch_size=cfg.batch_size, language=cfg.language,
                                  **cfg.extra)
            hypotheses = model.transcribe(paths)
            model.unload()

            df = pd.DataFrame({
                "utt_id":    [u.utt_id    for u in scored_utts],
                "singer_id": [u.singer_id for u in scored_utts],
                "technique": [u.technique for u in scored_utts],
                "group":     [u.group     for u in scored_utts],
                "text":      [u.text      for u in scored_utts],
                "hypothesis": hypotheses,
            })
            df = score_asr(df, ref_col="text", hyp_col="hypothesis")
            df.to_csv(self.out_dir / f"asr_{model_name}.csv", index=False)

            overall = aggregate_asr(df, thr)
            print(f"[asr] WER={overall['wer']:.3f}  PER={overall['per']:.3f}  "
                  f"halluc={overall['hallucination_rate']:.3f}  n={overall['n']}")
            pd.DataFrame([overall]).to_csv(
                self.out_dir / f"asr_{model_name}_summary.csv", index=False)

            for col in self.cfg.evaluation.stratify_by:
                strat = stratified_asr(df, col, thr)
                if not strat.empty:
                    strat.to_csv(
                        self.out_dir / f"asr_{model_name}_by_{col}.csv",
                        index=False)
            results.append(df)

        return results

    # ── Stage 2: alignment ───────────────────────────────────────────────────
    def run_alignment(self) -> pd.DataFrame | None:
        """Run the configured forced aligner and score Time Boundary Error.

        Ground-truth intervals come from each utterance's own annotation
        TextGrid; predicted intervals come from the aligner.

        Returns:
            A per-utterance TBE DataFrame, or ``None`` if alignment is disabled
            or no aligner output was produced.
        """
        cfg = self.cfg.alignment
        if not cfg.enabled:
            print("[alignment] disabled — skipping")
            return None

        print(f"[alignment] aligner={cfg.aligner}  device={cfg.device}")
        aligner = get_aligner(cfg.aligner, device=cfg.device, **cfg.extra)
        # Need both audio+text (to align) and a GT TextGrid (to score against).
        utts = [u for u in self.utterances if u.text.strip() and u.textgrid_path]
        results = aligner.align(utts, self.work_dir)

        if not results:
            print("[alignment] no aligner output to score yet")
            return None

        rows, word_tbe, phone_tbe = [], [], []
        for utt in _iter(utts, "Scoring alignment"):
            pred = results.get(utt.utt_id)
            if pred is None:
                continue
            gt_words, gt_phones = parse_textgrid(utt.textgrid_path)
            w = time_boundary_error(pred.words, gt_words)
            p = time_boundary_error(pred.phones, gt_phones)
            word_tbe.append(w)
            phone_tbe.append(p)
            rows.append({
                "utt_id": utt.utt_id, "singer_id": utt.singer_id,
                "technique": utt.technique,
                "word_mean_tbe": w["mean_tbe"], "word_within_50ms": w["within_50ms"],
                "phone_mean_tbe": p["mean_tbe"], "phone_within_50ms": p["within_50ms"],
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / f"alignment_{cfg.aligner}_tbe.csv", index=False)

        word_agg = aggregate_tbe(word_tbe)
        phone_agg = aggregate_tbe(phone_tbe)
        print(f"[alignment] word  mean TBE = {word_agg['mean_tbe']}  "
              f"<50ms = {word_agg['within_50ms']}")
        print(f"[alignment] phone mean TBE = {phone_agg['mean_tbe']}  "
              f"<50ms = {phone_agg['within_50ms']}")
        pd.DataFrame([{"level": "word", **word_agg},
                      {"level": "phone", **phone_agg}]).to_csv(
            self.out_dir / f"alignment_{cfg.aligner}_summary.csv", index=False)
        return df

    # ── Stage 3: pitch ───────────────────────────────────────────────────────
    def run_pitch(self) -> pd.DataFrame | None:
        """Run CREPE F0 extraction and save per-utterance F0 statistics.

        Returns:
            A per-utterance F0-statistics DataFrame, or ``None`` if the pitch
            stage is disabled.
        """
        cfg = self.cfg.pitch
        if not cfg.enabled:
            print("[pitch] disabled — skipping")
            return None

        print(f"[pitch] CREPE capacity={cfg.model_capacity}  device={cfg.device}")
        extractor = CrepeExtractor(model_capacity=cfg.model_capacity,
                                   step_ms=cfg.step_ms, device=cfg.device)
        rows = []
        for utt in _iter(self.utterances, "CREPE F0"):
            contour = extractor.extract(utt.audio_path)
            if contour is None:
                continue
            stats = extractor.summary_stats(contour)
            if not stats:
                continue
            stats.update({"utt_id": utt.utt_id, "singer_id": utt.singer_id,
                          "technique": utt.technique, "group": utt.group})
            rows.append(stats)

        df = pd.DataFrame(rows)
        df.to_csv(self.out_dir / "pitch_f0_stats.csv", index=False)
        print(f"[pitch] F0 stats for {len(df)} utterances")
        return df

    # ── Full run ─────────────────────────────────────────────────────────────
    def run(self) -> None:
        """Run every enabled stage in order and report the output folder."""
        print(f"\n=== Experiment: {self.cfg.experiment_name} ===")
        self.load_dataset()
        self.run_asr()
        self.run_alignment()
        self.run_pitch()
        make_report(self.out_dir)
        print(f"\n=== Done. Results in: {self.out_dir} ===")
