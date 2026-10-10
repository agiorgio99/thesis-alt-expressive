"""
engines.py — one entry point for every augmentation engine.

Callers (the batch builder, the pilot scripts, the Gradio app) ask for
"technique group X on clip Y with engine Z and parameters P" and get audio +
updated labels back, without knowing how each engine works internally. New
engines (``world_v2``, ``psola``) register themselves in ``ENGINES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .io import VOWELS_BY_LANGUAGE, load_utterance


@dataclass
class AugmentResult:
    """Output of one augmentation call.

    Attributes:
        wav:    Augmented waveform (mono float64).
        sr:     Its sample rate.
        meta:   Updated word annotation (technique flags set on changed words).
        engine: Engine name that produced it.
        params: Parameters actually used.
    """
    wav: np.ndarray
    sr: int
    meta: list[dict]
    engine: str
    params: dict


@dataclass
class Engine:
    """An augmentation engine registered for the callers.

    Attributes:
        name:        Registry key ("legacy", "world_v2", "psola").
        description: One-line human description (shown in the app).
        groups:      Real technique groups it can synthesise
                     ("Vibrato_Group", "Falsetto_Group", ...).
        presets:     ``presets(group, language)`` → list of parameter dicts
                     (the variants v0..vN).
        run:         ``run(wav_path, json_path, group, params, language)``
                     → ``AugmentResult``.
    """
    name: str
    description: str
    groups: list[str]
    presets: Callable[[str, str], list[dict]]
    run: Callable[..., AugmentResult]


# ── legacy (v1) ──────────────────────────────────────────────────────────────
# Real technique group → key of legacy.TECHNIQUE_CONFIG. v1 has no Mixed Voice.
_LEGACY_KEY: dict[str, str] = {
    "Vibrato_Group": "Vibrato",
    "Breathy_Group": "Breathy",
    "Glissando_Group": "Glissando",
    "Pharyngeal_Group": "Pharyngeal",
    "Falsetto_Group": "Mixed_Voice_and_Falsetto",
}


def _legacy_presets(group: str, language: str) -> list[dict]:
    """v1 presets are language-independent (chosen on English)."""
    from . import legacy
    return [dict(p) for p in legacy.TECHNIQUE_CONFIG[_LEGACY_KEY[group]]["params"]]


def _legacy_run(wav_path: str | Path, json_path: str | Path, group: str,
                params: dict, language: str) -> AugmentResult:
    """Run one v1 transform exactly as the batch builder does (16 kHz, full clip)."""
    from . import legacy
    cfg = legacy.TECHNIQUE_CONFIG[_LEGACY_KEY[group]]
    wav, meta = load_utterance(wav_path, json_path, sr=legacy.SR)
    f0, sp, ap = legacy.world_decompose(wav)
    out, meta_aug = cfg["augment_fn"](f0, sp, ap, meta, params,
                                      VOWELS_BY_LANGUAGE[language])
    return AugmentResult(out, legacy.SR, meta_aug, "legacy", params)


ENGINES: dict[str, Engine] = {
    "legacy": Engine(
        name="legacy",
        description="v1 — original WORLD transforms, 16 kHz, whole-clip "
                    "resynthesis, fixed English presets",
        groups=list(_LEGACY_KEY),
        presets=_legacy_presets,
        run=_legacy_run,
    ),
}


# ── world_v2 (data-driven, 48 kHz, local resynthesis) ───────────────────────
# Imported lazily inside the callables: world_v2 imports this module.
def _v2_presets(group: str, language: str) -> list[dict]:
    """Quartile variants from the language's calibration (see world_v2)."""
    from . import world_v2
    return world_v2.presets(group, language)


def _v2_run(*args, **kwargs) -> AugmentResult:
    """Run the v2 engine (see ``world_v2.run``)."""
    from . import world_v2
    return world_v2.run(*args, **kwargs)


ENGINES["world_v2"] = Engine(
    name="world_v2",
    description="v2 — data-driven per-language parameters, 48 kHz, local "
                "resynthesis with crossfades, applied where the singer used "
                "the technique",
    groups=["Vibrato_Group", "Glissando_Group", "Breathy_Group", "Pharyngeal_Group",
            "Falsetto_Group", "Mixed_Voice_Group"],
    presets=_v2_presets,
    run=_v2_run,
)


# ── psola (Praat overlap-add, pitch techniques only) ────────────────────────
def _psola_presets(group: str, language: str) -> list[dict]:
    """Same data-driven variants as v2 (see psola.presets)."""
    from . import psola
    return psola.presets(group, language)


def _psola_run(*args, **kwargs) -> AugmentResult:
    """Run the PSOLA engine (see ``psola.run``)."""
    from . import psola
    return psola.run(*args, **kwargs)


ENGINES["psola"] = Engine(
    name="psola",
    description="PSOLA (Praat overlap-add) — same data-driven pitch gestures as "
                "v2 but re-spacing the recording's own periods (pitch techniques only)",
    groups=["Vibrato_Group", "Glissando_Group", "Falsetto_Group"],
    presets=_psola_presets,
    run=_psola_run,
)


def augment(engine: str, wav_path: str | Path, group: str, params: dict,
            language: str, json_path: str | Path | None = None) -> AugmentResult:
    """Augment one control clip toward a technique with a registered engine.

    Args:
        engine:    Registry key in ``ENGINES``.
        wav_path:  Control-take WAV.
        group:     Target real technique group ("Vibrato_Group", ...).
        params:    Engine parameters (e.g. one of ``Engine.presets(...)``).
        language:  Lowercase language name (selects the vowel inventory).
        json_path: Annotation path (default: next to ``wav_path``).

    Returns:
        The ``AugmentResult``.

    Raises:
        KeyError:   Unknown engine.
        ValueError: The engine does not support ``group``.
    """
    eng = ENGINES[engine]
    if group not in eng.groups:
        raise ValueError(f"engine '{engine}' does not support {group} "
                         f"(supported: {eng.groups})")
    json_path = json_path or Path(wav_path).with_suffix(".json")
    return eng.run(wav_path, json_path, group, params, language)
