"""
config.py — experiment configuration (hybrid YAML + typed dataclasses).

Design
------
* Every config section is a frozen-ish ``@dataclass`` so you get IDE
  autocomplete and type checking instead of dictionary-key guessing.
* ``load_config()`` reads a YAML file (see ``configs/baseline.yaml``) and
  fills those dataclasses.
* ``apply_overrides()`` lets the CLI patch any field with dotted keys
  (e.g. ``asr.device=cpu``) so you can run quick variants without editing YAML.

Why a registry-friendly config?
The whole pipeline is built so that *which* model runs each subtask is just a
string in this config. Change ``asr.model_name`` or ``alignment.aligner`` and a
different implementation is selected — no code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


# ─────────────────────────────────────────────────────────────────────────────
#  Section dataclasses — one per pipeline subtask
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PathsConfig:
    """Filesystem locations used across the whole run.

    Attributes:
        dataset_root: Root folder of the chosen dataset.
        results_dir:  Folder where metric CSVs and plots are written.
        work_dir:     Scratch folder for 16 kHz WAVs, MFA corpus, etc.
    """
    dataset_root: str = "data/GTSinger_English"
    results_dir: str = "results"
    work_dir: str = "data/_work"


@dataclass
class DataConfig:
    """Dataset-selection options.

    Attributes:
        name:     Registry key picking the DatasetAdapter (e.g. "gtsinger").
        language: Dataset-specific language filter.
        limit:    If set, keep only the first N utterances (fast smoke tests).
    """
    name: str = "gtsinger"
    language: str = "english"
    limit: int | None = None


@dataclass
class ASRConfig:
    """Automatic-speech-recognition subtask options.

    Attributes:
        enabled:    Run the ASR stage at all.
        model_name: Registry key picking the ASRModel implementation.
        device:     "cuda" or "cpu" — chosen independently of other subtasks.
        batch_size: Utterances processed per inference batch.
        language:   Decoding language hint passed to the model.
        extra:      Free-form model-specific keyword arguments.
    """
    enabled: bool = True
    model_name: str = "whisper_largev3"
    device: str = "cuda"
    batch_size: int = 8
    language: str = "en"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentConfig:
    """Forced-alignment subtask options.

    Attributes:
        enabled: Run the alignment stage at all.
        aligner: Registry key picking the Aligner implementation (mfa | sofa).
        device:  "cuda" or "cpu" (MFA is CPU-only; SOFA can use a GPU).
        extra:   Aligner-specific keyword arguments (model names, checkpoints).
    """
    enabled: bool = True
    aligner: str = "mfa"
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PitchConfig:
    """Pitch-tracking (CREPE) subtask options.

    Attributes:
        enabled:        Run the F0-extraction stage at all.
        model_capacity: CREPE model size (tiny | small | medium | large | full).
        device:         "cuda" or "cpu".
        step_ms:        F0 frame step in milliseconds.
    """
    enabled: bool = True
    model_capacity: str = "tiny"
    device: str = "cuda"
    step_ms: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation / reporting options.

    Attributes:
        stratify_by:              Metadata columns to break metrics down by.
        hallucination_threshold:  WER >= this value flags a hallucination.
    """
    stratify_by: list[str] = field(default_factory=lambda: ["technique", "singer_id"])
    hallucination_threshold: float = 1.0


@dataclass
class ExperimentConfig:
    """Top-level config aggregating every subtask section.

    Attributes:
        experiment_name: Human-readable run name (used to name output folders).
        paths:           PathsConfig instance.
        data:            DataConfig instance.
        asr:             ASRConfig instance.
        alignment:       AlignmentConfig instance.
        pitch:           PitchConfig instance.
        evaluation:      EvaluationConfig instance.
    """
    experiment_name: str = "baseline"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


# ─────────────────────────────────────────────────────────────────────────────
#  YAML loading
# ─────────────────────────────────────────────────────────────────────────────
def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate a (possibly nested) dataclass from a dict.

    Args:
        cls:  The dataclass type to build.
        data: A dict whose keys map onto the dataclass fields. Unknown keys
              raise a ``KeyError`` so typos in YAML fail loudly.

    Returns:
        An instance of ``cls`` with nested dataclass fields built recursively.
    """
    if not isinstance(data, dict):
        return data

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")

    # ``get_type_hints`` resolves string annotations (created by
    # ``from __future__ import annotations``) back into real type objects, so
    # nested config dataclasses can be detected and built recursively.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue                                  # keep dataclass default
        field_type = hints.get(name)
        if is_dataclass(field_type):
            kwargs[name] = _build_dataclass(field_type, data[name])
        else:
            kwargs[name] = data[name]
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the YAML config file (e.g. "configs/baseline.yaml").

    Returns:
        A fully populated ExperimentConfig. Fields absent from the YAML keep
        their dataclass defaults.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build_dataclass(ExperimentConfig, raw)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI overrides
# ─────────────────────────────────────────────────────────────────────────────
def _coerce(value: str) -> Any:
    """Convert a CLI string token into the most plausible Python type.

    Args:
        value: The raw string from the command line.

    Returns:
        ``None``/``bool``/``int``/``float`` when the string clearly encodes one,
        otherwise the original string.
    """
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Patch config fields in place using dotted ``key=value`` CLI strings.

    Args:
        cfg:       The ExperimentConfig to mutate.
        overrides: Strings like ``["asr.device=cpu", "data.limit=20"]``.

    Returns:
        The same ``cfg`` object, mutated for convenient chaining.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        dotted, raw_value = item.split("=", 1)
        target: Any = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:                              # walk into nested sections
            target = getattr(target, p)
        if not hasattr(target, leaf):
            raise AttributeError(f"No config field named {dotted!r}")
        setattr(target, leaf, _coerce(raw_value))
    return cfg
