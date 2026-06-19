"""
alignment.py — forced-alignment wrappers (MFA, SOFA) and a registry.

A forced aligner takes audio + its known transcript and returns time-stamped
word and phoneme intervals. As with ASR, the concrete aligner is picked by a
string key in the config (``alignment.aligner``).

Both supported aligners are external command-line tools, so these wrappers
mostly: (1) build the corpus layout each tool expects, (2) shell out to run it,
(3) parse the resulting TextGrids back into a common ``AlignmentResult``.

Add a new aligner in three steps:
    1. Subclass ``Aligner`` and implement ``align()``.
    2. Register it with ``@register_aligner("yourname")``.
    3. Set ``alignment.aligner: yourname`` in the YAML config.
"""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .text import SKIP_LABELS, clean_text


# ─────────────────────────────────────────────────────────────────────────────
#  Common result types
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Interval:
    """A single labelled time span.

    Attributes:
        label: The word or phoneme symbol.
        start: Start time in seconds.
        end:   End time in seconds.
    """
    label: str
    start: float
    end: float


@dataclass
class AlignmentResult:
    """Aligned word and phoneme intervals for one utterance.

    Attributes:
        utt_id: The utterance identifier.
        words:  Word-level intervals.
        phones: Phoneme-level intervals.
    """
    utt_id: str
    words: list[Interval] = field(default_factory=list)
    phones: list[Interval] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base class
# ─────────────────────────────────────────────────────────────────────────────
class Aligner(ABC):
    """Common interface for every forced-alignment wrapper.

    Args:
        device: "cuda" or "cpu" (MFA ignores this; SOFA can use a GPU).
        extra:  Aligner-specific keyword arguments.
    """

    def __init__(self, device: str = "cpu", **extra: Any) -> None:
        self.device = device
        self.extra = extra

    @abstractmethod
    def align(self, utterances: list, work_dir: str | Path) -> dict[str, AlignmentResult]:
        """Align a list of utterances and return their interval results.

        Args:
            utterances: ``Utterance`` objects (each needs audio + text).
            work_dir:   Scratch folder for the corpus and tool output.

        Returns:
            A dict mapping ``utt_id`` -> ``AlignmentResult``. Utterances the
            aligner could not process are simply absent from the dict.
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
ALIGNER_REGISTRY: dict[str, type[Aligner]] = {}


def register_aligner(name: str) -> Callable[[type[Aligner]], type[Aligner]]:
    """Class decorator that registers an ``Aligner`` under a name.

    Args:
        name: Registry key (matches ``alignment.aligner`` in the config).

    Returns:
        A decorator returning the class unchanged.
    """
    def _decorator(cls: type[Aligner]) -> type[Aligner]:
        ALIGNER_REGISTRY[name] = cls
        return cls
    return _decorator


def get_aligner(name: str, device: str = "cpu", **extra: Any) -> Aligner:
    """Instantiate the aligner registered under ``name``.

    Args:
        name:   Registry key (e.g. "mfa").
        device: "cuda" or "cpu".
        extra:  Aligner-specific keyword arguments.

    Returns:
        A ready-to-use ``Aligner`` instance.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in ALIGNER_REGISTRY:
        raise KeyError(
            f"Unknown aligner {name!r}. Registered: {sorted(ALIGNER_REGISTRY)}"
        )
    return ALIGNER_REGISTRY[name](device=device, **extra)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared TextGrid parsing helper
# ─────────────────────────────────────────────────────────────────────────────
def parse_textgrid(tg_path: str | Path) -> tuple[list[Interval], list[Interval]]:
    """Parse word and phoneme tiers from a TextGrid file.

    Args:
        tg_path: Path to a ``.TextGrid`` file.

    Returns:
        A tuple ``(words, phones)`` of ``Interval`` lists. Empty lists are
        returned if the file is missing or has no matching tiers.
    """
    words: list[Interval] = []
    phones: list[Interval] = []
    tg_path = Path(tg_path)
    if not tg_path.exists():
        return words, phones
    try:
        from praatio import textgrid as ptextgrid
        tg = ptextgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
        for name in tg.tierNames:
            ivs = [Interval(e.label, e.start, e.end)
                   for e in tg.getTier(name).entries
                   if e.label.strip() not in SKIP_LABELS]
            if "word" in name.lower():
                words = ivs
            elif "phone" in name.lower():
                phones = ivs
    except Exception:
        pass
    return words, phones


# ─────────────────────────────────────────────────────────────────────────────
#  MFA — Montreal Forced Aligner
# ─────────────────────────────────────────────────────────────────────────────
@register_aligner("mfa")
class MFAAligner(Aligner):
    """Montreal Forced Aligner wrapper (GMM-HMM, speech-trained).

    Requires the ``mfa`` command on ``PATH`` (install via conda-forge). The
    pretrained acoustic model + dictionary names are taken from ``extra``::

        alignment:
          aligner: mfa
          extra:
            acoustic_model: english_mfa
            dictionary:     english_mfa
            mfa_bin:        mfa          # optional explicit path to the binary
    """

    def _run(self, *args: str) -> None:
        """Run an ``mfa`` subcommand, raising on failure.

        Args:
            args: Arguments appended after the ``mfa`` binary.

        Raises:
            RuntimeError: If the subprocess exits with a non-zero code.
        """
        import os
        mfa_bin = self.extra.get("mfa_bin", "mfa")
        cmd = [mfa_bin, *map(str, args)]
        print("  $ " + " ".join(cmd))
        env = os.environ.copy()
        env["PATH"] = str(Path(mfa_bin).parent) + ":" + env.get("PATH", "")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"MFA failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}")

    def align(self, utterances: list, work_dir: str | Path) -> dict[str, AlignmentResult]:
        """Build an MFA corpus, run alignment, and parse the output TextGrids.

        Args:
            utterances: ``Utterance`` objects with audio + ground-truth text.
            work_dir:   Scratch folder; ``<work_dir>/mfa_corpus`` and
                        ``<work_dir>/mfa_output`` are created inside it.

        Returns:
            A dict mapping ``utt_id`` -> ``AlignmentResult``.
        """
        work_dir = Path(work_dir)
        corpus = work_dir / "mfa_corpus"
        output = work_dir / "mfa_output"
        corpus.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)

        # 1. Build the corpus: one WAV + matching .lab transcript per utterance.
        for utt in utterances:
            lyrics = clean_text(utt.text)
            if not lyrics:
                continue
            dst_wav = corpus / f"{utt.utt_id}.wav"
            if not dst_wav.exists():
                shutil.copy2(utt.audio_path, dst_wav)
            (corpus / f"{utt.utt_id}.lab").write_text(lyrics, encoding="utf-8")

        acoustic = self.extra.get("acoustic_model", "english_mfa")
        dictionary = self.extra.get("dictionary", "english_mfa")

        # 2. Run alignment (validation is best-effort; alignment is required).
        try:
            self._run("validate", corpus, dictionary, "--ignore_acoustics", "--clean")
        except RuntimeError as exc:
            print(f"  [MFA] validate warning: {exc}")
        self._run("align", "--clean", corpus, dictionary, acoustic, output)

        # 3. Parse every output TextGrid back into AlignmentResult objects.
        results: dict[str, AlignmentResult] = {}
        for utt in utterances:
            tg = output / f"{utt.utt_id}.TextGrid"
            if not tg.exists():
                continue
            words, phones = parse_textgrid(tg)
            results[utt.utt_id] = AlignmentResult(utt.utt_id, words, phones)
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  SOFA — Singing-Oriented Forced Aligner
# ─────────────────────────────────────────────────────────────────────────────
@register_aligner("sofa")
class SOFAAligner(Aligner):
    """SOFA wrapper (neural, singing-trained forced aligner).

    SOFA is not on PyPI and its inference is run from its own repo. This
    wrapper builds the corpus SOFA expects and, if SOFA output TextGrids are
    already present, parses them. Point the config at SOFA via ``extra``::

        alignment:
          aligner: sofa
          extra:
            output_dir: /path/to/sofa/output   # where SOFA wrote TextGrids

    Workflow: run this once to build the corpus, run SOFA externally on it,
    then run again with ``output_dir`` set so the TextGrids are parsed.
    """

    def align(self, utterances: list, work_dir: str | Path) -> dict[str, AlignmentResult]:
        """Build the SOFA corpus and parse SOFA output TextGrids if available.

        Args:
            utterances: ``Utterance`` objects with audio + ground-truth text.
            work_dir:   Scratch folder; ``<work_dir>/sofa_segments`` holds the
                        per-utterance input folders SOFA expects.

        Returns:
            A dict mapping ``utt_id`` -> ``AlignmentResult``. It is empty until
            SOFA has been run externally and ``extra['output_dir']`` is set.
        """
        work_dir = Path(work_dir)
        segments = work_dir / "sofa_segments"
        segments.mkdir(parents=True, exist_ok=True)

        # 1. Build SOFA corpus: one folder per utterance with audio.wav+audio.lab.
        for utt in utterances:
            lyrics = clean_text(utt.text)
            if not lyrics:
                continue
            utt_dir = segments / utt.utt_id
            utt_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(utt.audio_path, utt_dir / "audio.wav")
            (utt_dir / "audio.lab").write_text(lyrics, encoding="utf-8")

        # 2. Parse SOFA output if it has already been produced externally.
        output_dir = self.extra.get("output_dir")
        results: dict[str, AlignmentResult] = {}
        if not output_dir or not Path(output_dir).exists():
            print(f"  [SOFA] corpus built at {segments}. Run SOFA externally, "
                  f"then set alignment.extra.output_dir to parse results.")
            return results

        # SOFA writes one TextGrid per utterance; match by utt_id in the name.
        tg_by_id = {tg.stem: tg for tg in Path(output_dir).rglob("*.TextGrid")}
        for utt in utterances:
            tg = tg_by_id.get(utt.utt_id)
            if tg is None:
                continue
            words, phones = parse_textgrid(tg)
            results[utt.utt_id] = AlignmentResult(utt.utt_id, words, phones)
        return results
