"""
asr.py — Automatic Speech Recognition model wrappers and a registry.

Every ASR model behaves the same from the pipeline's point of view:
``load()`` once, ``transcribe(list_of_wav_paths)`` to get a list of strings,
``unload()`` to free memory. The concrete model is chosen by a string key in
the config (``asr.model_name``), resolved through ``ASR_REGISTRY``.

Add a new ASR model in three steps:
    1. Subclass ``ASRModel`` and implement ``load()`` + ``_transcribe_batch()``.
    2. Register a factory with ``@register_asr("yourname")``.
    3. Set ``asr.model_name: yourname`` in the YAML config.

Whisper uses the HuggingFace ``transformers`` implementation so the very same
checkpoints can be fine-tuned later in Phase 3.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import TARGET_SR, load_audio


# ─────────────────────────────────────────────────────────────────────────────
#  Abstract base class
# ─────────────────────────────────────────────────────────────────────────────
class ASRModel(ABC):
    """Common interface for every ASR model wrapper.

    Args:
        device:     "cuda" or "cpu" — where this model runs.
        batch_size: Utterances processed per forward pass.
        language:   Decoding language hint (ISO code, e.g. "en").
        extra:      Model-specific keyword arguments.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 8,
                 language: str = "en", **extra: Any) -> None:
        self.device = device
        self.batch_size = batch_size
        self.language = language
        self.extra = extra
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load weights / processor onto ``self.device``. Idempotent."""
        raise NotImplementedError

    @abstractmethod
    def _transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe one batch of audio files.

        Args:
            audio_paths: Paths of at most ``batch_size`` audio files.

        Returns:
            One lowercase hypothesis string per input path, in order.
        """
        raise NotImplementedError

    def transcribe(self, audio_paths: list[str]) -> list[str]:
        """Transcribe any number of audio files, batching automatically.

        Args:
            audio_paths: Paths of the audio files to transcribe.

        Returns:
            One hypothesis string per input path, in the same order. A failed
            batch yields empty strings for that batch rather than crashing.
        """
        if not self._loaded:
            self.load()
        hyps: list[str] = []
        n_total = len(audio_paths)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size
        try:
            from tqdm import tqdm
            bar = tqdm(range(0, n_total, self.batch_size), total=n_batches,
                       desc=f"ASR {self.__class__.__name__}", unit="batch",
                       postfix={"utt": f"0/{n_total}"})
        except ImportError:
            bar = range(0, n_total, self.batch_size)
        done = 0
        for i in bar:
            batch = audio_paths[i:i + self.batch_size]
            try:
                hyps.extend(self._transcribe_batch(batch))
            except Exception as exc:                  # keep the run alive
                print(f"  [ASR] batch {i // self.batch_size} failed: {exc}")
                hyps.extend([""] * len(batch))
            done += len(batch)
            if hasattr(bar, "set_postfix"):
                bar.set_postfix({"utt": f"{done}/{n_total}"})
        return hyps

    def unload(self) -> None:
        """Release model weights and free GPU memory if applicable."""
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────────────────
# Maps a config key -> a factory callable that returns an ASRModel.
ASR_REGISTRY: dict[str, Callable[..., ASRModel]] = {}


def register_asr(name: str) -> Callable[[Callable[..., ASRModel]], Callable[..., ASRModel]]:
    """Decorator that registers an ASR factory (a class or function) by name.

    Args:
        name: Registry key (matches ``asr.model_name`` in the config).

    Returns:
        A decorator returning the factory unchanged.
    """
    def _decorator(factory: Callable[..., ASRModel]) -> Callable[..., ASRModel]:
        ASR_REGISTRY[name] = factory
        return factory
    return _decorator


def get_asr_model(name: str, device: str = "cpu", batch_size: int = 8,
                  language: str = "en", **extra: Any) -> ASRModel:
    """Instantiate the ASR model registered under ``name``.

    Args:
        name:       Registry key (e.g. "whisper_largev3").
        device:     "cuda" or "cpu".
        batch_size: Utterances per forward pass.
        language:   Decoding language hint.
        extra:      Model-specific keyword arguments.

    Returns:
        An un-loaded ``ASRModel`` instance (call ``.transcribe`` to use it).

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in ASR_REGISTRY:
        raise KeyError(
            f"Unknown ASR model {name!r}. Registered: {sorted(ASR_REGISTRY)}"
        )
    return ASR_REGISTRY[name](device=device, batch_size=batch_size,
                              language=language, **extra)


# ─────────────────────────────────────────────────────────────────────────────
#  Whisper (HuggingFace transformers)
# ─────────────────────────────────────────────────────────────────────────────
class WhisperHF(ASRModel):
    """Whisper wrapper using ``transformers.WhisperForConditionalGeneration``.

    The HF implementation is used (rather than faster-whisper) so the exact
    same checkpoints can be fine-tuned in Phase 3.

    Args:
        model_id: HuggingFace model id (e.g. "openai/whisper-large-v3").
        (plus all base ``ASRModel`` arguments)
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id

    def load(self) -> None:
        """Load the Whisper processor and model onto ``self.device``."""
        if self._loaded:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self._loaded = True

    def _transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe a batch of WAVs with greedy/beam Whisper decoding.

        Args:
            audio_paths: Paths of the audio files in this batch.

        Returns:
            One lowercase hypothesis per input path.
        """
        import torch
        audios = [load_audio(p, sr=TARGET_SR)[0] for p in audio_paths]
        inputs = self.processor(audios, sampling_rate=TARGET_SR,
                                return_tensors="pt")
        features = inputs.input_features.to(self.device, self.model.dtype)
        with torch.no_grad():
            generated = self.model.generate(
                features, language=self.language, task="transcribe")
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [t.strip().lower() for t in texts]


# Register the three Whisper sizes used as baselines. Each factory just binds
# the matching HuggingFace model id; everything else is shared.
@register_asr("whisper_small")
def _whisper_small(**kw: Any) -> WhisperHF:
    """Factory for Whisper small. Returns a configured ``WhisperHF``."""
    return WhisperHF(model_id="openai/whisper-small", **kw)


@register_asr("whisper_largev2")
def _whisper_largev2(**kw: Any) -> WhisperHF:
    """Factory for Whisper large-v2. Returns a configured ``WhisperHF``."""
    return WhisperHF(model_id="openai/whisper-large-v2", **kw)


@register_asr("whisper_largev3")
def _whisper_largev3(**kw: Any) -> WhisperHF:
    """Factory for Whisper large-v3. Returns a configured ``WhisperHF``."""
    return WhisperHF(model_id="openai/whisper-large-v3", **kw)


@register_asr("whisper_finetuned")
def _whisper_finetuned(**kw: Any) -> WhisperHF:
    """Factory for a locally fine-tuned Whisper checkpoint.

    Pass the checkpoint path via ``asr.extra.model_id`` in the config, e.g.::

        asr:
          model_names: [whisper_finetuned]
          extra:
            model_id: results/finetune_whisper/best_model
    """
    model_id = kw.pop("model_id", "results/finetune_whisper/best_model")
    return WhisperHF(model_id=model_id, **kw)


# ─────────────────────────────────────────────────────────────────────────────
#  wav2vec 2.0 (HuggingFace transformers, CTC)
# ─────────────────────────────────────────────────────────────────────────────
@register_asr("wav2vec2")
class Wav2Vec2CTC(ASRModel):
    """wav2vec 2.0 CTC wrapper (``facebook/wav2vec2-large-960h`` by default).

    Args:
        (all base ``ASRModel`` arguments; pass ``model_id`` via ``extra`` to
        use a different checkpoint)
    """

    def load(self) -> None:
        """Load the wav2vec2 processor and CTC model onto ``self.device``."""
        if self._loaded:
            return
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        model_id = self.extra.get("model_id", "facebook/wav2vec2-large-960h")
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._loaded = True

    def _transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe a batch of WAVs with CTC argmax decoding.

        Args:
            audio_paths: Paths of the audio files in this batch.

        Returns:
            One lowercase hypothesis per input path.
        """
        import torch
        audios = [load_audio(p, sr=TARGET_SR)[0].astype(np.float32)
                  for p in audio_paths]
        inputs = self.processor(audios, sampling_rate=TARGET_SR,
                                return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)
        with torch.no_grad():
            logits = self.model(input_values, attention_mask=attn).logits
        pred_ids = torch.argmax(logits, dim=-1)
        texts = self.processor.batch_decode(pred_ids)
        return [t.strip().lower() for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
#  FireRedASR (external repository)
# ─────────────────────────────────────────────────────────────────────────────
@register_asr("fireredasr")
class FireRedASRModel(ASRModel):
    """FireRedASR-AED wrapper.

    FireRedASR is not on PyPI: clone https://github.com/FireRedTeam/FireRedASR
    and download the weights, then point the config at them via ``extra``::

        asr:
          model_name: fireredasr
          extra:
            repo_dir:  /path/to/FireRedASR
            model_dir: /path/to/FireRedASR-AED-L

    Args:
        (base ``ASRModel`` arguments; ``repo_dir`` and ``model_dir`` come
        through ``extra``)
    """

    def load(self) -> None:
        """Import FireRedASR from its cloned repo and load the AED model.

        Raises:
            FileNotFoundError: If ``repo_dir`` / ``model_dir`` are missing.
        """
        if self._loaded:
            return
        import torch
        import argparse

        repo_dir = self.extra.get("repo_dir")
        model_dir = self.extra.get("model_dir")
        if not repo_dir or not Path(repo_dir).exists():
            raise FileNotFoundError(
                "FireRedASR repo not found. Set asr.extra.repo_dir to a clone "
                "of https://github.com/FireRedTeam/FireRedASR")
        if not model_dir or not Path(model_dir).exists():
            raise FileNotFoundError(
                "FireRedASR weights not found. Set asr.extra.model_dir.")

        sys.path.insert(0, str(repo_dir))
        # FireRedASR checkpoints were saved with argparse.Namespace objects;
        # allow-list it so torch.load (weights_only) does not reject them.
        torch.serialization.add_safe_globals([argparse.Namespace])

        from fireredasr.models.fireredasr import FireRedAsr
        self.model = FireRedAsr.from_pretrained("aed", str(model_dir))
        # Decoding hyper-parameters (override via asr.extra if needed).
        self._params = {
            "use_gpu": 1 if self.device == "cuda" else 0,
            "beam_size": self.extra.get("beam_size", 3),
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0,
        }
        self._loaded = True

    def _transcribe_batch(self, audio_paths: list[str]) -> list[str]:
        """Transcribe a batch of WAVs with FireRedASR.

        Args:
            audio_paths: Paths of the audio files in this batch.

        Returns:
            One lowercase hypothesis per input path.
        """
        utt_ids = [Path(p).stem for p in audio_paths]
        results = self.model.transcribe(utt_ids, audio_paths, self._params)
        return [r.get("text", "").strip().lower() for r in results]
