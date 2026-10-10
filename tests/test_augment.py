"""
Tests for the alt.augment package.

Fast tests use synthetic signals only; tests that need the GTSinger data are
skipped automatically when ``data/GTSinger`` is not present.

Run:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from alt.augment import legacy  # noqa: E402
from alt.augment.io import VOWELS_BY_LANGUAGE, is_vowel, t2f  # noqa: E402
from alt.augment.labels import update_labels  # noqa: E402
from alt.augment.pairing import iter_pairs, song_key, voice_type  # noqa: E402

DATA = REPO / "data" / "GTSinger"


def _synthetic_utterance(dur: float = 1.2, f0_hz: float = 220.0):
    """A harmonic tone plus a one-word annotation with a long vowel.

    Returns:
        ``(wav, meta)`` at the legacy 16 kHz rate.
    """
    t = np.arange(int(dur * legacy.SR)) / legacy.SR
    wav = sum(np.sin(2 * np.pi * f0_hz * h * t) / h for h in range(1, 8)) * 0.2
    meta = [{
        "word": "la", "start_time": 0.1, "end_time": 1.1,
        "ph": ["L", "AA1"], "ph_start": [0.1, 0.2], "ph_end": [0.2, 1.1],
        "note": [57], "note_dur": [1.0], "tech": "0",
        "vibrato": ["0", "0"], "breathy": ["0", "0"], "glissando": ["0", "0"],
        "pharyngeal": ["0", "0"], "falsetto": ["0", "0"], "mix": ["0", "0"],
    }]
    return wav.astype(np.float64), meta


def test_song_key_normalises_case_space_and_unicode():
    """Same title with different case/spacing/Unicode form → same key."""
    a = song_key("x/EN-Alto-1/Vibrato/Let It  Go/Vibrato_Group/0000.wav")
    b = song_key("x/EN-Tenor-1/Breathy/let it go/Control_Group/0003.wav")
    assert a == b == "let it go"
    # Decomposed Hangul (NFD, as left by some zip tools) equals composed form.
    import unicodedata
    nfd = unicodedata.normalize("NFD", "사랑")
    assert song_key(f"r/s/t/{nfd}/g/0.wav") == song_key("r/s/t/사랑/g/0.wav")


def test_voice_type():
    assert voice_type("EN-Alto-1") == "alto"
    assert voice_type("ZH-Tenor-2") == "tenor"
    assert voice_type("weird") == "unknown"


def test_helpers():
    assert t2f(0.5) == 100                      # 5 ms frames
    assert is_vowel("AA1", VOWELS_BY_LANGUAGE["english"])
    assert not is_vowel("L", VOWELS_BY_LANGUAGE["english"])


def test_update_labels_sets_flags_and_tech_code():
    _, meta = _synthetic_utterance()
    out = update_labels(meta, [0], "vibrato")
    assert out[0]["vibrato"] == ["1", "1"] and out[0]["tech"] == "6"
    assert meta[0]["tech"] == "0"               # input left untouched


@pytest.mark.parametrize("technique", list(legacy.TECHNIQUE_CONFIG))
def test_legacy_transforms_are_deterministic_and_finite(technique):
    """Every v1 preset yields a finite waveform, identical on re-run."""
    wav, meta = _synthetic_utterance()
    f0, sp, ap = legacy.world_decompose(wav)
    cfg = legacy.TECHNIQUE_CONFIG[technique]
    for params in cfg["params"]:
        a, labels = cfg["augment_fn"](f0, sp, ap, meta, params,
                                      VOWELS_BY_LANGUAGE["english"])
        b, _ = cfg["augment_fn"](f0, sp, ap, meta, params,
                                 VOWELS_BY_LANGUAGE["english"])
        assert np.isfinite(a).all() and np.array_equal(a, b)
        assert abs(len(a) - len(wav)) < legacy.SR * 0.01
        assert labels[0][cfg["technique_key"]] == ["1", "1"]


@pytest.mark.skipif(not (DATA / "Russian").is_dir(), reason="GTSinger not downloaded")
def test_iter_pairs_matches_stems_and_excludes_songs():
    """Pairs share file stems; excluded songs never appear."""
    pairs = list(iter_pairs(DATA / "Russian", "russian"))
    assert pairs and all(p.tech_wav.stem == p.control_wav.stem for p in pairs)
    first = pairs[0].song_key
    rest = list(iter_pairs(DATA / "Russian", "russian", exclude_songs={first}))
    assert all(p.song_key != first for p in rest)
    assert len(rest) < len(pairs)


# ── Estimator round-trip tests (features.py) ─────────────────────────────────

from alt.augment import features  # noqa: E402


@pytest.mark.parametrize("rate,extent", [(5.0, 30.0), (6.0, 60.0), (7.0, 120.0)])
def test_vibrato_estimator_recovers_rate_and_extent(rate, extent):
    """A synthetic vibrato on a rising melody is measured within 10 %."""
    fs = 1000.0 / features.FRAME_MS
    t = np.arange(int(1.5 * fs)) / fs
    melody = 1200 * np.log2(330 / 440) + 100 * t          # slow upward drift
    cents = melody + extent * np.sin(2 * np.pi * rate * t)
    f0 = 440 * 2 ** (cents / 1200)
    v = features.vibrato_features(f0)
    assert abs(v["vib_rate_hz"] - rate) / rate < 0.10
    assert abs(v["vib_extent_cents"] - extent) / extent < 0.10
    assert v["vib_regularity"] > 0.5


def test_vibrato_estimator_rejects_short_spans():
    assert np.isnan(features.vibrato_features(np.full(20, 220.0))["vib_rate_hz"])


@pytest.mark.parametrize("glide", [-300.0, 200.0])
def test_glide_estimator_recovers_signed_size(glide):
    """A sigmoid glide of known size and direction is measured within 10 %."""
    n = 80
    s = 1 / (1 + np.exp(-np.linspace(-6, 6, n)))
    f0 = 300 * 2 ** (glide * s / 1200)
    g = features.glide_features(f0)
    assert np.sign(g["glide_cents"]) == np.sign(glide)
    assert abs(g["glide_cents"] - glide) / abs(glide) < 0.10


def test_quality_features_on_harmonic_tone():
    """Voice-quality features are finite and sensible on a harmonic tone."""
    sr = 48_000
    t = np.arange(int(1.0 * sr)) / sr
    wav = sum(np.sin(2 * np.pi * 220 * h * t) / h for h in range(1, 20)) * 0.1
    a = features.analyse(wav, sr, "alto")
    q = features.span_features(a, 0.2, 0.8)
    assert q["voiced_ratio"] > 0.9
    assert abs(q["f0_cents"] - 1200 * np.log2(220 / 440)) < 20
    assert q["tilt_db_oct"] < 0                         # 1/h harmonic roll-off
    assert 0 <= q["ap_0k_1k"] <= 1 and len(q["ltas"]) == len(features.LTAS_CENTRES)


# ── world_core (v2 machinery) ────────────────────────────────────────────────

from alt.augment import world_core  # noqa: E402


def test_splice_keeps_original_outside_regions():
    """Only span ± pad ± crossfade may differ from the original."""
    sr = 48_000
    orig = np.random.default_rng(0).standard_normal(sr)          # 1 s
    voc = np.zeros(sr)
    out = world_core.splice(orig, voc, sr, [(0.4, 0.6)], pad_s=0.04, xfade_s=0.03)
    lo, hi = int((0.4 - 0.07) * sr), int((0.6 + 0.07) * sr)
    assert np.array_equal(out[:lo], orig[:lo]) and np.array_equal(out[hi:], orig[hi:])
    assert np.allclose(out[int(0.45 * sr):int(0.55 * sr)], 0.0)  # fully vocoded core


def test_span_mask_ramps_to_zero_at_edges():
    m = world_core.span_mask(200, [(0.2, 0.6)], ramp_s=0.03)     # frames of 5 ms
    assert m[40] == 0.0 and m[80] == 1.0 and 0 < m[42] < 1
    assert m[119] < 0.2 and m[130] == 0.0


def test_merge_spans():
    assert world_core.merge_spans([(0.5, 0.7), (0.1, 0.2), (0.21, 0.3)]) == \
        [(0.1, 0.3), (0.5, 0.7)]


def test_seed_is_stable():
    assert world_core.seed_for("0003", "Vibrato_Group", 1) == \
        world_core.seed_for("0003", "Vibrato_Group", 1)


# ── Sampled parameters (world_v2.sample_presets) ─────────────────────────────
from alt.augment import world_v2  # noqa: E402
from alt.augment.features import QGRID  # noqa: E402


def test_strength_quantiles_are_ordered_and_in_range():
    rng = np.random.default_rng(0)
    for gamma in (1.0, 0.6):
        u = world_v2.strength_quantiles(3, rng, gamma)
        assert np.all(np.diff(u) > 0)
        assert u.min() >= world_v2.U_RANGE[0] and u.max() <= world_v2.U_RANGE[1]


def test_strength_quantiles_reproduce_and_favour_tails():
    # gamma = 1 → uniform in quantile space; gamma < 1 → more mass in the tails.
    rng = np.random.default_rng(1)
    u1 = np.concatenate([world_v2.strength_quantiles(3, rng, 1.0) for _ in range(4000)])
    u6 = np.concatenate([world_v2.strength_quantiles(3, rng, 0.6) for _ in range(4000)])
    tail = lambda u: np.mean((u < 0.1) | (u > 0.9))  # noqa: E731
    # Uniform on [0.02, 0.98]: P(u < 0.1 or u > 0.9) = 0.16 / 0.96.
    assert abs(tail(u1) - 0.16 / 0.96) < 0.02
    assert tail(u6) > tail(u1) + 0.05


def test_quantile_value_interpolates_grid_and_orders_by_magnitude():
    grid = list(np.linspace(-200, 0, len(QGRID)))           # negative effect
    spec = {"q25": grid[6], "q50": grid[11], "q75": grid[16], "qgrid": grid}
    weak, strong = world_v2.quantile_value(spec, 0.1), world_v2.quantile_value(spec, 0.9)
    assert abs(strong) > abs(weak)                           # mirrored for q50 < 0
    old = {"q25": 1.0, "q50": 2.0, "q75": 3.0}               # no grid → quartiles
    assert world_v2.quantile_value(old, 0.5) == 2.0
    assert world_v2.quantile_value(None, 0.5) == 0.0


def test_sample_presets_deterministic_per_clip():
    a = world_v2.sample_presets("Vibrato_Group", "english", "clipA")
    b = world_v2.sample_presets("Vibrato_Group", "english", "clipA")
    c = world_v2.sample_presets("Vibrato_Group", "english", "clipB")
    assert a == b and a != c
    assert [p["extent_cents"] for p in a] == sorted(p["extent_cents"] for p in a)
