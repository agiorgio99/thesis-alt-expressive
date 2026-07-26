#!/usr/bin/env python3
"""
clap_analysis.py — does WORLD augmentation land in the right place in CLAP space?

Run ``clap_embed.py`` first; this script consumes ``embeddings.npz`` +
``metadata.csv`` and answers seven questions, each with a number you can put
in the thesis:

  1. GEOMETRY     Where do the three populations sit? Centroid cosine
                  similarities between real-technique, augmented-technique and
                  the neutral Control_Group they were synthesised from.
  2. DIRECTION    Augmentation is a *displacement* from control. Does it point
                  the same way as the real technique?
                     dir_score = cos(aug_c - ctrl_c, real_c - ctrl_c)
                     magnitude = ||aug_c - ctrl_c|| / ||real_c - ctrl_c||
                  dir_score near 1 = right direction; magnitude near 1 = right
                  amount. This is the single most diagnostic number here.
  3. DISTRIBUTION Fréchet CLAP Distance (the FAD recipe applied to CLAP
                  features) and kernel MMD, real vs aug per technique, with
                  real-vs-control as the "no augmentation at all" reference.
  4. SEPARABILITY Can a classifier tell real from augmented? Singer-disjoint
                  CV ROC-AUC. 0.5 = indistinguishable, 1.0 = artefact cluster.
  5. TRANSFER     Train a technique classifier on REAL audio, test on
                  AUGMENTED (and vice versa). This is the CLAP-space analogue
                  of your Phase-3 fine-tuning experiment.
  6. RETRIEVAL    For each augmented clip, how many of its k nearest REAL
                  neighbours carry the same technique label (kNN purity).
  7. ZERO-SHOT    CLAP is a joint audio/text space: score every clip against
                  natural-language technique prompts ("a singer using
                  vibrato") and compare real vs augmented accuracy.

Plus paired analysis (each augmented clip against the exact control clip it
was synthesised from), 2-D projections, and an optional join against the
per-technique WER tables so you can correlate acoustic distance with ASR gain.

Usage
─────
    python scripts/clap_analysis.py --emb-dir results/clap
    python scripts/clap_analysis.py --emb-dir results/clap --no-zero-shot
    python scripts/clap_analysis.py --emb-dir results/clap --projection umap

Outputs land in ``<emb-dir>/analysis/``: one CSV per section, the figures, and
CLAP_ANALYSIS.md tying it together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (confusion_matrix, roc_auc_score,
                                 balanced_accuracy_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    _SKLEARN = True
except ImportError:                                      # pragma: no cover
    _SKLEARN = False
    print("[warn] scikit-learn not installed — sections 4/5 and the 2-D "
          "projection will be skipped. `pip install scikit-learn`")

from scipy import linalg                                  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Text prompts for the zero-shot probe (section 7)
# ─────────────────────────────────────────────────────────────────────────────
PROMPTS: dict[str, list[str]] = {
    "vibrato": [
        "a singer singing with vibrato",
        "a vocal performance with a wavering oscillating pitch",
        "singing with a pulsating tremulous tone",
    ],
    "breathy": [
        "a breathy singing voice",
        "singing with a soft airy whispery tone",
        "a voice with a lot of breath noise while singing",
    ],
    "glissando": [
        "a singer sliding smoothly between pitches, a glissando",
        "portamento singing, gliding from one note to another",
        "a continuous pitch slide in a sung phrase",
    ],
    "pharyngeal": [
        "a pressed tense pharyngeal singing voice",
        "a nasal twangy constricted vocal tone",
        "a strained bright singing timbre",
    ],
    "mixed_falsetto": [
        "a singer in falsetto, light head voice",
        "a high light flute-like head register",
        "a mixed voice singing high and softly",
    ],
    "control": [
        "a plain neutral singing voice with no special technique",
        "an ordinary straight-tone singing performance",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Distance / divergence helpers
# ─────────────────────────────────────────────────────────────────────────────
def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians fitted to two embedding sets.

    This is the FAD (Fréchet Audio Distance) recipe applied to CLAP features:
    ``||mu_x - mu_y||^2 + Tr(S_x + S_y - 2 (S_x S_y)^(1/2))``. Lower is more
    similar. With 512 dimensions and only a few hundred samples the covariance
    is rank-deficient, so a small ridge ``eps * I`` is added for stability.

    Args:
        x:   ``(N, D)`` embeddings of population A.
        y:   ``(M, D)`` embeddings of population B.
        eps: Ridge added to both covariances.

    Returns:
        The Fréchet distance, or ``nan`` if either set has fewer than 2 rows.
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mu_x, mu_y = x.mean(0), y.mean(0)
    s_x = np.cov(x, rowvar=False) + eps * np.eye(x.shape[1])
    s_y = np.cov(y, rowvar=False) + eps * np.eye(y.shape[1])
    diff = mu_x - mu_y
    covmean, _ = linalg.sqrtm(s_x @ s_y, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s_x) + np.trace(s_y) - 2.0 * np.trace(covmean))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None,
            n_perm: int = 200, max_n: int = 400,
            seed: int = 0) -> tuple[float, float]:
    """Unbiased squared MMD with an RBF kernel, plus a permutation p-value.

    A distribution-free companion to the Fréchet distance: it makes no
    Gaussian assumption, which matters because CLAP features are not Gaussian.
    The raw MMD^2 value is not interpretable on its own (the unbiased
    estimator can go slightly negative), so a label-permutation test is run to
    say whether the two populations are distinguishable *at all*.

    Both sets are subsampled to the same size so the estimate is not biased by
    the 3:1 augmented-to-real imbalance.

    Args:
        x:      ``(N, D)`` embeddings of population A.
        y:      ``(M, D)`` embeddings of population B.
        gamma:  RBF bandwidth. ``None`` uses the median-distance heuristic.
        n_perm: Permutations for the p-value (0 disables it).
        max_n:  Cap on the per-population sample size.
        seed:   RNG seed for the subsampling and permutations.

    Returns:
        ``(mmd2, p_value)``. A small p means the two populations are
        statistically distinguishable in CLAP space.
    """
    if len(x) < 4 or len(y) < 4:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = min(len(x), len(y), max_n)
    x = x[rng.choice(len(x), n, replace=False)]
    y = y[rng.choice(len(y), n, replace=False)]

    z = np.vstack([x, y])
    d2 = (np.sum(z ** 2, 1)[:, None] + np.sum(z ** 2, 1)[None, :] - 2.0 * z @ z.T)
    np.maximum(d2, 0.0, out=d2)
    if gamma is None:
        med = np.median(d2[np.triu_indices_from(d2, k=1)])
        gamma = 1.0 / max(med, 1e-8)
    k = np.exp(-gamma * d2)
    np.fill_diagonal(k, 0.0)

    def _stat(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        kaa = k[np.ix_(idx_a, idx_a)].sum() / (n * (n - 1))
        kbb = k[np.ix_(idx_b, idx_b)].sum() / (n * (n - 1))
        kab = k[np.ix_(idx_a, idx_b)].mean()
        return float(kaa + kbb - 2.0 * kab)

    a0, b0 = np.arange(n), np.arange(n, 2 * n)
    observed = _stat(a0, b0)

    if n_perm <= 0:
        return observed, float("nan")
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(2 * n)
        null[i] = _stat(perm[:n], perm[n:])
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return observed, p


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))


def unit(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit L2 norm."""
    return v / max(np.linalg.norm(v), 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  Section 1 + 2 — centroid geometry and displacement direction
# ─────────────────────────────────────────────────────────────────────────────
def centroid_geometry(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Compare real / augmented / control centroids for every technique.

    Args:
        df:  Metadata table (needs ``origin``, ``group``, ``technique``).
        emb: ``(N, D)`` embedding matrix aligned with ``df``.

    Returns:
        One row per technique with centroid cosine similarities, the
        displacement direction score and the displacement magnitude ratio.
    """
    rows = []
    for tech in sorted(t for t in df["technique"].unique() if t != "none"):
        sel = df["technique"] == tech
        i_real = np.where(sel & (df["origin"] == "real") & (df["group"] == "technique"))[0]
        i_aug = np.where(sel & (df["origin"] == "aug"))[0]
        i_ctrl = np.where(sel & (df["group"] == "control"))[0]
        if len(i_real) == 0 or len(i_aug) == 0 or len(i_ctrl) == 0:
            print(f"[geom] skipping {tech}: real={len(i_real)} aug={len(i_aug)} "
                  f"ctrl={len(i_ctrl)}")
            continue

        c_real, c_aug, c_ctrl = (unit(emb[i].mean(0)) for i in (i_real, i_aug, i_ctrl))
        d_real, d_aug = c_real - c_ctrl, c_aug - c_ctrl

        rows.append({
            "technique": tech,
            "n_real": len(i_real), "n_aug": len(i_aug), "n_control": len(i_ctrl),
            # Section 1 — absolute positions
            "cos_real_aug": cos(c_real, c_aug),
            "cos_real_control": cos(c_real, c_ctrl),
            "cos_aug_control": cos(c_aug, c_ctrl),
            # Section 2 — the displacement induced by augmentation
            "direction_score": cos(d_aug, d_real),
            "magnitude_ratio": float(np.linalg.norm(d_aug)
                                     / max(np.linalg.norm(d_real), 1e-8)),
            # Within-population spread, for context on the distances above
            "spread_real": float(np.mean([1 - cos(v, c_real) for v in emb[i_real]])),
            "spread_aug": float(np.mean([1 - cos(v, c_aug) for v in emb[i_aug]])),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Section 3 — distribution-level distances
# ─────────────────────────────────────────────────────────────────────────────
def distribution_distances(df: pd.DataFrame, emb: np.ndarray,
                           pca_dim: int | None = 64) -> pd.DataFrame:
    """Fréchet distance and MMD between real / augmented / control populations.

    Args:
        df:      Metadata table.
        emb:     ``(N, D)`` embedding matrix.
        pca_dim: If set, PCA-reduce before the Fréchet computation so the
                 covariance estimate is well-conditioned. MMD always uses the
                 full-dimensional embeddings.

    Returns:
        One row per technique with ``fd_real_aug``, ``fd_real_control``,
        ``mmd_real_aug``, ``mmd_real_control`` and the ratio of the two
        Fréchet distances.
    """
    x = emb
    if pca_dim and _SKLEARN and emb.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=0).fit_transform(emb)

    rows = []
    for tech in sorted(t for t in df["technique"].unique() if t != "none"):
        sel = df["technique"] == tech
        i_real = np.where(sel & (df["origin"] == "real") & (df["group"] == "technique"))[0]
        i_aug = np.where(sel & (df["origin"] == "aug"))[0]
        i_ctrl = np.where(sel & (df["group"] == "control"))[0]
        if min(len(i_real), len(i_aug), len(i_ctrl)) < 2:
            continue
        fd_ra = frechet_distance(x[i_real], x[i_aug])
        fd_rc = frechet_distance(x[i_real], x[i_ctrl])
        mmd_ra, p_ra = mmd_rbf(emb[i_real], emb[i_aug])
        mmd_rc, p_rc = mmd_rbf(emb[i_real], emb[i_ctrl])
        rows.append({
            "technique": tech,
            "fd_real_aug": fd_ra,
            "fd_real_control": fd_rc,
            # < 1 means the augmented clips are closer to the real technique
            # than the untouched control clips are — augmentation helped.
            "fd_ratio_aug_over_control": fd_ra / max(fd_rc, 1e-8),
            "mmd_real_aug": mmd_ra, "mmd_p_real_aug": p_ra,
            "mmd_real_control": mmd_rc, "mmd_p_real_control": p_rc,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Section 4 — real vs augmented separability
# ─────────────────────────────────────────────────────────────────────────────
def separability(df: pd.DataFrame, emb: np.ndarray, n_splits: int = 3) -> pd.DataFrame:
    """Singer-disjoint ROC-AUC for a real-vs-augmented classifier.

    Interpretation: AUC ~0.5 means CLAP cannot tell the WORLD-resynthesised
    clips from real recordings (good for the augmentation claim); AUC ~1.0
    means the augmented clips form a systematically distinguishable cluster —
    which does not by itself make them useless, but it does mean the model is
    learning a domain, not a technique.

    Args:
        df:       Metadata table.
        emb:      ``(N, D)`` embedding matrix.
        n_splits: Number of singer-disjoint CV folds.

    Returns:
        One row per technique plus an ``ALL`` row, with mean/std ROC-AUC.
    """
    if not _SKLEARN:
        return pd.DataFrame()

    def _auc(idx: np.ndarray) -> tuple[float, float]:
        y = (df["origin"].to_numpy()[idx] == "aug").astype(int)
        g = df["singer_id"].to_numpy()[idx]
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        k = min(n_splits, len(np.unique(g)))
        if k < 2:
            return float("nan"), float("nan")
        x = StandardScaler().fit_transform(emb[idx])
        scores = []
        for tr, te in GroupKFold(n_splits=k).split(x, y, groups=g):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(x[tr], y[tr])
            scores.append(roc_auc_score(y[te], clf.predict_proba(x[te])[:, 1]))
        if not scores:
            return float("nan"), float("nan")
        return float(np.mean(scores)), float(np.std(scores))

    tech_mask = df["group"] == "technique"
    rows = []
    for tech in sorted(t for t in df["technique"].unique() if t != "none"):
        idx = np.where(tech_mask & (df["technique"] == tech))[0]
        m, s = _auc(idx)
        rows.append({"technique": tech, "n": len(idx), "auc_mean": m, "auc_std": s})
    m, s = _auc(np.where(tech_mask)[0])
    rows.append({"technique": "ALL", "n": int(tech_mask.sum()),
                 "auc_mean": m, "auc_std": s})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Section 5 — cross-domain technique probe
# ─────────────────────────────────────────────────────────────────────────────
def transfer_probe(df: pd.DataFrame, emb: np.ndarray,
                   out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Train a technique classifier on one domain and test it on the other.

    ``real -> aug`` answers: does a model that learned real vibrato/breathy/…
    recognise the synthesised versions? ``aug -> real`` is the CLAP-space
    analogue of fine-tuning Whisper on augmented data and evaluating on real
    singing — the core Phase-3 claim.

    Args:
        df:      Metadata table.
        emb:     ``(N, D)`` embedding matrix.
        out_dir: Folder for the two confusion-matrix PNGs.

    Returns:
        A summary DataFrame and a dict of per-direction confusion matrices.
    """
    if not _SKLEARN:
        return pd.DataFrame(), {}

    tech_mask = (df["group"] == "technique") & (df["technique"] != "none")
    i_real = np.where(tech_mask & (df["origin"] == "real"))[0]
    i_aug = np.where(tech_mask & (df["origin"] == "aug"))[0]
    labels = sorted(set(df["technique"].to_numpy()[i_real])
                    & set(df["technique"].to_numpy()[i_aug]))
    if len(labels) < 2:
        return pd.DataFrame(), {}

    scaler = StandardScaler().fit(emb[np.concatenate([i_real, i_aug])])
    rows, cms = [], {}

    for name, tr_idx, te_idx in (("real->aug", i_real, i_aug),
                                 ("aug->real", i_aug, i_real)):
        tr = tr_idx[np.isin(df["technique"].to_numpy()[tr_idx], labels)]
        te = te_idx[np.isin(df["technique"].to_numpy()[te_idx], labels)]
        y_tr = df["technique"].to_numpy()[tr]
        y_te = df["technique"].to_numpy()[te]

        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(scaler.transform(emb[tr]), y_tr)
        pred = clf.predict(scaler.transform(emb[te]))

        cm = confusion_matrix(y_te, pred, labels=labels, normalize="true")
        cms[name] = cm
        rows.append({
            "direction": name,
            "n_train": len(tr), "n_test": len(te),
            "accuracy": float((pred == y_te).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "chance": 1.0 / len(labels),
        })

        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Technique probe: train {name.split('->')[0]}, "
                     f"test {name.split('->')[1]}")
        for a in range(len(labels)):
            for b in range(len(labels)):
                ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center",
                        color="w" if cm[a, b] < 0.6 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / f"probe_confusion_{name.replace('->', '_to_')}.png",
                    dpi=160)
        plt.close(fig)

    return pd.DataFrame(rows), cms


# ─────────────────────────────────────────────────────────────────────────────
#  Section 6 — kNN retrieval purity
# ─────────────────────────────────────────────────────────────────────────────
def knn_purity(df: pd.DataFrame, emb: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Fraction of an augmented clip's k nearest REAL neighbours that match it.

    Because the embeddings are L2-normalised, the nearest neighbour under
    cosine similarity is just the largest dot product.

    Args:
        df:  Metadata table.
        emb: ``(N, D)`` embedding matrix.
        k:   Neighbourhood size.

    Returns:
        One row per technique with augmented-query purity and, as a reference
        ceiling, real-query purity (leave-one-out).
    """
    tech_mask = (df["group"] == "technique") & (df["technique"] != "none")
    i_real = np.where(tech_mask & (df["origin"] == "real"))[0]
    if len(i_real) <= k:
        return pd.DataFrame()

    ref = emb[i_real]
    ref_lab = df["technique"].to_numpy()[i_real]

    def _purity(query_idx: np.ndarray, drop_self: bool) -> float:
        if len(query_idx) == 0:
            return float("nan")
        sims = emb[query_idx] @ ref.T
        if drop_self:
            pos = {g: p for p, g in enumerate(i_real)}
            for r, g in enumerate(query_idx):
                if g in pos:
                    sims[r, pos[g]] = -np.inf
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        hit = ref_lab[nn] == df["technique"].to_numpy()[query_idx][:, None]
        return float(hit.mean())

    rows = []
    for tech in sorted(set(ref_lab)):
        q_aug = np.where(tech_mask & (df["origin"] == "aug")
                         & (df["technique"] == tech))[0]
        q_real = np.where(tech_mask & (df["origin"] == "real")
                          & (df["technique"] == tech))[0]
        rows.append({
            "technique": tech, "k": k, "n_aug_queries": len(q_aug),
            "purity_aug_query": _purity(q_aug, drop_self=False),
            "purity_real_query": _purity(q_real, drop_self=True),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Section 7 — zero-shot text probing
# ─────────────────────────────────────────────────────────────────────────────
def zero_shot(df: pd.DataFrame, emb: np.ndarray, model_name: str,
              device: str) -> pd.DataFrame:
    """Score every clip against natural-language technique prompts.

    Prompt embeddings for a class are averaged over several paraphrases and
    re-normalised, then every audio embedding is assigned the highest-scoring
    class. Reported per technique for real and augmented clips separately.

    Args:
        df:         Metadata table.
        emb:        ``(N, D)`` audio embedding matrix.
        model_name: HF CLAP checkpoint (must match the one used for audio).
        device:     "cuda" or "cpu".

    Returns:
        One row per (technique, origin) with zero-shot accuracy and the mean
        similarity to the correct prompt.
    """
    from clap_embed import ClapEmbedder                 # local import: heavy

    embedder = ClapEmbedder(model_name, device)
    classes = [c for c in PROMPTS if c in set(df["technique"]) or c == "control"]
    proto = np.stack([unit(embedder.embed_texts(PROMPTS[c]).mean(0))
                      for c in classes])                # (C, D)

    sims = emb @ proto.T                                # (N, C)
    pred = np.array(classes)[sims.argmax(1)]

    rows = []
    for tech in sorted(t for t in df["technique"].unique() if t in classes):
        ci = classes.index(tech)
        for origin in ("real", "aug"):
            idx = np.where((df["technique"] == tech) & (df["origin"] == origin)
                           & (df["group"] == "technique"))[0]
            if len(idx) == 0:
                continue
            rows.append({
                "technique": tech, "origin": origin, "n": len(idx),
                "zeroshot_accuracy": float((pred[idx] == tech).mean()),
                "mean_sim_correct_prompt": float(sims[idx, ci].mean()),
                "chance": 1.0 / len(classes),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Paired analysis — augmented clip vs the exact control clip it came from
# ─────────────────────────────────────────────────────────────────────────────
def paired_shift(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Per-pair displacement between an augmented clip and its source control.

    ``build_augmented_dataset.py`` writes ``{orig_stem}_v{k}.wav``, so an
    augmented clip is matched to its source by (singer, song, source_stem).
    This isolates the effect of the augmentation itself from any content
    difference between songs.

    Args:
        df:  Metadata table.
        emb: ``(N, D)`` embedding matrix.

    Returns:
        One row per matched pair with the cosine similarity to its source and
        the alignment of its individual displacement with the real-technique
        direction for that technique.
    """
    # NOTE: the technique MUST be part of the key. GTSinger stores a separate
    # Control_Group under every technique folder, and the stems restart at
    # 0001 in each — without the technique the keys collide and augmented
    # clips get paired with an unrelated control recording.
    ctrl = df[df["group"] == "control"]
    key_to_idx = {(r.singer_id, r.technique, r.song, r.source_stem): i
                  for i, r in zip(ctrl.index, ctrl.itertuples())}

    # Real-technique displacement direction, per technique (the target).
    targets: dict[str, np.ndarray] = {}
    for tech in df["technique"].unique():
        i_real = np.where((df["technique"] == tech) & (df["origin"] == "real")
                          & (df["group"] == "technique"))[0]
        i_ctrl = np.where((df["technique"] == tech) & (df["group"] == "control"))[0]
        if len(i_real) and len(i_ctrl):
            targets[tech] = unit(unit(emb[i_real].mean(0)) - unit(emb[i_ctrl].mean(0)))

    rows = []
    for i, r in zip(df.index, df.itertuples()):
        if r.origin != "aug":
            continue
        src = key_to_idx.get((r.singer_id, r.technique, r.song, r.source_stem))
        if src is None:
            continue
        d = emb[i] - emb[src]
        rows.append({
            "utt_id": r.utt_id, "technique": r.technique, "variant": r.variant,
            "singer_id": r.singer_id,
            "cos_to_source": cos(emb[i], emb[src]),
            "shift_norm": float(np.linalg.norm(d)),
            "shift_alignment": cos(d, targets[r.technique])
            if r.technique in targets else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────────────────────────────────────
def plot_projection(df: pd.DataFrame, emb: np.ndarray, out_path: Path,
                    method: str = "tsne", max_points: int = 4000) -> None:
    """Scatter the embeddings in 2-D, coloured by technique, shaped by origin.

    Args:
        df:         Metadata table.
        emb:        ``(N, D)`` embedding matrix.
        out_path:   Where to write the PNG.
        method:     "tsne", "pca" or "umap".
        max_points: Random subsample cap (t-SNE gets slow beyond a few k).
    """
    if not _SKLEARN:
        return
    rng = np.random.default_rng(0)
    idx = np.arange(len(df))
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)

    x = emb[idx]
    if method == "umap":
        try:
            import umap
            xy = umap.UMAP(n_neighbors=25, min_dist=0.1,
                           random_state=0).fit_transform(x)
        except ImportError:
            print("[warn] umap-learn not installed — falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        x50 = PCA(n_components=min(50, x.shape[1]), random_state=0).fit_transform(x)
        xy = TSNE(n_components=2, perplexity=30, init="pca",
                  random_state=0).fit_transform(x50)
    elif method == "pca":
        xy = PCA(n_components=2, random_state=0).fit_transform(x)

    sub = df.iloc[idx]
    techs = sorted(t for t in sub["technique"].unique())
    # Index tab10 directly — sampling it with linspace lands on the grey and
    # olive entries, which then clash with the grey used for the control group.
    colours = plt.cm.tab10(np.arange(len(techs)) % 10)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)

    # ── Left: technique clips only, real vs augmented ────────────────────────
    ax = axes[0]
    for ti, tech in enumerate(techs):
        m_aug = ((sub["technique"] == tech) & (sub["origin"] == "aug")
                 & (sub["group"] == "technique")).to_numpy()
        if m_aug.any():
            ax.scatter(xy[m_aug, 0], xy[m_aug, 1], s=13, marker="^", alpha=0.35,
                       color=colours[ti], linewidths=0, zorder=2,
                       label=f"{tech} (aug)")
        # Real clips are the minority class — draw them on top with an outline.
        m_real = ((sub["technique"] == tech) & (sub["origin"] == "real")
                  & (sub["group"] == "technique")).to_numpy()
        if m_real.any():
            ax.scatter(xy[m_real, 0], xy[m_real, 1], s=26, marker="o", alpha=0.9,
                       color=colours[ti], edgecolors="k", linewidths=0.35,
                       zorder=3, label=f"{tech} (real)")
    ax.set_title("Technique clips — real (circles, outlined) vs augmented "
                 "(triangles)", fontsize=10)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5),
              ncol=1, framealpha=0.9, markerscale=1.4)

    # ── Right: the same, with the neutral control clips underneath ───────────
    ax = axes[1]
    m_ctrl = (sub["group"] == "control").to_numpy()
    if m_ctrl.any():
        ax.scatter(xy[m_ctrl, 0], xy[m_ctrl, 1], s=16, marker="x", alpha=0.45,
                   color="0.45", linewidths=0.6, zorder=1, label="control")
    for ti, tech in enumerate(techs):
        m_aug = ((sub["technique"] == tech) & (sub["origin"] == "aug")).to_numpy()
        if m_aug.any():
            ax.scatter(xy[m_aug, 0], xy[m_aug, 1], s=11, marker="^", alpha=0.3,
                       color=colours[ti], linewidths=0, zorder=2)
        m_real = ((sub["technique"] == tech) & (sub["origin"] == "real")
                  & (sub["group"] == "technique")).to_numpy()
        if m_real.any():
            ax.scatter(xy[m_real, 0], xy[m_real, 1], s=22, marker="o", alpha=0.85,
                       color=colours[ti], edgecolors="k", linewidths=0.3, zorder=3)
    ax.set_title("Same view with the neutral Control_Group (grey ×) — the "
                 "audio augmentation started from", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"CLAP embedding space ({method.upper()})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_bars(geom: pd.DataFrame, dist: pd.DataFrame,
                       out_path: Path) -> None:
    """Bar charts of the direction score and the Fréchet-distance comparison.

    Args:
        geom:     Output of ``centroid_geometry``.
        dist:     Output of ``distribution_distances``.
        out_path: Where to write the PNG.
    """
    if geom.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    t = geom["technique"]
    xs = np.arange(len(t))

    axes[0].bar(xs, geom["direction_score"], color="#4c72b0")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("Direction score\ncos(aug−ctrl, real−ctrl)", fontsize=10)

    axes[1].bar(xs, geom["magnitude_ratio"], color="#dd8452")
    axes[1].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[1].set_title("Magnitude ratio\n‖aug−ctrl‖ / ‖real−ctrl‖", fontsize=10)

    if not dist.empty:
        d = dist.set_index("technique").reindex(t)
        w = 0.38
        axes[2].bar(xs - w / 2, d["fd_real_aug"], w, label="real vs aug",
                    color="#55a868")
        axes[2].bar(xs + w / 2, d["fd_real_control"], w, label="real vs control",
                    color="#c44e52")
        axes[2].legend(fontsize=8)
        axes[2].set_title("Fréchet CLAP distance\n(lower = more similar)",
                          fontsize=10)

    for ax in axes:
        ax.set_xticks(xs, t, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: correlate acoustic distance with the ASR results you already have
# ─────────────────────────────────────────────────────────────────────────────
def join_wer(dist: pd.DataFrame, repo: Path) -> pd.DataFrame:
    """Merge per-technique CLAP distances with the per-technique WER tables.

    Args:
        dist: Output of ``distribution_distances``.
        repo: Repository root (used to locate ``results/``).

    Returns:
        A merged table, or an empty DataFrame when the WER CSVs are missing.
    """
    base = repo / "results/baseline_english/asr_whisper_largev3_by_technique.csv"
    aug = repo / "results/augmented_eval/asr_whisper_largev3_by_technique.csv"
    if dist.empty or not base.exists() or not aug.exists():
        return pd.DataFrame()
    b = pd.read_csv(base)[["technique", "wer", "per"]].rename(
        columns={"wer": "wer_real", "per": "per_real"})
    a = pd.read_csv(aug)[["technique", "wer", "per"]].rename(
        columns={"wer": "wer_aug", "per": "per_aug"})
    out = dist.merge(b, on="technique", how="left").merge(a, on="technique", how="left")
    out["wer_gap_aug_minus_real"] = out["wer_aug"] - out["wer_real"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Report
# ─────────────────────────────────────────────────────────────────────────────
def write_report(out_dir: Path, tables: dict[str, pd.DataFrame],
                 meta: dict) -> None:
    """Write CLAP_ANALYSIS.md summarising every section.

    Args:
        out_dir: Analysis output folder.
        tables:  Section name -> DataFrame.
        meta:    Run metadata (counts, model name, flags).
    """
    L: list[str] = []
    L.append("# CLAP embedding analysis — augmented vs original\n")
    L.append(f"- Model: `{meta['model']}`")
    L.append(f"- Utterances embedded: {meta['n_total']} "
             f"(real {meta['n_real']}, augmented {meta['n_aug']}, "
             f"control {meta['n_control']})")
    L.append(f"- Embedding dim: {meta['dim']}\n")

    L.append("## How to read this\n")
    L.append("- **direction_score** — cosine between the displacement "
             "augmentation applies to a control clip and the displacement that "
             "separates real technique recordings from control. **1.0 = the "
             "augmentation pushes audio exactly the way the real technique "
             "does; 0 = orthogonal; negative = the wrong way.**")
    L.append("- **magnitude_ratio** — how far it pushes, relative to the real "
             "gap. <1 under-shoots (too subtle), >1 over-shoots (caricature).")
    L.append("- **fd_ratio_aug_over_control** — Fréchet(real, aug) / "
             "Fréchet(real, control). **<1 means augmentation moved the audio "
             "closer to real technique than doing nothing.**")
    L.append("- **auc_mean** — real-vs-augmented classifier. 0.5 = "
             "indistinguishable, 1.0 = the augmentation leaves an obvious "
             "domain signature.")
    L.append("- **aug->real accuracy** — a technique classifier trained only "
             "on synthetic audio, tested on real audio: the CLAP-space "
             "analogue of the Phase-3 fine-tuning result.")
    L.append("- **mmd_p_*** — label-permutation p-value for the MMD. A raw "
             "MMD^2 is not interpretable on its own (the unbiased estimator "
             "can go slightly negative when the populations overlap); read the "
             "p-value instead. Note that MMD is low-powered at these sample "
             "sizes in 512 dimensions — treat `auc_mean` as the more sensitive "
             "test and MMD as a distribution-free confirmation.\n")

    titles = {
        "geometry": "1–2. Centroid geometry and displacement direction",
        "distances": "3. Distribution distances (Fréchet / MMD)",
        "separability": "4. Real vs augmented separability (singer-disjoint AUC)",
        "probe": "5. Cross-domain technique probe",
        "knn": "6. kNN retrieval purity against real audio",
        "zeroshot": "7. Zero-shot text prompting",
        "paired_summary": "Paired analysis (augmented clip vs its source control)",
        "wer_join": "CLAP distance vs measured WER",
    }
    for key, title in titles.items():
        t = tables.get(key)
        if t is None or t.empty:
            continue
        L.append(f"## {title}\n")
        try:
            L.append(t.round(4).to_markdown(index=False))   # needs `tabulate`
        except ImportError:
            L.append("```\n" + t.round(4).to_string(index=False) + "\n```")
        L.append("")

    L.append("## Figures\n")
    for png in sorted(out_dir.glob("*.png")):
        L.append(f"- `{png.name}`")
    (out_dir / "CLAP_ANALYSIS.md").write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Analyse CLAP embeddings: augmented vs original singing.")
    p.add_argument("--emb-dir", default="results/clap",
                   help="Folder written by clap_embed.py")
    p.add_argument("--out", default=None,
                   help="Analysis output folder (default: <emb-dir>/analysis)")
    p.add_argument("--model", default="laion/larger_clap_music_and_speech",
                   help="CLAP checkpoint for the zero-shot text prompts "
                        "(must match the one used for the audio)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-zero-shot", action="store_true",
                   help="Skip section 7 (avoids reloading the CLAP model)")
    p.add_argument("--projection", default="tsne", choices=["tsne", "pca", "umap"])
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--pca-dim", type=int, default=64,
                   help="PCA dim before the Fréchet distance (0 = full dim)")
    return p.parse_args()


def main() -> None:
    """Run every analysis section and write the CSVs, figures and report."""
    args = parse_args()
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out) if args.out else emb_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(emb_dir / "embeddings.npz", allow_pickle=True)
    emb = npz["emb"].astype(np.float32)
    df = pd.read_csv(emb_dir / "metadata.csv")
    assert len(df) == len(emb), "metadata.csv and embeddings.npz are out of sync"
    # Guard: everything below assumes unit-norm rows.
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

    print(f"[load] {len(df)} utterances, dim={emb.shape[1]}")
    print(df.groupby(["origin", "group"]).size().to_string(), "\n")

    tables: dict[str, pd.DataFrame] = {}

    print("[1-2] centroid geometry + direction …")
    tables["geometry"] = centroid_geometry(df, emb)

    print("[3] distribution distances …")
    tables["distances"] = distribution_distances(
        df, emb, args.pca_dim if args.pca_dim > 0 else None)

    print("[4] real vs aug separability …")
    tables["separability"] = separability(df, emb)

    print("[5] cross-domain technique probe …")
    tables["probe"], _ = transfer_probe(df, emb, out_dir)

    print("[6] kNN purity …")
    tables["knn"] = knn_purity(df, emb, args.knn_k)

    if not args.no_zero_shot:
        print("[7] zero-shot text prompting …")
        try:
            tables["zeroshot"] = zero_shot(df, emb, args.model, args.device)
        except Exception as exc:                          # pragma: no cover
            print(f"[warn] zero-shot skipped: {exc}")
            tables["zeroshot"] = pd.DataFrame()

    print("[+] paired shift …")
    paired = paired_shift(df, emb)
    if not paired.empty:
        paired.to_csv(out_dir / "paired_shift.csv", index=False)
        tables["paired_summary"] = (
            paired.groupby("technique")
            .agg(n=("utt_id", "size"),
                 cos_to_source=("cos_to_source", "mean"),
                 shift_norm=("shift_norm", "mean"),
                 shift_alignment=("shift_alignment", "mean"))
            .reset_index()
        )

    tables["wer_join"] = join_wer(tables["distances"], REPO)

    print("[fig] plots …")
    plot_geometry_bars(tables["geometry"], tables["distances"],
                       out_dir / "geometry_bars.png")
    plot_projection(df, emb, out_dir / f"projection_{args.projection}.png",
                    args.projection)

    for name, t in tables.items():
        if t is not None and not t.empty:
            t.to_csv(out_dir / f"{name}.csv", index=False)

    meta = {
        "model": args.model,
        "n_total": len(df),
        "n_real": int((df["origin"] == "real").sum()),
        "n_aug": int((df["origin"] == "aug").sum()),
        "n_control": int((df["group"] == "control").sum()),
        "dim": int(emb.shape[1]),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    write_report(out_dir, tables, meta)

    print(f"\n=== analysis written to {out_dir} ===")
    if not tables["geometry"].empty:
        print("\nHeadline numbers:")
        print(tables["geometry"][["technique", "cos_real_aug", "direction_score",
                                  "magnitude_ratio"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
