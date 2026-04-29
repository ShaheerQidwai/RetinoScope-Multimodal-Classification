"""
SHAP-based interpretability for RetinoScope.

Covers:
  1) XGBoost Fundus model    — TreeExplainer (fast, exact)
  2) XGBoost OCT model       — TreeExplainer
  3) Attention-fusion network — GradientExplainer (torch DL)

For each model the script saves under ``shap_outputs/``:
  - <model>_summary.png         beeswarm of top-K features across all classes
  - <model>_per_class_top.png   bar chart of mean |SHAP| top-K features for each class
  - <model>_top_features.json   machine-readable top-K features per class
  - <model>_meanabs.npy         mean |SHAP| matrix (n_classes, n_features) for downstream use

Usage:
  python shap_analysis.py --model all
  python shap_analysis.py --model xgb_fundus --top_k 15
  python shap_analysis.py --model attn_fusion --n_background 200 --n_explain 500

Notes:
  - SHAP is required: ``pip install shap``.
  - For the attention model we report SHAP wrt the *logit* of each class — a
    standard choice that captures both the gating and the MLP head jointly.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; avoids Tk image buffer errors on Windows
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "shap is required for this script. Install with: pip install shap"
    ) from exc


# ---------------------------------------------------------------------------
# Defaults — match latefusion / retina_attention_fusion paths
# ---------------------------------------------------------------------------

FUNDUS_MODEL_PATH = "models/best_model.pkl"
OCT_MODEL_PATH = "models/oct_best_model.pkl"
DL_MODEL_PATH = "models/retina_attention_fusion.pt"

FUNDUS_FEATURES_PATH = "features/test_features.npz"
FUNDUS_LABELS_PATH = "features/test_features_labels.npy"
OCT_FEATURES_PATH = "features/oct_test_features.npz"
OCT_LABELS_PATH = "features/oct_test_features_labels.npy"

FUNDUS_TRAIN_FEATURES = "features/train_features.npz"
OCT_TRAIN_FEATURES = "features/oct_train_features.npz"

OUT_DIR = "shap_outputs"

CLASS_NAMES = [
    "Normal",
    "Class 1",
    "Class 3",
    "Class 4",
    "Class 6",
    "Class 7",
    "Class 8",
]

LABEL_MAP_9_TO_7 = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6}


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def _load_npz(path: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    data = np.load(path, allow_pickle=True)
    X = (
        data["features"]
        if "features" in data
        else (data["arr_0"] if "arr_0" in data else data[list(data.keys())[0]])
    )
    names = data["feature_names"].tolist() if "feature_names" in data else None
    return X, names


def _map_labels(y_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isin(y_raw, list(LABEL_MAP_9_TO_7.keys()))
    y = np.vectorize(LABEL_MAP_9_TO_7.get)(y_raw[valid])
    return y, valid


def _names_from_selector(model_pkg: dict, fallback_n: int, all_names: Optional[List[str]]) -> List[str]:
    """Best-effort: recover the *post-selection* feature names for an XGBoost model."""
    selector = model_pkg.get("feature_selector") if isinstance(model_pkg, dict) else None
    if selector is not None and hasattr(selector, "selected_features"):
        idx = list(selector.selected_features)
        if all_names is not None and len(all_names) > max(idx):
            return [all_names[i] for i in idx]
        return [f"f{i}" for i in idx]
    if all_names is not None and len(all_names) == fallback_n:
        return all_names
    return [f"f{i}" for i in range(fallback_n)]


def _apply_selector(X: np.ndarray, model_pkg: dict) -> np.ndarray:
    selector = model_pkg.get("feature_selector") if isinstance(model_pkg, dict) else None
    if selector is None:
        return X
    return selector.transform(X)


# ---------------------------------------------------------------------------
# SHAP plotting helpers
# ---------------------------------------------------------------------------


def _normalise_shap_to_cnf(
    sv: Any,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """
    Normalise SHAP output to canonical shape ``(C, N, F)``.

    SHAP versions differ:
      - older multiclass: list of length C, each (N, F)
      - some versions:    ndarray (C, N, F)
      - newer:            ndarray (N, F, C)
      - or:               ndarray (N, C, F)
      - binary:           ndarray (N, F)

    This function inspects the shape and produces ``(C, N, F)`` regardless.
    """
    if isinstance(sv, list):
        return np.stack(sv, axis=0)  # already (C, N, F)
    sv = np.asarray(sv)
    if sv.ndim == 2:
        return sv[None, :, :]  # binary -> (1, N, F)
    if sv.ndim != 3:
        raise ValueError(f"Unexpected SHAP output ndim={sv.ndim}, shape={sv.shape}")

    dims = list(sv.shape)
    # Find which axis holds N and which holds F
    n_pos = next((i for i, d in enumerate(dims) if d == n_samples), -1)
    f_pos = next((i for i, d in enumerate(dims) if d == n_features and i != n_pos), -1)
    if n_pos < 0 or f_pos < 0 or n_pos == f_pos:
        raise ValueError(
            f"Could not identify N={n_samples} / F={n_features} axes in SHAP shape {sv.shape}"
        )
    c_pos = ({0, 1, 2} - {n_pos, f_pos}).pop()
    sv = np.transpose(sv, (c_pos, n_pos, f_pos))
    return sv


def _plot_per_class_topk(
    mean_abs: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    top_k: int,
    title: str,
    out_path: str,
) -> Dict[str, List[Dict[str, float]]]:
    """One subplot per class with the top-K features by mean |SHAP|. Also returns JSON-friendly top-k."""
    assert mean_abs.ndim == 2 and mean_abs.shape[0] == len(class_names)
    n_classes = mean_abs.shape[0]

    cols = 3 if n_classes >= 3 else n_classes
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    out_summary: Dict[str, List[Dict[str, float]]] = {}
    for ci, cname in enumerate(class_names):
        ax = axes[ci]
        order = np.argsort(mean_abs[ci])[-top_k:][::-1]
        labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in order]
        values = mean_abs[ci, order]
        ax.barh(range(top_k), values[::-1], color="#3b82c4")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(labels[::-1], fontsize=8)
        ax.set_title(cname, fontsize=10)
        ax.set_xlabel("mean |SHAP|", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        out_summary[cname] = [
            {"feature": labels[r], "mean_abs_shap": float(values[r])}
            for r in range(top_k)
        ]

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_summary


def _save_beeswarm(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    out_path: str,
    top_k: int,
    title: str,
) -> None:
    """A single beeswarm aggregating across classes (mean |SHAP| over class axis)."""
    if shap_values.ndim == 3:
        agg = np.mean(np.abs(shap_values), axis=0)  # (N, F) — class-averaged magnitudes
    else:
        agg = np.abs(shap_values)
    # Use global feature ordering by mean |SHAP|
    feat_order = np.argsort(agg.mean(axis=0))[-top_k:]
    feat_names_top = [feature_names[i] for i in feat_order]

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        agg[:, feat_order],
        X_explain[:, feat_order],
        feature_names=feat_names_top,
        plot_type="dot",
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 1) XGBoost (Fundus / OCT) — TreeExplainer
# ---------------------------------------------------------------------------


def explain_xgboost(
    model_path: str,
    test_features_path: str,
    test_labels_path: str,
    train_features_path: str,
    out_prefix: str,
    label: str,
    top_k: int = 15,
) -> None:
    print(f"\n[{label}] Loading model from {model_path}")
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model = pkg["model"] if isinstance(pkg, dict) and "model" in pkg else pkg

    X_raw, _ = _load_npz(test_features_path)
    y_raw = np.load(test_labels_path)
    y, mask = _map_labels(y_raw)
    X_raw = X_raw[mask]
    print(f"  raw test: X={X_raw.shape}, y_classes={np.unique(y)}")

    # Feature names (pre-selection)
    _, all_names = _load_npz(train_features_path)
    X_sel = _apply_selector(X_raw, pkg)
    sel_names = _names_from_selector(pkg, X_sel.shape[1], all_names)
    print(f"  post-selection: X={X_sel.shape}, |names|={len(sel_names)}")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sel)
    sv = _normalise_shap_to_cnf(sv, X_sel.shape[0], X_sel.shape[1])

    # mean |SHAP| per (class, feature)
    mean_abs = np.abs(sv).mean(axis=1)  # (C, F)
    np.save(f"{out_prefix}_meanabs.npy", mean_abs)

    n_classes = mean_abs.shape[0]
    cls_names = CLASS_NAMES[:n_classes] if n_classes <= len(CLASS_NAMES) else [
        f"Class {i}" for i in range(n_classes)
    ]
    summary = _plot_per_class_topk(
        mean_abs,
        sel_names,
        cls_names,
        top_k,
        f"Top-{top_k} features per class (mean |SHAP|) — {label}",
        f"{out_prefix}_per_class_top.png",
    )
    with open(f"{out_prefix}_top_features.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Beeswarm aggregated across classes
    _save_beeswarm(
        sv,
        X_sel,
        sel_names,
        f"{out_prefix}_summary.png",
        top_k,
        f"{label} — class-averaged SHAP beeswarm (top {top_k})",
    )
    print(f"  saved: {out_prefix}_summary.png, _per_class_top.png, _top_features.json, _meanabs.npy")


# ---------------------------------------------------------------------------
# 2) Attention fusion network — GradientExplainer
# ---------------------------------------------------------------------------


def explain_attention_fusion(
    model_path: str,
    out_prefix: str,
    top_k: int = 15,
    n_background: int = 200,
    n_explain: int = 500,
) -> None:
    """SHAP for the trained RetinoScopeFusionNetwork using torch GradientExplainer."""
    import torch
    from retina_attention_fusion import (
        RetinoScopeFusionNetwork,
        _load_paired_modalities,
        load_checkpoint,
    )

    print(f"\n[Attention Fusion] Loading checkpoint {model_path}")
    ckpt = load_checkpoint(model_path, map_location="cpu")
    cfg = ckpt["config"]
    fundus_dim = int(getattr(cfg, "fundus_dim", 139))
    oct_dim = int(getattr(cfg, "oct_dim", 139))
    num_classes = int(getattr(cfg, "num_classes", 7))
    d_in = fundus_dim + oct_dim
    print(f"  input_dim={d_in} (fundus {fundus_dim} + OCT {oct_dim}), classes={num_classes}")

    # Reconstruct architecture exactly as in run_training
    attn_hidden = max(64, min(256, d_in // 2))
    mlp_hidden1 = max(64, min(128, d_in // 2))
    model = RetinoScopeFusionNetwork(
        input_dim=d_in,
        num_classes=num_classes,
        attn_hidden=attn_hidden,
        mlp_hidden1=mlp_hidden1,
        mlp_hidden2=int(getattr(cfg, "hidden2", 32)),
        dropout=float(getattr(cfg, "dropout", 0.35)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Train/test data — train as background, test as explain set
    print("  loading train (background) and test (explain) feature pairs")
    X_train, _, fd_tr, od_tr = _load_paired_modalities(
        "features/train_features.npz",
        "features/train_features_labels.npy",
        "features/oct_train_features.npz",
        "features/oct_train_features_labels.npy",
        "train",
    )
    X_test, y_test, fd_te, od_te = _load_paired_modalities(
        FUNDUS_FEATURES_PATH,
        FUNDUS_LABELS_PATH,
        OCT_FEATURES_PATH,
        OCT_LABELS_PATH,
        "test",
    )
    if (fd_tr, od_tr) != (fundus_dim, oct_dim) or X_test.shape[1] != d_in:
        raise RuntimeError(
            f"Feature widths inconsistent with checkpoint: train={fd_tr}+{od_tr}, "
            f"test cols={X_test.shape[1]}, expected {d_in}"
        )

    # Apply checkpoint's StandardScaler if present
    if "scaler_mean" in ckpt and "scaler_scale" in ckpt:
        mean_ = ckpt["scaler_mean"].astype(np.float32)
        scale_ = ckpt["scaler_scale"].astype(np.float32)
        X_train = (X_train.astype(np.float32) - mean_) / scale_
        X_test = (X_test.astype(np.float32) - mean_) / scale_
        print("  applied saved StandardScaler (train-time fit)")

    # Subsample background and explain sets for speed
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex_idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    X_bg = torch.tensor(X_train[bg_idx], dtype=torch.float32)
    X_ex = torch.tensor(X_test[ex_idx], dtype=torch.float32)

    # SHAP GradientExplainer expects a model that returns logits (not a tuple)
    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            logits, _attn = self.inner(x)
            return logits

    wrapper = LogitsOnly(model)
    print(f"  GradientExplainer: |bg|={len(X_bg)}, |explain|={len(X_ex)}")
    explainer = shap.GradientExplainer(wrapper, X_bg)
    sv = explainer.shap_values(X_ex)
    sv = _normalise_shap_to_cnf(sv, X_ex.shape[0], d_in)

    mean_abs = np.abs(sv).mean(axis=1)  # (C, F)
    np.save(f"{out_prefix}_meanabs.npy", mean_abs)

    feature_names = (
        [f"Fundus[{i}]" for i in range(fundus_dim)]
        + [f"OCT[{i}]" for i in range(oct_dim)]
    )
    cls_names = CLASS_NAMES[:num_classes]

    summary = _plot_per_class_topk(
        mean_abs,
        feature_names,
        cls_names,
        top_k,
        f"Attention-Fusion: top-{top_k} features per class (mean |SHAP|)",
        f"{out_prefix}_per_class_top.png",
    )
    with open(f"{out_prefix}_top_features.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Beeswarm
    _save_beeswarm(
        sv,
        X_ex.detach().cpu().numpy(),
        feature_names,
        f"{out_prefix}_summary.png",
        top_k,
        f"Attention-Fusion — class-averaged SHAP beeswarm (top {top_k})",
    )

    # Modality-level mean |SHAP| (Fundus block vs OCT block) for the report
    fundus_mass = mean_abs[:, :fundus_dim].sum(axis=1)
    oct_mass = mean_abs[:, fundus_dim:].sum(axis=1)
    total = fundus_mass + oct_mass
    share = np.stack([fundus_mass / np.maximum(total, 1e-9), oct_mass / np.maximum(total, 1e-9)], axis=1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(num_classes)
    ax.bar(idx, share[:, 0], label="Fundus block", color="#2ecc71")
    ax.bar(idx, share[:, 1], bottom=share[:, 0], label="OCT block", color="#9b59b6")
    ax.set_xticks(idx)
    ax.set_xticklabels(cls_names, rotation=20)
    ax.set_ylabel("Share of total |SHAP| (per class)")
    ax.set_title("Per-class modality contribution (SHAP)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_modality_share.png", dpi=150)
    plt.close(fig)

    with open(f"{out_prefix}_modality_share.json", "w") as f:
        json.dump(
            {
                cls_names[i]: {
                    "fundus_share": float(share[i, 0]),
                    "oct_share": float(share[i, 1]),
                }
                for i in range(num_classes)
            },
            f,
            indent=2,
        )
    print(
        f"  saved: {out_prefix}_summary.png, _per_class_top.png, "
        f"_modality_share.png, _top_features.json, _modality_share.json, _meanabs.npy"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="SHAP interpretability for RetinoScope models")
    p.add_argument(
        "--model",
        choices=["xgb_fundus", "xgb_oct", "attn_fusion", "all"],
        default="all",
    )
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument(
        "--n_background",
        type=int,
        default=200,
        help="Background set size for the DL GradientExplainer",
    )
    p.add_argument(
        "--n_explain",
        type=int,
        default=500,
        help="Number of test samples to explain for the DL model",
    )
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model in ("xgb_fundus", "all"):
        explain_xgboost(
            FUNDUS_MODEL_PATH,
            FUNDUS_FEATURES_PATH,
            FUNDUS_LABELS_PATH,
            FUNDUS_TRAIN_FEATURES,
            os.path.join(args.out_dir, "xgb_fundus"),
            "XGBoost-Fundus",
            top_k=args.top_k,
        )
    if args.model in ("xgb_oct", "all"):
        explain_xgboost(
            OCT_MODEL_PATH,
            OCT_FEATURES_PATH,
            OCT_LABELS_PATH,
            OCT_TRAIN_FEATURES,
            os.path.join(args.out_dir, "xgb_oct"),
            "XGBoost-OCT",
            top_k=args.top_k,
        )
    if args.model in ("attn_fusion", "all"):
        explain_attention_fusion(
            DL_MODEL_PATH,
            os.path.join(args.out_dir, "attn_fusion"),
            top_k=args.top_k,
            n_background=args.n_background,
            n_explain=args.n_explain,
        )

    print(f"\nAll outputs written under: {args.out_dir}/")


if __name__ == "__main__":
    main()
