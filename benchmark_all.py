"""
End-to-end benchmark: ML vs DL approaches, on the same paired test set.

Approaches compared (all evaluated against the same class-paired test set
produced by ``retina_attention_fusion._load_paired_modalities``):

  1) Fundus-only XGBoost              (decision from Fundus probabilities)
  2) OCT-only XGBoost                 (decision from OCT probabilities)
  3) Late fusion (weighted average)   — grid-searched fundus/OCT weights
  4) Late fusion (geometric / max)    — fixed-rule baselines
  5) Attention mid-level fusion (DL)  — saved checkpoint
  6) Hybrid Attention→XGBoost         — gate Fundus+OCT features with the DL
                                        attention head, then refit an XGBoost
                                        on the gated 278-D vector

Outputs (under ``benchmark_outputs/``):
  - benchmark_results.json   metrics for every approach (accuracy, balanced acc,
                             macro-F1, weighted-F1, per-class F1, support)
  - benchmark_results.csv    same data as a tidy table
  - benchmark_comparison.png grouped bar chart (acc / macro-F1 / balanced acc)
  - confusion_<approach>.png confusion matrix per approach

Run:  python benchmark_all.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; renders straight to file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

OUT_DIR = "benchmark_outputs"

CLASS_NAMES = [
    "Normal",
    "Class 1",
    "Class 3",
    "Class 4",
    "Class 6",
    "Class 7",
    "Class 8",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {
            int(c): float(s)
            for c, s in zip(
                np.unique(y_true),
                f1_score(y_true, y_pred, labels=np.unique(y_true), average=None, zero_division=0),
            )
        },
        "support": {int(c): int((y_true == c).sum()) for c in np.unique(y_true)},
    }


def _save_cm(y_true: np.ndarray, y_pred: np.ndarray, name: str, out_dir: str) -> None:
    n_cls = max(int(y_true.max()), int(y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    labels = CLASS_NAMES[:n_cls]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{name.replace(' ', '_').replace('/', '_')}.png"), dpi=150)
    plt.close()


def _grid_search_weights(
    proba_f: np.ndarray,
    proba_o: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.05,
) -> Tuple[float, float, float]:
    best_w, best_score = (0.5, 0.5), -1.0
    for w_f in np.arange(0.0, 1.0 + 1e-9, step):
        w_o = 1.0 - w_f
        p = w_f * proba_f + w_o * proba_o
        s = f1_score(y_true, np.argmax(p, axis=1), average="macro", zero_division=0)
        if s > best_score:
            best_score = float(s)
            best_w = (float(w_f), float(w_o))
    return best_w[0], best_w[1], best_score


# ---------------------------------------------------------------------------
# 1) Build paired test set (and aligned XGBoost-input views)
# ---------------------------------------------------------------------------


def _load_paired() -> Dict[str, Any]:
    """
    Load the *raw* (139+139)-D paired test set, plus the *post-selection* views
    that the XGBoost models expect (50-D each).

    The pairing is identical to ``latefusion.py`` and ``retina_attention_fusion.py``:
    by class label, first n indices per class.
    """
    from retina_attention_fusion import (
        load_npz_features,
        map_labels_9_to_7,
    )

    # Per-modality test features
    Xf_raw = load_npz_features("features/test_features.npz")
    yf_raw = np.load("features/test_features_labels.npy")
    Xo_raw = load_npz_features("features/oct_test_features.npz")
    yo_raw = np.load("features/oct_test_features_labels.npy")
    yf, mf = map_labels_9_to_7(yf_raw)
    yo, mo = map_labels_9_to_7(yo_raw)
    Xf_raw, Xo_raw = Xf_raw[mf], Xo_raw[mo]

    # Class-balanced pairing on RAW features (same as latefusion)
    f_indices, o_indices, y_paired = [], [], []
    classes = np.unique(yf)
    for c in classes:
        f_idx = np.where(yf == c)[0]
        o_idx = np.where(yo == c)[0]
        n = min(len(f_idx), len(o_idx))
        if n == 0:
            continue
        f_indices.extend(f_idx[:n].tolist())
        o_indices.extend(o_idx[:n].tolist())
        y_paired.extend([int(c)] * n)
    f_indices = np.array(f_indices)
    o_indices = np.array(o_indices)
    y = np.array(y_paired, dtype=np.int64)

    Xf_paired_raw = Xf_raw[f_indices]
    Xo_paired_raw = Xo_raw[o_indices]
    X_concat_raw = np.hstack([Xf_paired_raw, Xo_paired_raw])

    # Apply per-modality XGBoost feature selectors -> 50-D each
    with open("models/best_model.pkl", "rb") as f:
        fundus_pkg = pickle.load(f)
    with open("models/oct_best_model.pkl", "rb") as f:
        oct_pkg = pickle.load(f)
    fundus_model = fundus_pkg["model"] if isinstance(fundus_pkg, dict) and "model" in fundus_pkg else fundus_pkg
    oct_model = oct_pkg["model"] if isinstance(oct_pkg, dict) and "model" in oct_pkg else oct_pkg
    fundus_sel = fundus_pkg.get("feature_selector") if isinstance(fundus_pkg, dict) else None
    oct_sel = oct_pkg.get("feature_selector") if isinstance(oct_pkg, dict) else None

    Xf_paired_sel = fundus_sel.transform(Xf_paired_raw) if fundus_sel is not None else Xf_paired_raw
    Xo_paired_sel = oct_sel.transform(Xo_paired_raw) if oct_sel is not None else Xo_paired_raw

    return {
        "y": y,
        "X_concat_raw": X_concat_raw,  # (N, 278) for the DL model
        "Xf_paired_sel": Xf_paired_sel,  # (N, 50) for fundus XGB
        "Xo_paired_sel": Xo_paired_sel,
        "fundus_model": fundus_model,
        "oct_model": oct_model,
        "fundus_dim": Xf_paired_raw.shape[1],
        "oct_dim": Xo_paired_raw.shape[1],
    }


# ---------------------------------------------------------------------------
# 2) DL (attention) — load + score + extract gated features for hybrid path
# ---------------------------------------------------------------------------


def _attention_fusion_outputs(X_concat_raw: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Run the trained attention model on the paired test set; also return the gated features."""
    import torch
    from retina_attention_fusion import RetinoScopeFusionNetwork, load_checkpoint

    ckpt_path = "models/retina_attention_fusion.pt"
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    fundus_dim = int(getattr(cfg, "fundus_dim", 139))
    oct_dim = int(getattr(cfg, "oct_dim", 139))
    num_classes = int(getattr(cfg, "num_classes", 7))
    d_in = fundus_dim + oct_dim
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

    X = X_concat_raw.astype(np.float32, copy=True)
    if "scaler_mean" in ckpt and "scaler_scale" in ckpt:
        X = (X - ckpt["scaler_mean"].astype(np.float32)) / ckpt["scaler_scale"].astype(np.float32)

    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        logits, attn = model(xb)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        attn_np = attn.cpu().numpy()
        gated_np = (attn * xb).cpu().numpy()
    y_pred = np.argmax(proba, axis=1)

    return {
        "y_pred": y_pred,
        "proba": proba,
        "attention": attn_np,
        "gated_features": gated_np,
        "fundus_dim": fundus_dim,
        "oct_dim": oct_dim,
    }


# ---------------------------------------------------------------------------
# 3) Hybrid Attention -> XGBoost (refit with the gated features as input)
# ---------------------------------------------------------------------------


def _hybrid_attention_xgb(
    X_concat_raw_train: np.ndarray,
    y_train: np.ndarray,
    X_concat_raw_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Use the trained attention head to produce gated features for the *training pool*
    (paired by class, identical to the DL pipeline) and the *test set*. Then fit an
    XGBoost classifier on the gated 278-D vectors. Reports test metrics.
    """
    import torch
    from xgboost import XGBClassifier
    from retina_attention_fusion import RetinoScopeFusionNetwork, load_checkpoint

    ckpt_path = "models/retina_attention_fusion.pt"
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    fundus_dim = int(getattr(cfg, "fundus_dim", 139))
    oct_dim = int(getattr(cfg, "oct_dim", 139))
    num_classes = int(getattr(cfg, "num_classes", 7))
    d_in = fundus_dim + oct_dim
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

    if "scaler_mean" in ckpt and "scaler_scale" in ckpt:
        mean_ = ckpt["scaler_mean"].astype(np.float32)
        scale_ = ckpt["scaler_scale"].astype(np.float32)
    else:
        mean_, scale_ = 0.0, 1.0

    def gate(X_raw: np.ndarray) -> np.ndarray:
        Xs = (X_raw.astype(np.float32) - mean_) / scale_
        with torch.no_grad():
            x = torch.tensor(Xs, dtype=torch.float32)
            _logits, attn = model(x)
            gated = (attn * x).cpu().numpy()
        return gated

    G_train = gate(X_concat_raw_train)
    G_test = gate(X_concat_raw_test)

    # sqrt class weights, mean=1.0 — same recipe as the DL pipeline
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw = 1.0 / np.sqrt(counts)
    cw = (raw / raw.mean()).astype(np.float32)
    sample_weight = cw[y_train]

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        n_jobs=-1,
        random_state=42,
        eval_metric="mlogloss",
    )
    clf.fit(G_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(G_test)
    proba = clf.predict_proba(G_test)
    return {"y_pred": y_pred, "proba": proba}


# ---------------------------------------------------------------------------
# 4) Driver
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Benchmark all RetinoScope approaches on the same paired test set")
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument(
        "--skip_hybrid",
        action="store_true",
        help="Skip the Attention->XGBoost refit (saves a few minutes; otherwise it refits XGBoost on gated train pool)",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loading paired test set + per-modality XGBoost models")
    paired = _load_paired()
    y = paired["y"]
    Xf_sel = paired["Xf_paired_sel"]
    Xo_sel = paired["Xo_paired_sel"]
    X_concat = paired["X_concat_raw"]
    fundus_model = paired["fundus_model"]
    oct_model = paired["oct_model"]
    print(f"  N_pairs={len(y)} | XGB(F)={Xf_sel.shape} | XGB(O)={Xo_sel.shape} | DL={X_concat.shape}")

    results: Dict[str, Any] = {}

    # ----- Fundus-only / OCT-only XGBoost -----
    proba_f = fundus_model.predict_proba(Xf_sel)
    proba_o = oct_model.predict_proba(Xo_sel)
    y_pred_f = np.argmax(proba_f, axis=1)
    y_pred_o = np.argmax(proba_o, axis=1)
    results["xgb_fundus_only"] = _metrics_block(y, y_pred_f)
    results["xgb_oct_only"] = _metrics_block(y, y_pred_o)
    _save_cm(y, y_pred_f, "xgb_fundus_only", args.out_dir)
    _save_cm(y, y_pred_o, "xgb_oct_only", args.out_dir)

    # ----- Late fusion: weighted average (grid search) -----
    w_f, w_o, _gs = _grid_search_weights(proba_f, proba_o, y)
    p_w = w_f * proba_f + w_o * proba_o
    y_pred_w = np.argmax(p_w, axis=1)
    results["late_fusion_weighted_avg"] = {
        **_metrics_block(y, y_pred_w),
        "fundus_weight": w_f,
        "oct_weight": w_o,
    }
    _save_cm(y, y_pred_w, "late_fusion_weighted_avg", args.out_dir)

    # ----- Late fusion: geometric and max (fixed rules) -----
    p_geo = np.sqrt(proba_f * proba_o)
    p_geo = p_geo / p_geo.sum(axis=1, keepdims=True)
    p_max = np.maximum(proba_f, proba_o)
    p_max = p_max / p_max.sum(axis=1, keepdims=True)
    results["late_fusion_geometric"] = _metrics_block(y, np.argmax(p_geo, axis=1))
    results["late_fusion_max"] = _metrics_block(y, np.argmax(p_max, axis=1))
    _save_cm(y, np.argmax(p_geo, axis=1), "late_fusion_geometric", args.out_dir)
    _save_cm(y, np.argmax(p_max, axis=1), "late_fusion_max", args.out_dir)

    # ----- Attention mid-level fusion (DL) -----
    print("Running attention-fusion DL model on paired test set")
    dl = _attention_fusion_outputs(X_concat, y)
    results["attention_mid_fusion_dl"] = _metrics_block(y, dl["y_pred"])
    _save_cm(y, dl["y_pred"], "attention_mid_fusion_dl", args.out_dir)
    # Modality gate snapshot for the report
    fdim, odim = dl["fundus_dim"], dl["oct_dim"]
    mean_gate_f = float(dl["attention"][:, :fdim].mean())
    mean_gate_o = float(dl["attention"][:, fdim:].mean())
    results["attention_mid_fusion_dl"]["mean_gate_fundus"] = mean_gate_f
    results["attention_mid_fusion_dl"]["mean_gate_oct"] = mean_gate_o
    print(f"  mean attention gate — Fundus={mean_gate_f:.4f}, OCT={mean_gate_o:.4f}")

    # ----- Hybrid Attention -> XGBoost (optional but strongly recommended) -----
    if not args.skip_hybrid:
        print("Building gated training features for Attention->XGBoost hybrid")
        from retina_attention_fusion import _load_paired_modalities

        X_train_concat, y_train, _, _ = _load_paired_modalities(
            "features/train_features.npz",
            "features/train_features_labels.npy",
            "features/oct_train_features.npz",
            "features/oct_train_features_labels.npy",
            "train",
        )
        hyb = _hybrid_attention_xgb(X_train_concat, y_train, X_concat, y)
        results["hybrid_attention_xgb"] = _metrics_block(y, hyb["y_pred"])
        _save_cm(y, hyb["y_pred"], "hybrid_attention_xgb", args.out_dir)

    # ----- Persist results -----
    with open(os.path.join(args.out_dir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # CSV table
    csv_lines = ["approach,accuracy,balanced_accuracy,f1_macro,f1_weighted,n_test"]
    for name, m in results.items():
        csv_lines.append(
            f"{name},{m['accuracy']:.4f},{m['balanced_accuracy']:.4f},{m['f1_macro']:.4f},{m['f1_weighted']:.4f},{sum(m['support'].values())}"
        )
    with open(os.path.join(args.out_dir, "benchmark_results.csv"), "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Comparison bar chart
    names = list(results.keys())
    accs = [results[k]["accuracy"] for k in names]
    bals = [results[k]["balanced_accuracy"] for k in names]
    f1s = [results[k]["f1_macro"] for k in names]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(10, 1.4 * len(names)), 6))
    ax.bar(x - w, accs, w, label="Accuracy", color="#1f77b4")
    ax.bar(x, bals, w, label="Balanced Accuracy", color="#2ca02c")
    ax.bar(x + w, f1s, w, label="Macro-F1", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("RetinoScope — approach comparison (same paired test set)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for xi, vals in enumerate(zip(accs, bals, f1s)):
        for j, (v, off) in enumerate(zip(vals, [-w, 0, w])):
            ax.text(xi + off, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "benchmark_comparison.png"), dpi=150)
    plt.close(fig)

    # Console summary
    print("\n" + "=" * 78)
    print("BENCHMARK SUMMARY (same paired test set)")
    print("=" * 78)
    print(f"{'approach':<32} {'acc':>7} {'bal_acc':>9} {'f1_macro':>10} {'f1_w':>7}")
    for name, m in results.items():
        print(
            f"{name:<32} {m['accuracy']:>7.4f} {m['balanced_accuracy']:>9.4f} "
            f"{m['f1_macro']:>10.4f} {m['f1_weighted']:>7.4f}"
        )
    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
