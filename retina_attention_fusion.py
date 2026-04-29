"""
Advanced Mid-Level Fusion: Attention-Based Multimodal Fusion for Fundus + OCT Radiomics.

This module fuses **Fundus + OCT radiomics** (same number of features per modality, e.g.
**139 + 139 = 278** after extraction, or **50 + 50** if you pre-select features elsewhere).
The **attention gate** rescales each input dimension in **[0, 1]** before the classifier,
acting as a learned, sample-wise soft mask (no separate ANOVA/RFE step required when using
the full radiomics vector).

Data note (FYP):
- **Train/val**: paired Fundus train + OCT train → stratified train/val split for
  optimization and checkpoint selection (macro F1 on val).
- **Test**: paired Fundus test + OCT test files (`test_features*.npz`, `oct_test_*.npz`)
  are evaluated once after training on the best val checkpoint (never used in training).
- If you do not have true patient-level paired Fundus+OCT, use the same *class-balanced
  synthetic pairing* as latefusion.py (per class, min counts, first n indices).

Dependencies: torch, numpy, scikit-learn, matplotlib (attention diagnostic plot on test set).

Training defaults: StandardScaler; **WeightedRandomSampler** (inverse-frequency); **focal loss**
(gamma=2) with sqrt class weights; **early stopping** (patience 15 on val macro-F1); metrics include
**balanced accuracy** and a **minority-class recall** line (classes with ≤2% of test support).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; renders straight to file
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# 1) Data: load NPZ features + labels, map to 7 classes, pair by class
# ---------------------------------------------------------------------------


def load_npz_features(path: str) -> np.ndarray:
    data = np.load(path)
    if "features" in data:
        return data["features"]
    if "arr_0" in data:
        return data["arr_0"]
    return data[list(data.keys())[0]]


def map_labels_9_to_7(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map raw labels (0-8) to model labels (0-6), excluding classes 2 and 5.
    Same mapping as latefusion.py / train_model_only.
    """
    mapping = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6}
    valid = np.isin(y, list(mapping.keys()))
    y_f = y[valid]
    y_m = np.vectorize(mapping.get)(y_f)
    return y_m, valid


def pair_by_class(
    X_fundus: np.ndarray,
    y_fundus: np.ndarray,
    X_oct: np.ndarray,
    y_oct: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build synthetic pairs: for each class c, take min(n_f, n_o) pairs
    (first n indices from each modality). Returns concatenated [N, D_fundus + D_oct], y [N].
    """
    X_f_list: List[np.ndarray] = []
    X_o_list: List[np.ndarray] = []
    y_list: List[int] = []

    classes = np.unique(y_fundus)
    for c in classes:
        f_idx = np.where(y_fundus == c)[0]
        o_idx = np.where(y_oct == c)[0]
        n = min(len(f_idx), len(o_idx))
        if n == 0:
            continue
        f_sel = f_idx[:n]
        o_sel = o_idx[:n]
        X_f_list.append(X_fundus[f_sel])
        X_o_list.append(X_oct[o_sel])
        y_list.extend([int(c)] * n)

    if not X_f_list:
        raise RuntimeError("No overlapping classes to pair; check labels and shapes.")

    Xf = np.vstack(X_f_list)
    Xo = np.vstack(X_o_list)
    y = np.array(y_list, dtype=np.int64)
    X_concat = np.hstack([Xf, Xo])
    return X_concat, y, np.arange(len(y))


def load_checkpoint(path: str, map_location: str = "cpu"):
    """
    Load a checkpoint saved by ``run_training``. The original training script may
    have been launched as ``__main__`` (so ``FusionConfig.__module__`` was pickled
    as ``__main__``). When loading from a different module we have to alias
    ``FusionConfig`` into ``__main__`` so the unpickler can resolve it.
    """
    import sys
    import torch  # local import keeps top-level light for callers that need only constants

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "FusionConfig"):
        main_mod.FusionConfig = FusionConfig
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@dataclass
class FusionConfig:
    # Set from loaded data (e.g. 139+139); used for plotting / checkpoint metadata
    fundus_dim: int = 139
    oct_dim: int = 139
    num_classes: int = 7
    hidden2: int = 32
    dropout: float = 0.35
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    epochs: int = 80
    seed: int = 42
    val_ratio: float = 0.15
    # Tabular radiomics: scale each dimension (train-only fit) — critical for MLPs vs tree baselines
    use_standard_scaler: bool = True
    grad_clip_norm: float = 1.0
    # Imbalance: oversample minority classes each epoch (train loader only)
    use_balanced_sampler: bool = True
    # Focal loss down-weights easy examples; pairs well with sqrt class weights
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    # Stop if val macro F1 does not improve for N epochs (0 = run all epochs)
    early_stop_patience: int = 15


# ---------------------------------------------------------------------------
# 2) PyTorch Dataset: concatenated Fundus + OCT vector per sample
# ---------------------------------------------------------------------------


class PairedRadiomicsDataset(Dataset):
    """
    Each item is one vector (D_fundus + D_oct features, e.g. 278-D) and a label in {0..6}.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2 and y.ndim == 1
        assert len(X) == len(y)
        self.X = torch.FloatTensor(np.asarray(X, dtype=np.float32))
        self.y = torch.LongTensor(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# 3) Attention-Based Fusion Network
# ---------------------------------------------------------------------------


class RetinoScopeFusionNetwork(nn.Module):
    """
    Mid-level fusion with soft **gating** attention over the concatenated feature vector.

    Mechanism (for report):
    - Input x has shape (batch, D) with D = D_fundus + D_oct (e.g. 139+139 = 278). Conceptually
      Fundus occupies indices [0 : D_fundus) and OCT [D_fundus : D). The gate is learned over
      **all D dimensions** so the network can down-weight redundant or noisy radiomics channels
      and emphasize informative ones **per sample** (soft, non-linear feature weighting).
    - Attention weights a = sigmoid(W2 * ReLU(W1 * x + b1) + b2) ∈ (0,1)^D,
      same shape as x. This is **soft attention / feature gating**: each feature is rescaled
      independently based on the sample context.
    - The gated vector x' = a ⊙ x is fed into a small MLP classifier head with dropout.

    Hidden sizes scale with D (capped) to keep capacity reasonable on small cohorts.
    """

    def __init__(
        self,
        input_dim: int = 278,
        num_classes: int = 7,
        attn_hidden: int = 64,
        mlp_hidden1: int = 64,
        mlp_hidden2: int = 32,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Attention MLP: produces one logit per feature -> sigmoid -> [0,1]
        self.attn_fc1 = nn.Linear(input_dim, attn_hidden)
        self.attn_fc2 = nn.Linear(attn_hidden, input_dim)

        # Classification head on gated features
        self.fc1 = nn.Linear(input_dim, mlp_hidden1)
        self.fc2 = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc_out = nn.Linear(mlp_hidden2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, num_classes) raw logits
            attn:  (batch, input_dim) attention weights in (0,1)
        """
        # Attention logits per feature
        h = F.relu(self.attn_fc1(x))
        attn_logits = self.attn_fc2(h)
        attn = torch.sigmoid(attn_logits)  # (0,1) per feature; soft gating

        # Element-wise multiply: if a feature is uninformative/noisy, gate → small
        x_gated = attn * x

        z = F.relu(self.fc1(x_gated))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        z = self.dropout(z)
        logits = self.fc_out(z)
        return logits, attn


# ---------------------------------------------------------------------------
# 4) Square-root frequency class weights (normalized to mean 1.0) -> per-sample loss
# ---------------------------------------------------------------------------


def compute_sqrt_frequency_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    For each class c with count n_c, define raw weight w_c = 1 / sqrt(n_c).
    Normalize so mean(w_c) = 1.0 over classes that appear (or use all classes).

    Returns:
        class_weights: shape (num_classes,) for use in loss weighting.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid div by zero for missing classes
    raw = 1.0 / np.sqrt(counts)
    raw_mean = raw.mean()
    class_weights = raw / raw_mean
    return class_weights.astype(np.float32)


def plot_attention_diagnostics(
    attention: np.ndarray,
    fundus_dim: int,
    oct_dim: int,
    out_path: str,
) -> None:
    """
    Visualize soft attention gates from RetinoScopeFusionNetwork.

    attention: (N, fundus_dim + oct_dim) with values in (0, 1) — per-sample gates.
    We plot (1) mean ± std across test samples for each feature index, and
    (2) aggregate mean gate for Fundus block vs OCT block (interpretability).
    """
    assert attention.ndim == 2
    d = fundus_dim + oct_dim
    assert attention.shape[1] == d

    mean_w = attention.mean(axis=0)
    std_w = attention.std(axis=0)
    idx = np.arange(d)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].fill_between(idx, mean_w - std_w, mean_w + std_w, alpha=0.25, color="steelblue")
    axes[0].plot(idx, mean_w, color="steelblue", linewidth=1.2, label="Mean gate ±1 std")
    axes[0].axvline(fundus_dim - 0.5, color="gray", linestyle="--", linewidth=1, label="Fundus | OCT")
    axes[0].set_xlim(-0.5, d - 0.5)
    axes[0].set_ylabel("Attention weight (0–1)")
    axes[0].set_title("Per-feature attention (mean over held-out test samples)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    m_f = float(mean_w[:fundus_dim].mean())
    m_o = float(mean_w[fundus_dim:].mean())
    axes[1].bar(
        [f"Fundus ({fundus_dim}D)", f"OCT ({oct_dim}D)"],
        [m_f, m_o],
        color=["#2ecc71", "#9b59b6"],
    )
    axes[1].set_ylabel("Mean gate")
    axes[1].set_title("Average gate strength by modality (lower ⇒ more down-weighting on average)")
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Attention visualization saved to {out_path}")
    print(f"  Mean gate — Fundus block: {m_f:.4f} | OCT block: {m_o:.4f}")


def plot_per_class_attention(
    attention: np.ndarray,
    y_true: np.ndarray,
    fundus_dim: int,
    oct_dim: int,
    out_path: str,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Per-class attention diagnostic. For each class:
      - mean gate per feature (line plot, Fundus and OCT halves separated)
      - aggregate Fundus vs OCT block mean (bar)

    Returns a JSON-friendly dict ``{class_name: {fundus_mean, oct_mean}}`` to
    persist alongside the plot — substantiates report claims like
    "the gate prioritises OCT for Glaucoma".
    """
    assert attention.ndim == 2 and y_true.ndim == 1
    d = fundus_dim + oct_dim
    assert attention.shape[1] == d
    classes = sorted(np.unique(y_true).tolist())
    if class_names is None or len(class_names) <= max(classes):
        class_names = [f"Class {c}" for c in range(max(classes) + 1)]

    cols = min(3, len(classes))
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    summary: Dict[str, Any] = {}
    for ci, c in enumerate(classes):
        mask = y_true == c
        if not mask.any():
            continue
        sub = attention[mask]
        mu = sub.mean(axis=0)
        f_mean = float(mu[:fundus_dim].mean())
        o_mean = float(mu[fundus_dim:].mean())
        ax = axes[ci]
        ax.plot(np.arange(d), mu, color="steelblue", linewidth=1.0)
        ax.axvline(fundus_dim - 0.5, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlim(-0.5, d - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"{class_names[c]}  (n={int(mask.sum())})\nFundus μ={f_mean:.3f} | OCT μ={o_mean:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("feature index")
        ax.set_ylabel("attention gate")
        ax.grid(True, alpha=0.3)
        summary[class_names[c]] = {
            "fundus_mean_gate": f_mean,
            "oct_mean_gate": o_mean,
            "n_samples": int(mask.sum()),
        }

    for j in range(len(classes), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-class attention gates (held-out test)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """Confusion matrix PNG for the held-out test set."""
    n_cls = max(int(y_true.max()), int(y_pred.max())) + 1
    if class_names is None or len(class_names) < n_cls:
        class_names = [f"Class {i}" for i in range(n_cls)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names[:n_cls],
        yticklabels=class_names[:n_cls],
    )
    plt.title("Confusion Matrix — Attention Mid-Level Fusion")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def weighted_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample cross-entropy with class-dependent weights:
    loss_i = w[y_i] * CE(logits_i, y_i) for each sample, then mean over batch.

    class_weights should be normalized (mean 1.0) as produced above.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")  # (batch,)
    w = class_weights.to(logits.device)[targets]
    return (ce * w).mean()


def focal_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss (Lin et al.) with sqrt-frequency class weights:
    FL_i = w[y_i] * (1 - p_t)^gamma * CE_i, where p_t is the predicted probability of the true class.
    Focuses gradients on hard / misclassified examples.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1.0 - pt) ** gamma) * ce
    w = class_weights.to(logits.device)[targets]
    return (focal * w).mean()


def training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    use_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    if use_focal:
        return focal_weighted_loss(logits, targets, class_weights, gamma=focal_gamma)
    return weighted_cross_entropy_loss(logits, targets, class_weights)


def make_balanced_sampler_weights(y: np.ndarray, num_classes: int) -> torch.DoubleTensor:
    """Per-sample weights for WeightedRandomSampler: inverse class frequency."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv_freq = 1.0 / counts
    w_per_class = inv_freq / inv_freq.mean()
    sample_w = w_per_class[y]
    return torch.from_numpy(sample_w.astype(np.float64))


def mean_recall_rare_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    frac_max: float = 0.02,
) -> Tuple[float, List[int]]:
    """
    Mean per-class recall over classes whose **support in y_true** is at most frac_max * n.
    Highlights minority-class behaviour (complements macro-F1 / balanced accuracy).
    """
    n = len(y_true)
    recalls: List[float] = []
    classes: List[int] = []
    for c in np.unique(y_true):
        sup = int((y_true == c).sum())
        if sup == 0 or sup / n > frac_max:
            continue
        m = y_true == c
        r = float((y_pred[m] == c).mean()) if m.any() else 0.0
        recalls.append(r)
        classes.append(int(c))
    if not recalls:
        return float("nan"), []
    return float(np.mean(recalls)), classes


# ---------------------------------------------------------------------------
# 5) Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _attn = model(xb)
        loss = training_loss(logits, yb, class_weights, use_focal, focal_gamma)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    include_predictions: bool = False,
    include_attention: bool = False,
    use_focal_for_loss: bool = False,
    focal_gamma: float = 2.0,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds: List[int] = []
    all_true: List[int] = []
    attn_batches: List[np.ndarray] = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, attn = model(xb)
        loss = training_loss(logits, yb, class_weights, use_focal_for_loss, focal_gamma)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        pred = logits.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy().tolist())
        all_true.extend(yb.cpu().numpy().tolist())
        if include_attention:
            attn_batches.append(attn.detach().cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    out: Dict[str, Any] = {
        "loss": total_loss / max(n, 1),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
    }
    if include_predictions:
        out["y_true"] = y_true
        out["y_pred"] = y_pred
    if include_attention:
        out["attention"] = np.vstack(attn_batches) if attn_batches else np.empty((0, 0))
    return out


def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: FusionConfig,
    save_path: str = "models/retina_attention_fusion.pt",
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    attention_plot_path: Optional[str] = "fusion_attention_test.png",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    assert X_train.shape[1] == cfg.fundus_dim + cfg.oct_dim
    class_weights_np = compute_sqrt_frequency_weights(y_train, cfg.num_classes)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    scaler: Optional[StandardScaler] = None
    if cfg.use_standard_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float64, copy=False)).astype(np.float32)
        X_val = scaler.transform(X_val.astype(np.float64, copy=False)).astype(np.float32)
        if X_test is not None:
            X_test = scaler.transform(X_test.astype(np.float64, copy=False)).astype(np.float32)
        print(
            "  StandardScaler: fit on train only (zero mean, unit variance per feature). "
            "Use this for inference with the saved checkpoint."
        )

    train_ds = PairedRadiomicsDataset(X_train, y_train)
    val_ds = PairedRadiomicsDataset(X_val, y_val)
    if cfg.use_balanced_sampler:
        samp_w = make_balanced_sampler_weights(y_train, cfg.num_classes)
        train_sampler = WeightedRandomSampler(
            samp_w,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=train_sampler,
            drop_last=False,
        )
        print(
            "  WeightedRandomSampler: inverse-frequency oversampling (minority classes seen more often)."
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    if cfg.use_focal_loss:
        print(f"  Focal loss: gamma={cfg.focal_gamma} (with sqrt class weights).")
    if cfg.early_stop_patience > 0:
        print(f"  Early stopping: patience={cfg.early_stop_patience} epochs on val macro-F1.")

    d_in = cfg.fundus_dim + cfg.oct_dim
    # Scale capacity with input size (278-D needs wider layers than 100-D), stay capped for small N
    attn_hidden = max(64, min(256, d_in // 2))
    mlp_hidden1 = max(64, min(128, d_in // 2))
    print(
        f"  Model: input_dim={d_in}, attn_hidden={attn_hidden}, "
        f"mlp={mlp_hidden1}->{cfg.hidden2}, dropout={cfg.dropout}"
    )

    model = RetinoScopeFusionNetwork(
        input_dim=d_in,
        num_classes=cfg.num_classes,
        attn_hidden=attn_hidden,
        mlp_hidden1=mlp_hidden1,
        mlp_hidden2=cfg.hidden2,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_f1 = -1.0
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            class_weights,
            device,
            grad_clip_norm=cfg.grad_clip_norm,
            use_focal=cfg.use_focal_loss,
            focal_gamma=cfg.focal_gamma,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            class_weights,
            device,
            include_predictions=False,
            include_attention=False,
            use_focal_for_loss=cfg.use_focal_loss,
            focal_gamma=cfg.focal_gamma,
        )

        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            payload: Dict[str, Any] = {
                "model_state": model.state_dict(),
                "config": cfg,
                "class_weights": class_weights_np,
            }
            if scaler is not None:
                payload["scaler_mean"] = scaler.mean_.astype(np.float32)
                payload["scaler_scale"] = scaler.scale_.astype(np.float32)
            torch.save(payload, save_path)
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(
                f"\nEarly stopping at epoch {epoch} (no val macro-F1 improvement for "
                f"{cfg.early_stop_patience} epochs)."
            )
            break

    print(f"\nBest val macro-F1: {best_f1:.4f} (checkpoint saved to {save_path})")

    # Held-out test set (separate Fundus/OCT test files): same sqrt weights from *training* labels
    if X_test is not None and y_test is not None and len(y_test) > 0:
        ckpt = load_checkpoint(save_path, map_location=str(device))
        model.load_state_dict(ckpt["model_state"])
        test_ds = PairedRadiomicsDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        test_metrics = evaluate(
            model,
            test_loader,
            class_weights,
            device,
            include_predictions=True,
            include_attention=True,
            use_focal_for_loss=cfg.use_focal_loss,
            focal_gamma=cfg.focal_gamma,
        )
        y_true = test_metrics.pop("y_true")
        y_pred = test_metrics.pop("y_pred")
        attention_test = test_metrics.pop("attention")
        print("\n" + "=" * 60)
        print("HELD-OUT TEST SET (Fundus test + OCT test, class-paired)")
        print("=" * 60)
        print(
            f"  n={len(y_test)} | loss={test_metrics['loss']:.4f} | "
            f"accuracy={test_metrics['accuracy']:.4f} | "
            f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        print("\n" + classification_report(y_true, y_pred, digits=4, zero_division=0))
        mr, mc = mean_recall_rare_classes(y_true, y_pred, frac_max=0.02)
        if mc:
            print(
                f"\n  Minority classes (support ≤ 2% of test n): mean recall={mr:.4f} | "
                f"class ids={mc}"
            )

        if attention_plot_path and attention_test.size > 0:
            plot_attention_diagnostics(
                attention_test,
                cfg.fundus_dim,
                cfg.oct_dim,
                attention_plot_path,
            )
            # Per-class diagnostic (substantiates "gate prioritised OCT for class X")
            per_class_path = os.path.splitext(attention_plot_path)[0] + "_per_class.png"
            per_class_summary = plot_per_class_attention(
                attention_test,
                y_true,
                cfg.fundus_dim,
                cfg.oct_dim,
                per_class_path,
            )
            print(f"Per-class attention plot saved to {per_class_path}")

        # Confusion matrix for the DL test predictions
        cm_path = os.path.splitext(save_path)[0] + "_test_confusion.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)
        print(f"Confusion matrix saved to {cm_path}")

        # Persist a machine-readable metrics summary (used by the report + dashboard)
        metrics_path = os.path.splitext(save_path)[0] + "_test_metrics.json"
        per_class_f1 = f1_score(
            y_true, y_pred, labels=list(range(cfg.num_classes)),
            average=None, zero_division=0,
        )
        mr_rare, mc_rare = mean_recall_rare_classes(y_true, y_pred, frac_max=0.02)
        payload_metrics: Dict[str, Any] = {
            "n_test": int(len(y_test)),
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "per_class_f1": {int(c): float(s) for c, s in enumerate(per_class_f1)},
            "support": {int(c): int((y_true == c).sum()) for c in range(cfg.num_classes)},
            "rare_class_mean_recall": None if np.isnan(mr_rare) else float(mr_rare),
            "rare_class_ids": mc_rare,
            "best_val_macro_f1": float(best_f1),
            "config": {
                "fundus_dim": cfg.fundus_dim,
                "oct_dim": cfg.oct_dim,
                "num_classes": cfg.num_classes,
                "use_focal_loss": cfg.use_focal_loss,
                "focal_gamma": cfg.focal_gamma,
                "use_balanced_sampler": cfg.use_balanced_sampler,
                "use_standard_scaler": cfg.use_standard_scaler,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "seed": cfg.seed,
            },
        }
        if attention_plot_path and attention_test.size > 0:
            payload_metrics["mean_gate_fundus"] = float(
                attention_test[:, : cfg.fundus_dim].mean()
            )
            payload_metrics["mean_gate_oct"] = float(
                attention_test[:, cfg.fundus_dim :].mean()
            )
            payload_metrics["per_class_attention"] = per_class_summary
        with open(metrics_path, "w") as f:
            json.dump(payload_metrics, f, indent=2)
        print(f"Test metrics JSON saved to {metrics_path}")


def evaluate_checkpoint(
    save_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attention_plot_path: Optional[str] = "fusion_attention_test.png",
) -> None:
    """
    Inference-only mode: load a saved checkpoint and regenerate the held-out
    test artefacts (attention plots, per-class plot, confusion matrix, metrics
    JSON). No training. Useful when the report needs the new diagnostics
    without paying the training cost again.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(save_path, map_location=str(device))
    cfg = ckpt["config"]
    fundus_dim = int(getattr(cfg, "fundus_dim", 139))
    oct_dim = int(getattr(cfg, "oct_dim", 139))
    num_classes = int(getattr(cfg, "num_classes", 7))
    d_in = fundus_dim + oct_dim

    if X_test.shape[1] != d_in:
        raise ValueError(
            f"Checkpoint expects input_dim={d_in} (fundus {fundus_dim}+oct {oct_dim}); "
            f"got X_test with {X_test.shape[1]} cols."
        )

    if "scaler_mean" in ckpt and "scaler_scale" in ckpt:
        mean_ = ckpt["scaler_mean"].astype(np.float32)
        scale_ = ckpt["scaler_scale"].astype(np.float32)
        X_test = (X_test.astype(np.float32) - mean_) / scale_
        print("  applied saved StandardScaler from checkpoint")

    attn_hidden = max(64, min(256, d_in // 2))
    mlp_hidden1 = max(64, min(128, d_in // 2))
    model = RetinoScopeFusionNetwork(
        input_dim=d_in,
        num_classes=num_classes,
        attn_hidden=attn_hidden,
        mlp_hidden1=mlp_hidden1,
        mlp_hidden2=int(getattr(cfg, "hidden2", 32)),
        dropout=float(getattr(cfg, "dropout", 0.35)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    cw_np = ckpt.get("class_weights")
    if cw_np is None:
        cw_np = np.ones(num_classes, dtype=np.float32)
    class_weights = torch.tensor(np.asarray(cw_np, dtype=np.float32), dtype=torch.float32)

    test_ds = PairedRadiomicsDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=int(getattr(cfg, "batch_size", 64)), shuffle=False)
    use_focal = bool(getattr(cfg, "use_focal_loss", False))
    focal_gamma = float(getattr(cfg, "focal_gamma", 2.0))
    test_metrics = evaluate(
        model, test_loader, class_weights, device,
        include_predictions=True, include_attention=True,
        use_focal_for_loss=use_focal, focal_gamma=focal_gamma,
    )
    y_true = test_metrics.pop("y_true")
    y_pred = test_metrics.pop("y_pred")
    attention_test = test_metrics.pop("attention")

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET (eval-only)")
    print("=" * 60)
    print(
        f"  n={len(y_test)} | loss={test_metrics['loss']:.4f} | "
        f"accuracy={test_metrics['accuracy']:.4f} | "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} | "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print("\n" + classification_report(y_true, y_pred, digits=4, zero_division=0))

    per_class_summary: Optional[Dict[str, Any]] = None
    if attention_plot_path and attention_test.size > 0:
        plot_attention_diagnostics(attention_test, fundus_dim, oct_dim, attention_plot_path)
        per_class_path = os.path.splitext(attention_plot_path)[0] + "_per_class.png"
        per_class_summary = plot_per_class_attention(
            attention_test, y_true, fundus_dim, oct_dim, per_class_path,
        )
        print(f"Per-class attention plot saved to {per_class_path}")

    cm_path = os.path.splitext(save_path)[0] + "_test_confusion.png"
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    metrics_path = os.path.splitext(save_path)[0] + "_test_metrics.json"
    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0,
    )
    mr_rare, mc_rare = mean_recall_rare_classes(y_true, y_pred, frac_max=0.02)
    payload_metrics: Dict[str, Any] = {
        "n_test": int(len(y_test)),
        "loss": float(test_metrics["loss"]),
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "per_class_f1": {int(c): float(s) for c, s in enumerate(per_class_f1)},
        "support": {int(c): int((y_true == c).sum()) for c in range(num_classes)},
        "rare_class_mean_recall": None if np.isnan(mr_rare) else float(mr_rare),
        "rare_class_ids": mc_rare,
        "config": {
            "fundus_dim": fundus_dim, "oct_dim": oct_dim, "num_classes": num_classes,
            "use_focal_loss": use_focal, "focal_gamma": focal_gamma,
        },
        "mode": "eval_only",
    }
    if attention_plot_path and attention_test.size > 0:
        payload_metrics["mean_gate_fundus"] = float(attention_test[:, :fundus_dim].mean())
        payload_metrics["mean_gate_oct"] = float(attention_test[:, fundus_dim:].mean())
        if per_class_summary is not None:
            payload_metrics["per_class_attention"] = per_class_summary
    with open(metrics_path, "w") as f:
        json.dump(payload_metrics, f, indent=2)
    print(f"Test metrics JSON saved to {metrics_path}")


def _load_paired_modalities(
    fundus_npz: str,
    fundus_labels: str,
    oct_npz: str,
    oct_labels: str,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Load two modalities, map labels to 7 classes, pair by class.

    Returns X_concat [N, D_fundus + D_oct], y [N], fundus_dim, oct_dim.
    Typical pipeline: **139-D Fundus + 139-D OCT** (full radiomics); the fusion net learns
    gates over all dimensions—no RFE step required for that path.
    """
    print(f"  [{label}] {fundus_npz} + {oct_npz}")
    Xf = load_npz_features(fundus_npz)
    yf_raw = np.load(fundus_labels)
    Xo = load_npz_features(oct_npz)
    yo_raw = np.load(oct_labels)

    if Xf.ndim != 2 or Xo.ndim != 2:
        raise ValueError(f"Expected 2D feature arrays; got Fundus {Xf.shape}, OCT {Xo.shape}")

    yf, mf = map_labels_9_to_7(yf_raw)
    Xf = Xf[mf]
    yo, mo = map_labels_9_to_7(yo_raw)
    Xo = Xo[mo]

    fundus_dim, oct_dim = Xf.shape[1], Xo.shape[1]
    if fundus_dim != oct_dim:
        print(
            f"[Note] Fundus D={fundus_dim} and OCT D={oct_dim} differ; "
            "concatenation is still valid (indices [0:D_f) vs [D_f:D_f+D_o))."
        )

    X_concat, y, _ = pair_by_class(Xf, yf, Xo, yo)
    return X_concat, y, fundus_dim, oct_dim


def main():
    parser = argparse.ArgumentParser(description="Attention-based mid-level fusion (Fundus+OCT)")
    parser.add_argument("--fundus_train", type=str, default="features/train_features.npz")
    parser.add_argument("--fundus_train_labels", type=str, default="features/train_features_labels.npy")
    parser.add_argument("--oct_train", type=str, default="features/oct_train_features.npz")
    parser.add_argument("--oct_train_labels", type=str, default="features/oct_train_features_labels.npy")
    parser.add_argument("--fundus_test", type=str, default="features/test_features.npz")
    parser.add_argument("--fundus_test_labels", type=str, default="features/test_features_labels.npy")
    parser.add_argument("--oct_test", type=str, default="features/oct_test_features.npz")
    parser.add_argument("--oct_test_labels", type=str, default="features/oct_test_features_labels.npy")
    parser.add_argument("--no_test", action="store_true", help="Skip held-out test evaluation")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument(
        "--no_scale",
        action="store_true",
        help="Disable StandardScaler (not recommended for MLP on raw radiomics)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max gradient norm (0 = disable clipping)",
    )
    parser.add_argument("--save", type=str, default="models/retina_attention_fusion.pt")
    parser.add_argument(
        "--attention_plot",
        type=str,
        default="fusion_attention_test.png",
        help="Where to save attention diagnostic plot (test set). Empty string disables.",
    )
    parser.add_argument(
        "--no_balanced_sampler",
        action="store_true",
        help="Disable inverse-frequency oversampling on the train loader",
    )
    parser.add_argument(
        "--no_focal",
        action="store_true",
        help="Use weighted CE only (disable focal loss)",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument(
        "--early_stop",
        type=int,
        default=15,
        help="Stop if val macro-F1 does not improve for N epochs (0 = disabled)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of random seeds for multi-seed evaluation, e.g. "
            "--seeds 42 7 13 99 2024. When provided, the script trains one model "
            "per seed (each writes a suffixed checkpoint), then aggregates "
            "test metrics into <save>_seed_summary.json (mean ± std)."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Skip training; load the checkpoint at --save and re-run held-out "
            "test evaluation only. Regenerates attention plots, per-class plot, "
            "confusion matrix, and the metrics JSON without retraining."
        ),
    )
    args = parser.parse_args()

    print("Loading TRAIN features and building class-balanced pairs...")
    X_concat, y, fundus_dim, oct_dim = _load_paired_modalities(
        args.fundus_train,
        args.fundus_train_labels,
        args.oct_train,
        args.oct_train_labels,
        "train",
    )
    print(f"Paired train pool: X={X_concat.shape}, y={y.shape}, classes={np.unique(y)}")

    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    if not args.no_test:
        paths_ok = (
            os.path.isfile(args.fundus_test)
            and os.path.isfile(args.fundus_test_labels)
            and os.path.isfile(args.oct_test)
            and os.path.isfile(args.oct_test_labels)
        )
        if paths_ok:
            print("\nLoading HELD-OUT TEST features (same pairing as latefusion)...")
            X_test, y_test, fd_t, od_t = _load_paired_modalities(
                args.fundus_test,
                args.fundus_test_labels,
                args.oct_test,
                args.oct_test_labels,
                "test",
            )
            if X_test.shape[1] != X_concat.shape[1]:
                raise ValueError(
                    f"Train concat width {X_concat.shape[1]} != test width {X_test.shape[1]} "
                    f"(train Fundus/OCT {fundus_dim}/{oct_dim}, test {fd_t}/{od_t}). "
                    "Use the same feature extraction for train and test."
                )
            print(f"Paired test set: X={X_test.shape}, y={y_test.shape}")
        else:
            print(
                "\n[Note] Test files not all found; skipping held-out test. "
                "Expected:\n"
                f"  {args.fundus_test}, {args.fundus_test_labels},\n"
                f"  {args.oct_test}, {args.oct_test_labels}\n"
                "Use --no_test to silence this, or extract features to these paths."
            )

    attn_path: Optional[str] = args.attention_plot.strip() or None

    if args.eval_only:
        if X_test is None or y_test is None:
            raise SystemExit(
                "--eval_only requires the held-out test files to be present. "
                "Check the --fundus_test/--oct_test paths."
            )
        if not os.path.isfile(args.save):
            raise SystemExit(f"--eval_only: checkpoint not found at {args.save}")
        print(f"\n[eval-only] reusing checkpoint {args.save}")
        evaluate_checkpoint(
            args.save,
            X_test,
            y_test,
            attention_plot_path=attn_path,
        )
        return

    seeds: List[int] = args.seeds if args.seeds else [42]
    multi_seed = len(seeds) > 1
    save_root, save_ext = os.path.splitext(args.save)
    attn_root, attn_ext = (
        os.path.splitext(attn_path) if attn_path else (None, None)
    )

    seed_records: List[Dict[str, Any]] = []
    for seed in seeds:
        print("\n" + "#" * 70)
        print(f"#  Training run with seed={seed}")
        print("#" * 70)

        cfg = FusionConfig(
            fundus_dim=fundus_dim,
            oct_dim=oct_dim,
            val_ratio=args.val_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_standard_scaler=not args.no_scale,
            grad_clip_norm=args.grad_clip,
            use_balanced_sampler=not args.no_balanced_sampler,
            use_focal_loss=not args.no_focal,
            focal_gamma=args.focal_gamma,
            early_stop_patience=args.early_stop,
            seed=seed,
        )
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_concat,
            y,
            test_size=cfg.val_ratio,
            random_state=cfg.seed,
            stratify=y,
        )
        print(f"Training split: {X_tr.shape[0]} | Validation (from train): {X_va.shape[0]}")

        if multi_seed:
            run_save = f"{save_root}_seed{seed}{save_ext}"
            run_attn = (
                f"{attn_root}_seed{seed}{attn_ext}"
                if attn_path is not None
                else None
            )
        else:
            run_save = args.save
            run_attn = attn_path

        run_training(
            X_tr,
            y_tr,
            X_va,
            y_va,
            cfg,
            save_path=run_save,
            X_test=X_test,
            y_test=y_test,
            attention_plot_path=run_attn,
        )

        # Pick up the metrics JSON the run just wrote
        metrics_file = os.path.splitext(run_save)[0] + "_test_metrics.json"
        if multi_seed and os.path.isfile(metrics_file):
            with open(metrics_file) as f:
                seed_records.append({"seed": seed, **json.load(f)})

    if multi_seed and seed_records:
        keys = ["accuracy", "balanced_accuracy", "macro_f1"]
        agg = {
            k: {
                "mean": float(np.mean([r[k] for r in seed_records])),
                "std": float(np.std([r[k] for r in seed_records])),
                "values": [float(r[k]) for r in seed_records],
            }
            for k in keys
        }
        summary_path = f"{save_root}_seed_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {"seeds": seeds, "aggregate": agg, "per_seed": seed_records},
                f,
                indent=2,
            )
        print("\n" + "=" * 70)
        print("MULTI-SEED SUMMARY")
        print("=" * 70)
        for k, v in agg.items():
            print(f"  {k:<20} mean={v['mean']:.4f}  std={v['std']:.4f}  vals={v['values']}")
        print(f"\nWritten to {summary_path}")


if __name__ == "__main__":
    main()
