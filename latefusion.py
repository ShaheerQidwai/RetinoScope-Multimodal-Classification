import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import os
import sys
import argparse

# ==========================================
# CONFIGURATION
# ==========================================
FUNDUS_MODEL_PATH = 'models/best_model.pkl'
OCT_MODEL_PATH = 'models/oct_best_model.pkl'

FUNDUS_FEATURES_PATH = 'features/test_features.npz'
FUNDUS_LABELS_PATH = 'features/test_features_labels.npy'

OCT_FEATURES_PATH = 'features/oct_test_features.npz'
OCT_LABELS_PATH = 'features/oct_test_features_labels.npy'

PLOT_OUTPUT = 'fusion_results.png'

def load_npz_features(path):
    """Helper to load features from .npz handling different key names."""
    data = np.load(path)
    if 'arr_0' in data:
        return data['arr_0']
    elif 'features' in data:
        return data['features']
    else:
        return data[list(data.keys())[0]]

def apply_feature_selection(X, model_pkg, modality_name):
    """Applies the saved feature selector (RFE) to reduce dimensions."""
    if isinstance(model_pkg, dict) and 'feature_selector' in model_pkg:
        selector = model_pkg['feature_selector']
        try:
            print(f"[{modality_name}] Applying Feature Selection: {X.shape} -> ", end="")
            X_new = selector.transform(X)
            print(f"{X_new.shape}")
            return X_new
        except Exception as e:
            print(f"\n[WARNING] Could not use saved selector for {modality_name}: {e}")
            return X
    else:
        print(f"[{modality_name}] No feature_selector found. Using raw features.")
        return X

def map_labels_to_model_format(y_raw, modality_name):
    """Maps raw labels (0-8) to model labels (0-6). Excludes classes 2 and 5."""
    print(f"\n[{modality_name}] Mapping Labels (Raw 0-8 -> Model 0-6)...")
    
    mapping = {
        0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 8: 6
    }
    
    valid_mask = np.isin(y_raw, list(mapping.keys()))
    y_filtered = y_raw[valid_mask]
    y_mapped = np.vectorize(mapping.get)(y_filtered)
    
    print(f"  Original shape: {y_raw.shape}")
    print(f"  Filtered shape: {y_mapped.shape} (Removed classes 2, 5)")
    
    return y_mapped, valid_mask

def fuse_probabilities(p_f, p_o, method='weighted_average', fundus_weight=0.3, oct_weight=0.7):
    """
    Fuse probability predictions from two models.
    
    Args:
        p_f: Fundus probabilities (n_samples, n_classes)
        p_o: OCT probabilities (n_samples, n_classes)
        method: 'weighted_average', 'average', 'product', 'max'
        fundus_weight: Weight for Fundus (for weighted_average)
        oct_weight: Weight for OCT (for weighted_average)
    """
    if method == 'weighted_average':
        total_weight = fundus_weight + oct_weight
        p_fused = (fundus_weight * p_f + oct_weight * p_o) / total_weight
    elif method == 'average':
        p_fused = (p_f + p_o) / 2.0
    elif method == 'product':
        # Geometric mean (product then normalize)
        p_fused = np.sqrt(p_f * p_o)
        p_fused = p_fused / p_fused.sum(axis=1, keepdims=True)
    elif method == 'max':
        # Take maximum probability for each class
        p_fused = np.maximum(p_f, p_o)
        p_fused = p_fused / p_fused.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    
    return p_fused

def find_optimal_weights(y_true, y_fundus_proba, y_oct_proba, method='f1_macro'):
    """
    Find optimal fusion weights using grid search.
    
    Args:
        y_true: True labels
        y_fundus_proba: Fundus probability predictions
        y_oct_proba: OCT probability predictions
        method: Metric to optimize ('f1_macro', 'accuracy')
    """
    best_score = 0
    best_weights = (0.5, 0.5)
    
    # Grid search over weight combinations
    weights = np.arange(0.1, 1.0, 0.1)
    
    for w_f in weights:
        w_o = 1.0 - w_f
        p_fused = fuse_probabilities(y_fundus_proba, y_oct_proba, 
                                    'weighted_average', w_f, w_o)
        y_pred = np.argmax(p_fused, axis=1)
        
        if method == 'f1_macro':
            score = f1_score(y_true, y_pred, average='macro')
        else:
            score = accuracy_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_weights = (w_f, w_o)
    
    return best_weights, best_score

def main():
    parser = argparse.ArgumentParser(description='Simulated Late Fusion')
    parser.add_argument('--fundus_weight', type=float, default=0.3,
                       help='Weight for Fundus model (default: 0.3)')
    parser.add_argument('--oct_weight', type=float, default=0.7,
                       help='Weight for OCT model (default: 0.7)')
    parser.add_argument('--fusion_method', type=str, default='weighted_average',
                       choices=['weighted_average', 'average', 'product', 'max'],
                       help='Fusion method')
    parser.add_argument('--optimize_weights', action='store_true',
                       help='Find optimal weights using grid search')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMULATED LATE FUSION (Class-Based Pairing)")
    print("="*70)

    # 1. DATA LOADING
    print("\n[Step 1] Loading Data...")
    X_fundus_raw = load_npz_features(FUNDUS_FEATURES_PATH)
    y_fundus_raw = np.load(FUNDUS_LABELS_PATH)
    X_oct_raw = load_npz_features(OCT_FEATURES_PATH)
    y_oct_raw = np.load(OCT_LABELS_PATH)

    # 1.5 LABEL MAPPING
    print("\n[Step 1.5] Applying Label Mapping...")
    y_fundus, f_mask = map_labels_to_model_format(y_fundus_raw, "Fundus")
    X_fundus = X_fundus_raw[f_mask]
    y_oct, o_mask = map_labels_to_model_format(y_oct_raw, "OCT")
    X_oct = X_oct_raw[o_mask]

    print(f"Fundus Final Data: {X_fundus.shape}")
    print(f"OCT Final Data:    {X_oct.shape}")

    # 2. MODEL LOADING
    print("\n[Step 2] Loading Models...")
    with open(FUNDUS_MODEL_PATH, 'rb') as f:
        fundus_pkg = pickle.load(f)
    with open(OCT_MODEL_PATH, 'rb') as f:
        oct_pkg = pickle.load(f)

    fundus_model = fundus_pkg['model'] if isinstance(fundus_pkg, dict) and 'model' in fundus_pkg else fundus_pkg
    oct_model = oct_pkg['model'] if isinstance(oct_pkg, dict) and 'model' in oct_pkg else oct_pkg

    # 2.5 APPLY FEATURE SELECTION
    print("\n[Step 2.5] Applying Feature Selection...")
    X_fundus = apply_feature_selection(X_fundus, fundus_pkg, "Fundus")
    X_oct = apply_feature_selection(X_oct, oct_pkg, "OCT")

    # 3. CONSISTENCY CHECK
    print("\n[Step 3] Checking Class Consistency...")
    probs_f = fundus_model.predict_proba(X_fundus[0:1])
    probs_o = oct_model.predict_proba(X_oct[0:1])

    if probs_f.shape[1] != probs_o.shape[1]:
        print(f"[ERROR] Class mismatch ({probs_f.shape[1]} vs {probs_o.shape[1]}).")
        return

    # 4. SYNTHETIC PAIRING & FUSION
    print("\n[Step 4] Running Class-Based Pairing & Fusion...")
    
    y_true_paired = []
    y_pred_fundus = []
    y_pred_oct = []
    y_pred_fused = []
    proba_fundus_all = []
    proba_oct_all = []

    classes = np.unique(y_fundus)

    for cls in classes:
        f_idx = np.where(y_fundus == cls)[0]
        o_idx = np.where(y_oct == cls)[0]
        
        n_pairs = min(len(f_idx), len(o_idx))
        
        if n_pairs == 0:
            continue
            
        f_sel = f_idx[:n_pairs]
        o_sel = o_idx[:n_pairs]

        # Get Probabilities
        p_f = fundus_model.predict_proba(X_fundus[f_sel])
        p_o = oct_model.predict_proba(X_oct[o_sel])

        proba_fundus_all.append(p_f)
        proba_oct_all.append(p_o)

        # Fusion
        p_fused = fuse_probabilities(p_f, p_o, args.fusion_method, 
                                    args.fundus_weight, args.oct_weight)

        y_true_paired.extend([cls] * n_pairs)
        y_pred_fundus.extend(np.argmax(p_f, axis=1))
        y_pred_oct.extend(np.argmax(p_o, axis=1))
        y_pred_fused.extend(np.argmax(p_fused, axis=1))

    # Convert to arrays
    y_true = np.array(y_true_paired)
    y_pred_f = np.array(y_pred_fundus)
    y_pred_o = np.array(y_pred_oct)
    y_pred_fusion = np.array(y_pred_fused)
    
    proba_f_all = np.vstack(proba_fundus_all)
    proba_o_all = np.vstack(proba_oct_all)

    # Optimize weights if requested
    if args.optimize_weights:
        print("\n[Step 4.5] Optimizing Fusion Weights...")
        best_weights, best_score = find_optimal_weights(y_true, proba_f_all, proba_o_all)
        args.fundus_weight, args.oct_weight = best_weights
        print(f"Optimal weights: Fundus={best_weights[0]:.2f}, OCT={best_weights[1]:.2f}")
        print(f"Best F1-macro: {best_score:.4f}")
        
        # Re-fuse with optimal weights
        p_fused_opt = fuse_probabilities(proba_f_all, proba_o_all, 
                                        'weighted_average', 
                                        args.fundus_weight, args.oct_weight)
        y_pred_fusion = np.argmax(p_fused_opt, axis=1)

    # 5. RESULTS
    print("\n" + "="*70)
    print("FUSION RESULTS")
    print("="*70)
    print(f"\nFusion Method: {args.fusion_method}")
    print(f"Fundus Weight: {args.fundus_weight:.2f}")
    print(f"OCT Weight: {args.oct_weight:.2f}")
    print("-" * 70)
    
    print("\nIndividual Model Performance:")
    print(f"Fundus Only:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred_f):.4f}")
    print(f"  F1-macro: {f1_score(y_true, y_pred_f, average='macro'):.4f}")
    print(f"  F1-weighted: {f1_score(y_true, y_pred_f, average='weighted'):.4f}")
    
    print(f"\nOCT Only:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred_o):.4f}")
    print(f"  F1-macro: {f1_score(y_true, y_pred_o, average='macro'):.4f}")
    print(f"  F1-weighted: {f1_score(y_true, y_pred_o, average='weighted'):.4f}")
    
    print(f"\nFUSED SYSTEM:")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred_fusion):.4f}")
    print(f"  F1-macro: {f1_score(y_true, y_pred_fusion, average='macro'):.4f}")
    print(f"  F1-weighted: {f1_score(y_true, y_pred_fusion, average='weighted'):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred_fusion, average='macro', zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_true, y_pred_fusion, average='macro', zero_division=0):.4f}")
    print("-" * 70)

    print("\nClassification Report (Fused System):")
    target_names = ['Normal', 'Class 1', 'Class 3', 'Class 4', 'Class 6', 'Class 7', 'Class 8']
    print(classification_report(y_true, y_pred_fusion, digits=4, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_fusion)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Fused System Confusion Matrix\nAccuracy: {accuracy_score(y_true, y_pred_fusion):.4f} | F1-macro: {f1_score(y_true, y_pred_fusion, average="macro"):.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"\nPlot saved to {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()