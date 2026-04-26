import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
# Paths to your trained models
FUNDUS_MODEL_PATH = 'models/best_model.pkl'
OCT_MODEL_PATH = 'models/oct_best_model.pkl'

# Paths to your Test Data (Features & Labels)
FUNDUS_FEATURES_PATH = 'features/test_features.npz'
FUNDUS_LABELS_PATH = 'features/test_features_labels.npy'

OCT_FEATURES_PATH = 'features/oct_test_features.npz'
OCT_LABELS_PATH = 'features/oct_test_features_labels.npy'

# Output plot filename
PLOT_OUTPUT = 'fusion_results.png'

def load_npz_features(path):
    """Helper to load features from .npz handling different key names."""
    data = np.load(path)
    if 'arr_0' in data:
        return data['arr_0']
    elif 'features' in data:
        return data['features']
    else:
        # Fallback: return the first array found
        return data[list(data.keys())[0]]

def apply_feature_selection(X, model_pkg, modality_name):
    """
    Applies the saved feature selector (RFE) to reduce dimensions (e.g., 139 -> 50).
    """
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

# ... (Imports and Configuration remain the same) ...

def map_labels_to_model_format(y_raw, modality_name):
    """
    Maps raw labels (0-8) to model labels (0-6).
    Excludes classes 2 and 5.
    
    Mapping:
    0 -> 0
    1 -> 1
    3 -> 2
    4 -> 3
    6 -> 4
    7 -> 5
    8 -> 6
    """
    print(f"\n[{modality_name}] Mapping Labels (Raw 0-8 -> Model 0-6)...")
    
    # Define the mapping dictionary
    mapping = {
        0: 0,
        1: 1,
        3: 2,
        4: 3,
        6: 4,
        7: 5,
        8: 6
    }
    
    # Create mask for valid classes (exclude 2 and 5)
    valid_mask = np.isin(y_raw, list(mapping.keys()))
    
    # Filter the arrays
    y_filtered = y_raw[valid_mask]
    
    # Apply mapping
    y_mapped = np.vectorize(mapping.get)(y_filtered)
    
    print(f"  Original shape: {y_raw.shape}")
    print(f"  Filtered shape: {y_mapped.shape} (Removed classes 2, 5)")
    
    return y_mapped, valid_mask

def main():
    print("="*50)
    print("STARTING SIMULATED LATE FUSION (With Label Mapping)")
    print("="*50)

    # -------------------------------------------------------
    # 1. DATA LOADING
    # -------------------------------------------------------
    print("\n[Step 1] Loading Data...")
    
    X_fundus_raw = load_npz_features(FUNDUS_FEATURES_PATH)
    y_fundus_raw = np.load(FUNDUS_LABELS_PATH)
    
    X_oct_raw = load_npz_features(OCT_FEATURES_PATH)
    y_oct_raw = np.load(OCT_LABELS_PATH)

    # -------------------------------------------------------
    # 1.5 LABEL MAPPING (THE FIX)
    # -------------------------------------------------------
    print("\n[Step 1.5] Applying Label Mapping...")
    
    # Map Fundus Labels
    y_fundus, f_mask = map_labels_to_model_format(y_fundus_raw, "Fundus")
    X_fundus = X_fundus_raw[f_mask] # Apply same filter to features
    
    # Map OCT Labels
    y_oct, o_mask = map_labels_to_model_format(y_oct_raw, "OCT")
    X_oct = X_oct_raw[o_mask]     # Apply same filter to features

    print(f"Fundus Final Data: {X_fundus.shape}")
    print(f"OCT Final Data:    {X_oct.shape}")

    # -------------------------------------------------------
    # 2. MODEL LOADING
    # -------------------------------------------------------
    print("\n[Step 2] Loading Models...")

    with open(FUNDUS_MODEL_PATH, 'rb') as f:
        fundus_pkg = pickle.load(f)

    with open(OCT_MODEL_PATH, 'rb') as f:
        oct_pkg = pickle.load(f)

    fundus_model = fundus_pkg['model'] if isinstance(fundus_pkg, dict) and 'model' in fundus_pkg else fundus_pkg
    oct_model = oct_pkg['model'] if isinstance(oct_pkg, dict) and 'model' in oct_pkg else oct_pkg

    # -------------------------------------------------------
    # 2.5 APPLY FEATURE SELECTION
    # -------------------------------------------------------
    print("\n[Step 2.5] Applying Feature Selection...")
    X_fundus = apply_feature_selection(X_fundus, fundus_pkg, "Fundus")
    X_oct = apply_feature_selection(X_oct, oct_pkg, "OCT")

    # -------------------------------------------------------
    # 3. CONSISTENCY CHECK
    # -------------------------------------------------------
    print("\n[Step 3] Checking Class Consistency...")
    probs_f = fundus_model.predict_proba(X_fundus[0:1])
    probs_o = oct_model.predict_proba(X_oct[0:1])

    if probs_f.shape[1] != probs_o.shape[1]:
        print(f"[CRITICAL ERROR] Class mismatch ({probs_f.shape[1]} vs {probs_o.shape[1]}).")
        return

    # -------------------------------------------------------
    # 4. SYNTHETIC PAIRING & FUSION
    # -------------------------------------------------------
    print("\n[Step 4] Running Weighted Fusion...")
    
    y_true_paired = []
    y_pred_fundus = []
    y_pred_oct = []
    y_pred_fused = []

    # Iterate through model classes (0 to 6)
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

        # Fusion: 30% Fundus, 70% OCT
        p_fused = (0.3 * p_f) + (0.7 * p_o)

        y_true_paired.extend([cls] * n_pairs)
        y_pred_fundus.extend(np.argmax(p_f, axis=1))
        y_pred_oct.extend(np.argmax(p_o, axis=1))
        y_pred_fused.extend(np.argmax(p_fused, axis=1))

    # Convert to arrays
    y_true = np.array(y_true_paired)
    y_pred_f = np.array(y_pred_fundus)
    y_pred_o = np.array(y_pred_oct)
    y_pred_fusion = np.array(y_pred_fused)

    # -------------------------------------------------------
    # 5. RESULTS
    # -------------------------------------------------------
    print("\n[Step 5] Final Fusion Results")
    print("-" * 40)
    print(f"Fundus Only Accuracy: {accuracy_score(y_true, y_pred_f):.4f}")
    print(f"OCT Only Accuracy:    {accuracy_score(y_true, y_pred_o):.4f}")
    print(f"FUSED ACCURACY:       {accuracy_score(y_true, y_pred_fusion):.4f}")
    print("-" * 40)

    print("\nClassification Report (Fused System):")
    # Use target_names to make the report readable
    target_names = ['Normal', 'Class 1', 'Class 3', 'Class 4', 'Class 6', 'Class 7', 'Class 8']
    print(classification_report(y_true, y_pred_fusion, digits=4, target_names=target_names))

    cm = confusion_matrix(y_true, y_pred_fusion)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Fused System Confusion Matrix\nAccuracy: {accuracy_score(y_true, y_pred_fusion):.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(PLOT_OUTPUT)
    print(f"\nPlot saved to {PLOT_OUTPUT}")

if __name__ == "__main__":
    main()