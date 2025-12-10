"""
Script to train models using pre-extracted and selected features.
Skips feature extraction and selection steps.
"""
import os
import numpy as np
import pickle
import argparse
from pathlib import Path
import time

from feature_selector import FeatureSelector
from model_trainer import ModelTrainer


def load_features(filepath: str):
    """Load features from disk."""
    data = np.load(filepath, allow_pickle=True)
    features = data['features']
    feature_names = data['feature_names'].tolist()
    return features, feature_names


def load_labels(filepath: str):
    """Load labels from disk."""
    return np.load(filepath)


def main():
    parser = argparse.ArgumentParser(description='Train models using pre-extracted features')
    parser.add_argument('--train_features_file', type=str,
                       default='features/train_features.npz',
                       help='Path to training features file')
    parser.add_argument('--train_labels_file', type=str,
                       default='features/train_features_labels.npy',
                       help='Path to training labels file')
    parser.add_argument('--test_features_file', type=str,
                       default='features/test_features.npz',
                       help='Path to test features file (optional)')
    parser.add_argument('--test_labels_file', type=str,
                       default='features/test_features_labels.npy',
                       help='Path to test labels file (optional)')
    parser.add_argument('--selected_features_file', type=str,
                       default=None,
                       help='Path to saved feature selector (optional, will re-select if not provided)')
    parser.add_argument('--n_features', type=int, default=50,
                       help='Number of features to select (if re-selecting)')
    parser.add_argument('--feature_selection_method', type=str, default='rfe',
                       choices=['f_test', 'mutual_info', 'rfe', 'rf_importance', 'combined'],
                       help='Feature selection method (default: rfe for correlated radiomics features)')
    parser.add_argument('--imbalance_method', type=str, default='none',
                       choices=['smote', 'adasyn', 'oversample', 'undersample', 
                               'smote_tomek', 'smote_enn', 'balanced_rf', 'none'],
                       help='Method to handle class imbalance (use "none" for class weighting)')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--save_model', type=str, default='models/best_model.pkl',
                       help='Path to save the best model')
    parser.add_argument('--skip_feature_selection', action='store_true',
                       help='Skip feature selection and use all features')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Model Training (Using Pre-extracted Features)")
    print("="*70)
    print(f"Training features: {args.train_features_file}")
    print(f"Feature selection: {args.feature_selection_method}")
    print(f"Imbalance handling: {args.imbalance_method}")
    print("="*70)
    
    # Step 1: Load training features and labels
    print("\n[Step 1] Loading training features and labels...")
    start_time = time.time()
    
    if not os.path.exists(args.train_features_file):
        raise FileNotFoundError(f"Training features file not found: {args.train_features_file}")
    
    train_features, feature_names = load_features(args.train_features_file)
    
    # Load labels
    if os.path.exists(args.train_labels_file):
        train_labels = load_labels(args.train_labels_file)
    else:
        # Try alternative location
        alt_labels_file = args.train_features_file.replace('.npz', '_labels.npy')
        if os.path.exists(alt_labels_file):
            train_labels = load_labels(alt_labels_file)
        else:
            raise FileNotFoundError(f"Labels file not found. Tried: {args.train_labels_file} and {alt_labels_file}")
    
    # Filter out classes 2 and 5
    print(f"\nOriginal class distribution: {np.bincount(train_labels)}")
    exclude_classes = [2, 5]
    mask = ~np.isin(train_labels, exclude_classes)
    train_features = train_features[mask]
    train_labels = train_labels[mask]
    
    # Remap labels to be consecutive (0, 1, 3, 4, 6, 7, 8 -> 0, 1, 2, 3, 4, 5, 6)
    unique_labels = np.unique(train_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
    train_labels = np.array([label_mapping[label] for label in train_labels])
    
    print(f"After excluding classes {exclude_classes}:")
    print(f"  Removed {np.sum(~mask)} samples")
    print(f"  Remaining: {len(train_labels)} samples")
    print(f"  Class distribution: {np.bincount(train_labels)}")
    print(f"  Label mapping: {label_mapping}")
    print(f"Loaded {train_features.shape[0]} samples with {train_features.shape[1]} features")
    print(f"Loading completed in {time.time() - start_time:.2f}s")
    
    # Step 2: Feature selection (or load pre-selected)
    if args.skip_feature_selection:
        print("\n[Step 2] Skipping feature selection - using all features")
        train_features_selected = train_features
        selected_feature_names = feature_names
        feature_selector = None
    elif args.selected_features_file and os.path.exists(args.selected_features_file):
        print(f"\n[Step 2] Loading pre-selected features from {args.selected_features_file}")
        with open(args.selected_features_file, 'rb') as f:
            feature_selector = pickle.load(f)
        train_features_selected = feature_selector.transform(train_features)
        selected_feature_names = feature_selector.selected_feature_names if hasattr(feature_selector, 'selected_feature_names') else feature_names
    else:
        print("\n[Step 2] Selecting features...")
        start_time = time.time()
        
        feature_selector = FeatureSelector(
            n_features=args.n_features,
            method=args.feature_selection_method
        )
        
        train_features_selected, selected_feature_names = feature_selector.select_features(
            train_features,
            train_labels,
            feature_names
        )
        
        print(f"Selected {len(selected_feature_names)} features in {time.time() - start_time:.2f}s")
        
        # Save feature selector for future use
        if args.selected_features_file:
            os.makedirs(os.path.dirname(args.selected_features_file), exist_ok=True)
            with open(args.selected_features_file, 'wb') as f:
                pickle.dump(feature_selector, f)
            print(f"Saved feature selector to {args.selected_features_file}")
    
    # Step 3: Load test data if available
    test_features_selected = None
    test_labels = None
    
    if args.test_features_file and os.path.exists(args.test_features_file):
        print("\n[Step 3] Loading test features...")
        test_features, _ = load_features(args.test_features_file)
        
        # Load test labels
        if os.path.exists(args.test_labels_file):
            test_labels = load_labels(args.test_labels_file)
        else:
            alt_test_labels = args.test_features_file.replace('.npz', '_labels.npy')
            if os.path.exists(alt_test_labels):
                test_labels = load_labels(alt_test_labels)
            else:
                print(f"Warning: Test labels not found. Skipping test evaluation.")
                test_features = None
        
        if test_features is not None:
            # Filter out classes 2 and 5 from test set
            print(f"\nOriginal test class distribution: {np.bincount(test_labels)}")
            exclude_classes = [2, 5]
            test_mask = ~np.isin(test_labels, exclude_classes)
            test_features = test_features[test_mask]
            test_labels = test_labels[test_mask]
            
            # Remap test labels using same mapping as training
            test_labels = np.array([label_mapping.get(label, -1) for label in test_labels])
            # Remove any labels that weren't in training (shouldn't happen, but safety check)
            valid_test_mask = test_labels >= 0
            test_features = test_features[valid_test_mask]
            test_labels = test_labels[valid_test_mask]
            
            print(f"After excluding classes {exclude_classes} from test:")
            print(f"  Removed {np.sum(~test_mask)} samples")
            print(f"  Remaining: {len(test_labels)} test samples")
            print(f"  Test class distribution: {np.bincount(test_labels)}")
            
            if feature_selector is not None:
                test_features_selected = feature_selector.transform(test_features)
            else:
                test_features_selected = test_features
            print(f"Test set: {test_features.shape[0]} images")
    
    # Step 4: Train models and select best
    print("\n[Step 4] Training models...")
    start_time = time.time()
    
    trainer = ModelTrainer(
        imbalance_method=args.imbalance_method,
        cv_folds=args.cv_folds
    )
    
    # Train and evaluate
    results = trainer.train_and_evaluate(
        train_features_selected,
        train_labels,
        test_features_selected,
        test_labels
    )
    
    print(f"\nModel training completed in {time.time() - start_time:.2f}s")
    
    # Step 5: Save model
    print("\n[Step 5] Saving model...")
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    
    model_data = {
        'model': results['best_model'],
        'model_name': results['best_model_name'],
        'feature_selector': feature_selector,
        'selected_feature_names': selected_feature_names if 'selected_feature_names' in locals() else feature_names,
        'cv_score': results['cv_score'],
        'test_metrics': results['test_metrics']
    }
    
    with open(args.save_model, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {args.save_model}")
    
    # Summary
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"Best Model: {results['best_model_name']}")
    print(f"CV F1-macro Score: {results['cv_score']:.4f}")
    if results['test_metrics']:
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1-macro: {results['test_metrics']['f1_macro']:.4f}")
        print(f"Test F1-weighted: {results['test_metrics']['f1_weighted']:.4f}")
        print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
        print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    print(f"Selected Features: {len(selected_feature_names) if 'selected_feature_names' in locals() else len(feature_names)}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()

