"""
Main pipeline for fundus image classification using radiomics features.
Orchestrates data loading, feature extraction, feature selection, and model training.
"""
import os
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Tuple, List
import time

from data_loader import (
    load_images_and_labels, 
    load_test_data,
    get_image_paths_and_labels,
    load_images_batch
)
from radiomics_extractor import RadiomicsExtractor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer


def save_features(features: np.ndarray, feature_names: list, filepath: str):
    """Save extracted features to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        features=features,
        feature_names=feature_names
    )
    print(f"Saved features to {filepath}")


def load_features(filepath: str):
    """Load features from disk."""
    data = np.load(filepath, allow_pickle=True)
    features = data['features']
    feature_names = data['feature_names'].tolist()
    return features, feature_names


def extract_features_batch_mode(
    image_path_label_pairs: List[tuple],
    extractor: RadiomicsExtractor,
    batch_size: int = 50,
    features_file: str = None
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Extract features in batches to save memory.
    
    Args:
        image_path_label_pairs: List of (image_path, label) tuples
        extractor: Feature extractor instance
        batch_size: Number of images to process per batch
        features_file: Optional file to save features incrementally
        
    Returns:
        Tuple of (features, feature_names, labels)
    """
    all_features = []
    all_labels = []
    feature_names = None
    
    total_images = len(image_path_label_pairs)
    print(f"Processing {total_images} images in batches of {batch_size}...")
    
    # Process in batches
    for batch_idx in range(0, total_images, batch_size):
        batch_pairs = image_path_label_pairs[batch_idx:batch_idx + batch_size]
        batch_paths = [path for path, _ in batch_pairs]
        batch_labels = [label for _, label in batch_pairs]
        
        # Load images for this batch
        batch_images = []
        valid_indices = []
        for i, img_path in enumerate(batch_paths):
            try:
                from PIL import Image
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                batch_images.append(img_array)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Extract features for this batch
        batch_features, batch_feature_names = extractor.extract_features_batch(
            batch_images,
            verbose=False
        )
        
        # Store feature names from first batch
        if feature_names is None:
            feature_names = batch_feature_names
        
        # Ensure feature dimensions match
        if batch_features.shape[1] != len(feature_names):
            # Pad or truncate if needed
            if batch_features.shape[1] < len(feature_names):
                padding = np.zeros((batch_features.shape[0], len(feature_names) - batch_features.shape[1]))
                batch_features = np.hstack([batch_features, padding])
            else:
                batch_features = batch_features[:, :len(feature_names)]
        
        all_features.append(batch_features)
        all_labels.extend([batch_labels[i] for i in valid_indices])
        
        # Print progress
        processed = min(batch_idx + batch_size, total_images)
        print(f"Processed {processed}/{total_images} images ({100*processed/total_images:.1f}%)")
        
        # Optional: Save incrementally
        if features_file and (batch_idx + batch_size) % (batch_size * 10) == 0:
            # Save checkpoint every 10 batches
            combined_features = np.vstack(all_features)
            labels_array = np.array(all_labels)
            save_features(combined_features, feature_names, features_file + '.checkpoint')
    
    # Combine all features
    if all_features:
        combined_features = np.vstack(all_features)
        labels_array = np.array(all_labels)
    else:
        combined_features = np.array([])
        labels_array = np.array([])
    
    print(f"Extracted {combined_features.shape[1]} features from {combined_features.shape[0]} images")
    
    return combined_features, feature_names, labels_array


def main():
    parser = argparse.ArgumentParser(description='Fundus Image Classification Pipeline')
    parser.add_argument('--data_dir', type=str, 
                       default='multieye_data/Fundus',
                       help='Directory containing train, dev, test folders and ImageData')
    parser.add_argument('--image_dir', type=str,
                       default='multieye_data/Fundus/ImageData/cfp-clahe-224x224',
                       help='Directory containing fundus images')
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract radiomics features (skip if already extracted)')
    parser.add_argument('--features_file', type=str,
                       default='features/train_features.npz',
                       help='Path to save/load extracted features')
    parser.add_argument('--test_features_file', type=str,
                       default='features/test_features.npz',
                       help='Path to save/load test features')
    parser.add_argument('--n_features', type=int, default=100,
                       help='Number of features to select')
    parser.add_argument('--feature_selection_method', type=str, default='combined',
                       choices=['f_test', 'mutual_info', 'rfe', 'rf_importance', 'combined'],
                       help='Feature selection method')
    parser.add_argument('--imbalance_method', type=str, default='smote',
                       choices=['smote', 'adasyn', 'oversample', 'undersample', 
                               'smote_tomek', 'smote_enn', 'balanced_rf', 'none'],
                       help='Method to handle class imbalance')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--save_model', type=str, default='models/best_model.pkl',
                       help='Path to save the best model')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Batch size for loading and processing images (default: 50)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir)
    train_label_file = data_dir / 'train' / 'large9cls.txt'
    dev_label_file = data_dir / 'dev' / 'large9cls.txt'
    test_label_file = data_dir / 'test' / 'large9cls.txt'
    
    print("="*70)
    print("Fundus Image Classification Pipeline")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Image directory: {image_dir}")
    print(f"Feature selection: {args.feature_selection_method}")
    print(f"Imbalance handling: {args.imbalance_method}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    # Step 1: Get image paths and labels (without loading images)
    print("\n[Step 1] Getting image paths and labels...")
    start_time = time.time()
    
    image_path_label_pairs, label_mapping = get_image_paths_and_labels(
        str(train_label_file),
        str(dev_label_file),
        str(image_dir)
    )
    
    print(f"Found {len(image_path_label_pairs)} images in {time.time() - start_time:.2f}s")
    
    # Step 2: Extract radiomics features in batches
    print("\n[Step 2] Extracting radiomics features (batch mode)...")
    
    features_file = Path(args.features_file)
    if features_file.exists() and not args.extract_features:
        print(f"Loading pre-extracted features from {features_file}")
        train_features, feature_names = load_features(str(features_file))
        # Get labels - we need to match the order from when features were extracted
        # For now, we'll extract labels from the saved order or reconstruct
        # If labels were saved separately, load them; otherwise reconstruct from paths
        labels_file = str(features_file).replace('.npz', '_labels.npy')
        if os.path.exists(labels_file):
            train_labels = np.load(labels_file)
        else:
            # Reconstruct labels from image paths (approximate - may not match exactly)
            print("Warning: Labels file not found. Reconstructing from image paths...")
            image_paths = [path for path, _ in image_path_label_pairs]
            train_labels = np.array([label_mapping.get(path, 0) for path in image_paths[:len(train_features)]])
    else:
        print("Extracting features from images in batches...")
        start_time = time.time()
        
        extractor = RadiomicsExtractor()
        
        # Extract features in batches
        train_features, feature_names, train_labels = extract_features_batch_mode(
            image_path_label_pairs,
            extractor,
            batch_size=args.batch_size,
            features_file=str(features_file)
        )
        
        print(f"Feature extraction completed in {time.time() - start_time:.2f}s")
        print(f"Extracted {train_features.shape[1]} features from {train_features.shape[0]} images")
        
        # Save features and labels
        save_features(train_features, feature_names, str(features_file))
        # Also save labels separately for easy loading
        labels_file = str(features_file).replace('.npz', '_labels.npy')
        np.save(labels_file, train_labels)
        print(f"Saved labels to {labels_file}")
    
    # Step 3: Feature selection
    print("\n[Step 3] Selecting features...")
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
    print(f"Selected features: {selected_feature_names[:10]}..." if len(selected_feature_names) > 10 else f"Selected features: {selected_feature_names}")
    
    # Step 4: Train models and select best
    print("\n[Step 4] Training models...")
    start_time = time.time()
    
    trainer = ModelTrainer(
        imbalance_method=args.imbalance_method,
        cv_folds=args.cv_folds
    )
    
    # Load test data if available
    test_features_selected = None
    test_labels = None
    
    if test_label_file.exists():
        print("\n[Step 4.5] Processing test data...")
        from data_loader import load_labels, find_image_file
        from PIL import Image
        
        test_labels_dict = load_labels(str(test_label_file))
        test_path_label_pairs = []
        
        for img_name, label in test_labels_dict.items():
            img_path = find_image_file(img_name, str(image_dir))
            if img_path:
                test_path_label_pairs.append((img_path, label))
        
        if len(test_path_label_pairs) > 0:
            # Extract test features in batches
            test_features_file = Path(args.test_features_file)
            if test_features_file.exists() and not args.extract_features:
                print(f"Loading pre-extracted test features from {test_features_file}")
                test_features, _ = load_features(str(test_features_file))
                test_labels = np.array([label for _, label in test_path_label_pairs[:len(test_features)]])
            else:
                print("Extracting test features in batches...")
                extractor = RadiomicsExtractor()
                test_features, _, test_labels = extract_features_batch_mode(
                    test_path_label_pairs,
                    extractor,
                    batch_size=args.batch_size,
                    features_file=str(test_features_file)
                )
                save_features(test_features, feature_names, str(test_features_file))
            
            # Transform test features using selected features
            test_features_selected = feature_selector.transform(test_features)
            print(f"Test set: {len(test_path_label_pairs)} images")
    
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
        'selected_feature_names': selected_feature_names,
        'cv_score': results['cv_score'],
        'test_metrics': results['test_metrics']
    }
    
    with open(args.save_model, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {args.save_model}")
    
    # Summary
    print("\n" + "="*70)
    print("Pipeline Summary")
    print("="*70)
    print(f"Best Model: {results['best_model_name']}")
    print(f"CV F1-macro Score: {results['cv_score']:.4f}")
    if results['test_metrics']:
        print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Test F1-macro: {results['test_metrics']['f1_macro']:.4f}")
    print(f"Selected Features: {len(selected_feature_names)}")
    print("="*70)
    
    return results


if __name__ == '__main__':
    main()


