"""
Script to create test labels file from test label text file.
"""
import numpy as np
from data_loader import load_labels, find_image_file
from pathlib import Path

def main():
    # Paths
    data_dir = Path('multieye_data/Fundus')
    image_dir = Path('multieye_data/Fundus/ImageData/cfp-clahe-224x224')
    test_label_file = data_dir / 'test' / 'large9cls.txt'
    test_features_file = 'features/test_features.npz'
    output_labels_file = 'features/test_features_labels.npy'
    
    # Load test features to get count
    import numpy as np
    data = np.load(test_features_file, allow_pickle=True)
    n_test_samples = data['features'].shape[0]
    print(f"Found {n_test_samples} test samples in features file")
    
    # Load test labels
    test_labels_dict = load_labels(str(test_label_file))
    print(f"Loaded {len(test_labels_dict)} labels from {test_label_file}")
    
    # Match labels to features (in order they were extracted)
    # We need to find images in the same order
    test_labels = []
    found_count = 0
    
    for img_name, label in test_labels_dict.items():
        img_path = find_image_file(img_name, str(image_dir))
        if img_path:
            test_labels.append(label)
            found_count += 1
            if found_count >= n_test_samples:
                break
    
    if len(test_labels) != n_test_samples:
        print(f"Warning: Found {len(test_labels)} labels but need {n_test_samples}")
        print("This might happen if some images weren't found during feature extraction")
        # Pad or truncate to match
        if len(test_labels) < n_test_samples:
            print(f"Padding with zeros to match {n_test_samples} samples")
            test_labels.extend([0] * (n_test_samples - len(test_labels)))
        else:
            test_labels = test_labels[:n_test_samples]
    
    test_labels_array = np.array(test_labels)
    
    # Save labels
    np.save(output_labels_file, test_labels_array)
    print(f"Saved {len(test_labels_array)} test labels to {output_labels_file}")
    print(f"Class distribution: {np.bincount(test_labels_array)}")

if __name__ == '__main__':
    main()

