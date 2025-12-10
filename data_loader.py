"""
Data loading module for fundus image dataset.
Loads images and labels from train and dev sets.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import glob


def load_labels(label_file: str) -> Dict[str, int]:
    """
    Load labels from a text file.
    Format: image_name label
    Handles UTF-8 encoding for Chinese characters in image names.
    
    Args:
        label_file: Path to the label file
        
    Returns:
        Dictionary mapping image names to labels
    """
    labels = {}
    # Try UTF-8 first (for Chinese characters), fallback to default encoding
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', None]  # Chinese encodings + default
    
    for encoding in encodings:
        try:
            with open(label_file, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.rsplit(' ', 1)  # Split from right to handle names with spaces
                    if len(parts) == 2:
                        img_name, label = parts
                        # Remove trailing period if present (e.g., "1ffa9600-8d87-11e8-9daf-6045cb817f5b. 7")
                        img_name = img_name.rstrip('.')
                        labels[img_name] = int(label)
            break  # Success, exit loop
        except (UnicodeDecodeError, UnicodeError):
            continue  # Try next encoding
        except Exception as e:
            if encoding == encodings[-1]:  # Last encoding failed
                raise e
            continue
    
    return labels


def find_image_file(image_name: str, image_dir: str) -> str:
    """
    Find the actual image file given the image name.
    Tries multiple extensions and naming patterns.
    Handles UTF-8 encoded paths (for Chinese characters).
    
    Args:
        image_name: Name of the image (from label file)
        image_dir: Directory containing images
        
    Returns:
        Path to the image file, or None if not found
    """
    # Clean image name (remove trailing periods, spaces)
    image_name = image_name.strip().rstrip('.')
    
    # Try different extensions
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    # Try exact match first
    for ext in extensions:
        try:
            path = os.path.join(image_dir, image_name + ext)
            if os.path.exists(path):
                return path
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Handle encoding issues with file paths
            pass
    
    # Try case-insensitive search (handle Unicode)
    try:
        image_name_lower = image_name.lower()
        for ext in extensions:
            # Use glob with proper encoding handling
            pattern = os.path.join(image_dir, f"*{image_name_lower}*{ext}")
            try:
                matches = glob.glob(pattern, recursive=False)
                if matches:
                    return matches[0]
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    except Exception:
        pass
    
    # Try with original case
    for ext in extensions:
        pattern = os.path.join(image_dir, f"*{image_name}*{ext}")
        try:
            matches = glob.glob(pattern, recursive=False)
            if matches:
                return matches[0]
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    
    # Try without extension (in case image name already has extension)
    try:
        path = os.path.join(image_dir, image_name)
        if os.path.exists(path):
            return path
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    
    return None


def load_images_and_labels(
    train_label_file: str,
    dev_label_file: str,
    image_dir: str
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Load images and labels from train and dev sets.
    
    Args:
        train_label_file: Path to training labels file
        dev_label_file: Path to dev labels file
        image_dir: Directory containing images
        
    Returns:
        Tuple of (images, labels, image_paths)
        - images: List of image arrays
        - labels: Array of labels
        - image_paths: List of image file paths
    """
    # Load labels
    train_labels = load_labels(train_label_file)
    dev_labels = load_labels(dev_label_file)
    
    # Combine train and dev labels
    all_labels = {**train_labels, **dev_labels}
    
    images = []
    labels = []
    image_paths = []
    not_found = []
    
    print(f"Total images in labels: {len(all_labels)}")
    print(f"Searching for images in: {image_dir}")
    
    for img_name, label in all_labels.items():
        img_path = find_image_file(img_name, image_dir)
        
        if img_path is None:
            not_found.append(img_name)
            continue
        
        try:
            # Load image
            img = Image.open(img_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert to numpy array
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            not_found.append(img_name)
            continue
    
    print(f"Successfully loaded {len(images)} images")
    print(f"Failed to find {len(not_found)} images")
    if len(not_found) > 0 and len(not_found) <= 10:
        print(f"Sample not found: {not_found[:5]}")
    
    labels_array = np.array(labels)
    
    # Print class distribution
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} images")
    
    return images, labels_array, image_paths


def load_test_data(test_label_file: str, image_dir: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Load test images and labels.
    
    Args:
        test_label_file: Path to test labels file
        image_dir: Directory containing images
        
    Returns:
        Tuple of (images, labels, image_paths)
    """
    test_labels = load_labels(test_label_file)
    
    images = []
    labels = []
    image_paths = []
    not_found = []
    
    print(f"Loading test data: {len(test_labels)} images")
    
    for img_name, label in test_labels.items():
        img_path = find_image_file(img_name, image_dir)
        
        if img_path is None:
            not_found.append(img_name)
            continue
        
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            images.append(img_array)
            labels.append(label)
            image_paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            not_found.append(img_name)
            continue
    
    print(f"Successfully loaded {len(images)} test images")
    if len(not_found) > 0:
        print(f"Failed to find {len(not_found)} test images")
    
    labels_array = np.array(labels)
    
    return images, labels_array, image_paths


def get_image_paths_and_labels(
    train_label_file: str,
    dev_label_file: str,
    image_dir: str
) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """
    Get image paths and labels without loading images into memory.
    
    Args:
        train_label_file: Path to training labels file
        dev_label_file: Path to dev labels file
        image_dir: Directory containing images
        
    Returns:
        Tuple of (image_path_label_pairs, label_mapping)
        - image_path_label_pairs: List of (image_path, label) tuples
        - label_mapping: Dictionary mapping image paths to labels
    """
    # Load labels
    train_labels = load_labels(train_label_file)
    dev_labels = load_labels(dev_label_file)
    
    # Combine train and dev labels
    all_labels = {**train_labels, **dev_labels}
    
    image_path_label_pairs = []
    label_mapping = {}
    not_found = []
    
    print(f"Total images in labels: {len(all_labels)}")
    print(f"Searching for images in: {image_dir}")
    
    for img_name, label in all_labels.items():
        img_path = find_image_file(img_name, image_dir)
        
        if img_path is None:
            not_found.append(img_name)
            continue
        
        image_path_label_pairs.append((img_path, label))
        label_mapping[img_path] = label
    
    print(f"Found {len(image_path_label_pairs)} images")
    print(f"Failed to find {len(not_found)} images")
    if len(not_found) > 0 and len(not_found) <= 10:
        print(f"Sample not found: {not_found[:5]}")
    
    # Print class distribution
    labels_only = [label for _, label in image_path_label_pairs]
    labels_array = np.array(labels_only)
    unique, counts = np.unique(labels_array, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} images")
    
    return image_path_label_pairs, label_mapping


def load_images_batch(image_paths: List[str], batch_size: int = 50):
    """
    Generator that loads images in batches to save memory.
    
    Args:
        image_paths: List of image file paths
        batch_size: Number of images to load per batch
        
    Yields:
        Tuple of (batch_images, batch_paths)
    """
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        valid_paths = []
        
        for img_path in batch_paths:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
                batch_images.append(img_array)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if batch_images:
            yield batch_images, valid_paths

