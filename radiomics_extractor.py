"""
Radiomics feature extraction module for fundus images.
Uses pyradiomics to extract comprehensive radiomics features.
Falls back to alternative extractor if pyradiomics is not available.
"""
import numpy as np
from typing import List, Dict
import os

# Try to import pyradiomics, fall back to alternative if not available
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    PYRADIOMICS_AVAILABLE = True
except ImportError:
    PYRADIOMICS_AVAILABLE = False
    print("Warning: pyradiomics not available. Using alternative feature extraction.")


class RadiomicsExtractor:
    """
    Extracts radiomics features from fundus images.
    Uses pyradiomics if available, otherwise falls back to alternative extractor.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the radiomics extractor.
        
        Args:
            config_path: Optional path to pyradiomics configuration file
        """
        self.use_pyradiomics = PYRADIOMICS_AVAILABLE
        
        if self.use_pyradiomics:
            # Create extractor with appropriate settings for fundus images
            if config_path and os.path.exists(config_path):
                self.extractor = featureextractor.RadiomicsFeatureExtractor(config_path)
            else:
                # Use default settings optimized for fundus images
                self.extractor = featureextractor.RadiomicsFeatureExtractor()
                
                # Configure for fundus images (2D, RGB)
                # We'll extract features from grayscale conversion
                self.extractor.settings.update({
                    'binWidth': 25,  # Binning width for intensity discretization
                    'interpolator': sitk.sitkBSpline,
                    'resampledPixelSpacing': None,  # Use original spacing
                    'label': 1,  # Label value for mask (we'll create a full mask)
                })
                
                # Enable specific feature classes relevant for fundus images
                self.extractor.enableAllFeatures()
        else:
            # Use alternative extractor
            from feature_extractor_alternative import AlternativeFeatureExtractor
            self.extractor = AlternativeFeatureExtractor()
    
    def extract_features_from_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract radiomics features from a single image.
        
        Args:
            image: Image array (H, W, 3) or (H, W)
            
        Returns:
            Dictionary of feature names and values
        """
        if self.use_pyradiomics:
            # Use pyradiomics
            # Convert to grayscale if RGB
            if len(image.shape) == 3:
                # Use luminance-weighted conversion
                gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = image
            
            # Convert to uint8 if needed
            if gray.dtype != np.uint8:
                gray = (gray * 255 / gray.max()).astype(np.uint8)
            
            # Convert numpy array to SimpleITK Image
            sitk_image = sitk.GetImageFromArray(gray)
            
            # Create a mask covering the entire image (for 2D images)
            mask_array = np.ones_like(gray, dtype=np.uint8)
            sitk_mask = sitk.GetImageFromArray(mask_array)
            
            try:
                # Extract features
                features = self.extractor.execute(sitk_image, sitk_mask)
                
                # Filter out diagnostic features and keep only feature values
                feature_dict = {}
                for key, value in features.items():
                    if not key.startswith('diagnostics'):
                        # Handle NaN and inf values
                        if isinstance(value, (int, float)):
                            if np.isnan(value) or np.isinf(value):
                                feature_dict[key] = 0.0
                            else:
                                feature_dict[key] = float(value)
                        else:
                            feature_dict[key] = 0.0
                
                return feature_dict
            except Exception as e:
                print(f"Error extracting features: {e}")
                # Return empty dict or default features
                return {}
        else:
            # Use alternative extractor
            return self.extractor.extract_features_from_image(image)
    
    def extract_features_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> tuple:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of image arrays
            verbose: Whether to print progress
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if not self.use_pyradiomics:
            # Use alternative extractor's batch method
            return self.extractor.extract_features_batch(images, verbose)
        
        # Use pyradiomics batch processing
        all_features = []
        
        for i, image in enumerate(images):
            if verbose and (i + 1) % 100 == 0:
                print(f"Extracting features from image {i+1}/{len(images)}")
            
            features = self.extract_features_from_image(image)
            all_features.append(features)
        
        # Convert to feature matrix
        # Get all unique feature names
        all_feature_names = set()
        for feat_dict in all_features:
            all_feature_names.update(feat_dict.keys())
        
        all_feature_names = sorted(list(all_feature_names))
        
        # Create feature matrix
        feature_matrix = np.zeros((len(images), len(all_feature_names)))
        
        for i, feat_dict in enumerate(all_features):
            for j, feat_name in enumerate(all_feature_names):
                feature_matrix[i, j] = feat_dict.get(feat_name, 0.0)
        
        if verbose:
            print(f"Extracted {len(all_feature_names)} features from {len(images)} images")
        
        return feature_matrix, all_feature_names
    
    def extract_multichannel_features(
        self,
        images: List[np.ndarray],
        verbose: bool = True
    ) -> np.ndarray:
        """
        Extract features from RGB channels separately and combine.
        This can capture color-specific patterns in fundus images.
        
        Args:
            images: List of RGB image arrays
            verbose: Whether to print progress
            
        Returns:
            Feature matrix (n_samples, n_features * 3)
        """
        if len(images) == 0:
            return np.array([]), []
        
        # Check if images are RGB
        if len(images[0].shape) != 3 or images[0].shape[2] != 3:
            # If not RGB, use standard extraction
            return self.extract_features_batch(images, verbose)
        
        # Extract features from each channel
        channel_features = []
        channel_names = []
        
        for channel_idx in range(3):
            if verbose:
                print(f"Extracting features from channel {channel_idx + 1}/3")
            
            channel_images = [img[:, :, channel_idx] for img in images]
            features, names = self.extract_features_batch(channel_images, verbose=False)
            
            # Prefix feature names with channel
            prefixed_names = [f"ch{channel_idx}_{name}" for name in names]
            channel_features.append(features)
            channel_names.extend(prefixed_names)
        
        # Combine features
        combined_features = np.hstack(channel_features)
        
        if verbose:
            print(f"Extracted {combined_features.shape[1]} features from {len(images)} images")
        
        return combined_features, channel_names

