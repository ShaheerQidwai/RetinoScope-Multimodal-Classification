"""
Alternative feature extraction module for fundus images.
Uses scikit-image and other standard libraries instead of pyradiomics.
Extracts texture, statistical, and morphological features.
"""
import numpy as np
from skimage import feature, filters, measure, morphology
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy import stats, ndimage
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class AlternativeFeatureExtractor:
    """
    Extracts comprehensive features from fundus images using standard libraries.
    Includes texture, statistical, morphological, and frequency domain features.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_features_from_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a single image.
        
        Args:
            image: Image array (H, W, 3) or (H, W)
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            # Use luminance-weighted conversion
            gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        else:
            gray = image.copy()
        
        # Normalize to 0-255 uint8
        if gray.dtype != np.uint8:
            gray_min, gray_max = gray.min(), gray.max()
            if gray_max > gray_min:
                gray = ((gray - gray_min) / (gray_max - gray_min) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)
        
        # 1. Statistical features
        features.update(self._statistical_features(gray))
        
        # 2. Texture features (GLCM)
        features.update(self._glcm_features(gray))
        
        # 3. Local Binary Pattern (LBP) features
        features.update(self._lbp_features(gray))
        
        # 4. Gabor filter features
        features.update(self._gabor_features(gray))
        
        # 5. Histogram features
        features.update(self._histogram_features(gray))
        
        # 6. Edge features
        features.update(self._edge_features(gray))
        
        # 7. Morphological features
        features.update(self._morphological_features(gray))
        
        # 8. Frequency domain features (if image is large enough)
        if gray.shape[0] > 32 and gray.shape[1] > 32:
            features.update(self._frequency_features(gray))
        
        # 9. Color features (if RGB)
        if len(image.shape) == 3:
            features.update(self._color_features(image))
        
        return features
    
    def _statistical_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract statistical features."""
        features = {}
        flat = gray.flatten()
        
        features['mean'] = float(np.mean(flat))
        features['std'] = float(np.std(flat))
        features['var'] = float(np.var(flat))
        features['min'] = float(np.min(flat))
        features['max'] = float(np.max(flat))
        features['median'] = float(np.median(flat))
        features['skewness'] = float(stats.skew(flat))
        features['kurtosis'] = float(stats.kurtosis(flat))
        features['entropy'] = float(stats.entropy(np.histogram(flat, bins=256)[0] + 1e-10))
        
        # Percentiles
        features['p10'] = float(np.percentile(flat, 10))
        features['p25'] = float(np.percentile(flat, 25))
        features['p75'] = float(np.percentile(flat, 75))
        features['p90'] = float(np.percentile(flat, 90))
        
        # Range
        features['range'] = features['max'] - features['min']
        features['iqr'] = features['p75'] - features['p25']
        
        return features
    
    def _glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Gray-Level Co-occurrence Matrix (GLCM) features."""
        features = {}
        
        # Downsample if too large for efficiency
        if gray.shape[0] > 256 or gray.shape[1] > 256:
            from skimage.transform import resize
            gray_small = resize(gray, (256, 256), anti_aliasing=True)
            gray_small = (gray_small * 255).astype(np.uint8)
        else:
            gray_small = gray
        
        # Quantize to fewer levels for GLCM
        gray_quantized = (gray_small // 32).astype(np.uint8)  # 8 levels
        
        try:
            # Calculate GLCM for different angles
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            for dist in distances:
                for angle in angles:
                    glcm = graycomatrix(
                        gray_quantized,
                        distances=[dist],
                        angles=[angle],
                        levels=8,
                        symmetric=True,
                        normed=True
                    )
                    
                    # Extract properties
                    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
                    for prop in props:
                        try:
                            value = graycoprops(glcm, prop)[0, 0]
                            features[f'glcm_{prop}_d{dist}_a{int(np.degrees(angle))}'] = float(value)
                        except:
                            features[f'glcm_{prop}_d{dist}_a{int(np.degrees(angle))}'] = 0.0
        except Exception as e:
            # If GLCM fails, add zeros
            pass
        
        return features
    
    def _lbp_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Local Binary Pattern (LBP) features."""
        features = {}
        
        try:
            # Calculate LBP
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Histogram of LBP
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-10)
            
            # Statistical features of LBP histogram
            features['lbp_mean'] = float(np.mean(lbp))
            features['lbp_std'] = float(np.std(lbp))
            features['lbp_entropy'] = float(stats.entropy(hist + 1e-10))
            
            # Uniformity (percentage of uniform patterns)
            n_uniform = np.sum(hist[:n_points + 1])
            features['lbp_uniformity'] = float(n_uniform)
        except:
            features['lbp_mean'] = 0.0
            features['lbp_std'] = 0.0
            features['lbp_entropy'] = 0.0
            features['lbp_uniformity'] = 0.0
        
        return features
    
    def _gabor_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract Gabor filter features."""
        features = {}
        
        try:
            # Downsample for efficiency
            if gray.shape[0] > 128 or gray.shape[1] > 128:
                from skimage.transform import resize
                gray_small = resize(gray, (128, 128), anti_aliasing=True)
                gray_small = (gray_small * 255).astype(np.uint8)
            else:
                gray_small = gray
            
            # Gabor filters at different frequencies and orientations
            frequencies = [0.1, 0.3, 0.5]
            orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            for freq in frequencies:
                for theta in orientations:
                    try:
                        gabor_response = np.real(filters.gabor(
                            gray_small.astype(float),
                            frequency=freq,
                            theta=theta
                        )[0])
                        features[f'gabor_f{freq:.1f}_t{int(np.degrees(theta))}_mean'] = float(np.mean(gabor_response))
                        features[f'gabor_f{freq:.1f}_t{int(np.degrees(theta))}_std'] = float(np.std(gabor_response))
                    except:
                        pass
        except:
            pass
        
        return features
    
    def _histogram_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract histogram-based features."""
        features = {}
        
        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        # Histogram statistics
        features['hist_mean'] = float(np.sum(bins[:-1] * hist))
        features['hist_std'] = float(np.sqrt(np.sum(((bins[:-1] - features['hist_mean'])**2) * hist)))
        features['hist_entropy'] = float(stats.entropy(hist + 1e-10))
        
        # Mode
        mode_idx = np.argmax(hist)
        features['hist_mode'] = float(bins[mode_idx])
        
        return features
    
    def _edge_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features."""
        features = {}
        
        try:
            # Canny edges
            edges = feature.canny(gray.astype(float), sigma=1.0)
            features['edge_density'] = float(np.sum(edges) / edges.size)
            
            # Sobel edges
            sobel = filters.sobel(gray.astype(float))
            features['sobel_mean'] = float(np.mean(sobel))
            features['sobel_std'] = float(np.std(sobel))
            features['sobel_max'] = float(np.max(sobel))
            
            # Laplacian
            laplacian = filters.laplace(gray.astype(float))
            features['laplacian_var'] = float(np.var(laplacian))
        except:
            features['edge_density'] = 0.0
            features['sobel_mean'] = 0.0
            features['sobel_std'] = 0.0
            features['sobel_max'] = 0.0
            features['laplacian_var'] = 0.0
        
        return features
    
    def _morphological_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract morphological features."""
        features = {}
        
        try:
            # Binary threshold
            threshold = filters.threshold_otsu(gray)
            binary = gray > threshold
            
            # Connected components
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            
            if len(regions) > 0:
                areas = [r.area for r in regions]
                features['num_components'] = float(len(regions))
                features['largest_component_area'] = float(max(areas))
                features['mean_component_area'] = float(np.mean(areas))
            else:
                features['num_components'] = 0.0
                features['largest_component_area'] = 0.0
                features['mean_component_area'] = 0.0
        except:
            features['num_components'] = 0.0
            features['largest_component_area'] = 0.0
            features['mean_component_area'] = 0.0
        
        return features
    
    def _frequency_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract frequency domain features."""
        features = {}
        
        try:
            # FFT
            fft = np.fft.fft2(gray.astype(float))
            fft_abs = np.abs(fft)
            fft_shifted = np.fft.fftshift(fft_abs)
            
            # Low and high frequency energy
            h, w = fft_shifted.shape
            center_h, center_w = h // 2, w // 2
            radius = min(h, w) // 4
            
            # Create mask for low frequencies
            y, x = np.ogrid[:h, :w]
            mask_low = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            mask_high = ~mask_low
            
            features['freq_low_energy'] = float(np.sum(fft_shifted[mask_low]))
            features['freq_high_energy'] = float(np.sum(fft_shifted[mask_high]))
            features['freq_total_energy'] = float(np.sum(fft_abs))
        except:
            features['freq_low_energy'] = 0.0
            features['freq_high_energy'] = 0.0
            features['freq_total_energy'] = 0.0
        
        return features
    
    def _color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-specific features."""
        features = {}
        
        # Mean and std for each channel
        for i, color in enumerate(['R', 'G', 'B']):
            channel = image[:, :, i].flatten()
            features[f'{color}_mean'] = float(np.mean(channel))
            features[f'{color}_std'] = float(np.std(channel))
        
        # Color ratios
        r_mean = features['R_mean']
        g_mean = features['G_mean']
        b_mean = features['B_mean']
        total = r_mean + g_mean + b_mean + 1e-10
        
        features['R_ratio'] = float(r_mean / total)
        features['G_ratio'] = float(g_mean / total)
        features['B_ratio'] = float(b_mean / total)
        
        return features
    
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
        all_features = []
        
        for i, image in enumerate(images):
            if verbose and (i + 1) % 100 == 0:
                print(f"Extracting features from image {i+1}/{len(images)}")
            
            features = self.extract_features_from_image(image)
            all_features.append(features)
        
        # Convert to feature matrix
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


