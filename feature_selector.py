"""
Feature selection module for fundus image radiomics features.
Implements multiple feature selection methods appropriate for medical imaging.
"""
import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selection for radiomics features.
    Uses multiple methods and combines them for robust selection.
    """
    
    def __init__(self, n_features: int = None, method: str = 'combined'):
        """
        Initialize feature selector.
        
        Args:
            n_features: Number of features to select. If None, uses automatic selection.
            method: Selection method ('f_test', 'mutual_info', 'rfe', 'rf_importance', 'combined')
        """
        self.n_features = n_features
        self.method = method
        self.selected_features = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select features using the specified method.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of features if not specified
        if self.n_features is None:
            # Use 10% of features or 100, whichever is smaller
            self.n_features = min(100, max(50, int(X.shape[1] * 0.1)))
        
        # Cap at available features
        self.n_features = min(self.n_features, X.shape[1])
        
        if self.method == 'combined':
            # Use combined approach: multiple methods + voting
            selected = self._combined_selection(X_scaled, y, feature_names)
        elif self.method == 'f_test':
            selected = self._f_test_selection(X_scaled, y, feature_names)
        elif self.method == 'mutual_info':
            selected = self._mutual_info_selection(X_scaled, y, feature_names)
        elif self.method == 'rfe':
            selected = self._rfe_selection(X_scaled, y, feature_names)
        elif self.method == 'rf_importance':
            selected = self._rf_importance_selection(X_scaled, y, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selected_features = selected
        
        # Get selected feature names
        selected_names = [feature_names[i] for i in selected]
        
        return X_scaled[:, selected], selected_names
    
    def _f_test_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[int]:
        """Select features using F-test (ANOVA F-value)."""
        try:
            selector = SelectKBest(score_func=f_classif, k=self.n_features)
            selector.fit(X, y)
            return selector.get_support(indices=True).tolist()
        except:
            # Fallback: select all features
            return list(range(X.shape[1]))
    
    def _mutual_info_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[int]:
        """Select features using mutual information."""
        try:
            selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
            selector.fit(X, y)
            return selector.get_support(indices=True).tolist()
        except:
            return list(range(X.shape[1]))
    
    def _rfe_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[int]:
        """Select features using Recursive Feature Elimination."""
        try:
            # Use a robust classifier for RFE that handles correlated features well
            # Random Forest is good because it can handle feature interactions
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
            
            # Use RFE with step size to speed up for large feature sets
            step_size = max(1, int(X.shape[1] / 20))  # Remove 5% of features at a time
            selector = RFE(
                estimator, 
                n_features_to_select=self.n_features,
                step=step_size,
                verbose=0
            )
            selector.fit(X, y)
            return selector.get_support(indices=True).tolist()
        except Exception as e:
            print(f"Warning: RFE selection failed: {e}. Using all features.")
            return list(range(X.shape[1]))
    
    def _rf_importance_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[int]:
        """Select features using Random Forest feature importance."""
        try:
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Select top features
            top_indices = np.argsort(importances)[-self.n_features:]
            return top_indices.tolist()
        except:
            return list(range(X.shape[1]))
    
    def _combined_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> List[int]:
        """
        Combined feature selection using multiple methods and voting.
        """
        all_selected = []
        
        # Try different methods
        methods = [
            ('f_test', self._f_test_selection),
            ('mutual_info', self._mutual_info_selection),
            ('rf_importance', self._rf_importance_selection)
        ]
        
        for method_name, method_func in methods:
            try:
                selected = method_func(X, y, feature_names)
                all_selected.append(selected)
            except Exception as e:
                print(f"Warning: {method_name} selection failed: {e}")
                continue
        
        if not all_selected:
            # Fallback: use all features
            return list(range(X.shape[1]))
        
        # Count votes for each feature
        feature_votes = np.zeros(X.shape[1])
        for selected in all_selected:
            for idx in selected:
                feature_votes[idx] += 1
        
        # Select features with most votes
        # Use features that appear in at least one method
        threshold = 1  # At least one method selected it
        voted_features = np.where(feature_votes >= threshold)[0]
        
        # If we have more than n_features, take top voted
        if len(voted_features) > self.n_features:
            top_voted = np.argsort(feature_votes)[-self.n_features:]
            return top_voted.tolist()
        else:
            return voted_features.tolist()
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using selected features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Transformed feature matrix with selected features
        """
        if self.selected_features is None:
            raise ValueError("Must call select_features first")
        
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_features]


