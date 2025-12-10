"""
Model training module with class imbalance handling and model selection.
"""
import numpy as np
from typing import Tuple
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Trains multiple models and selects the best one.
    Handles class imbalance using various techniques.
    """
    
    def __init__(self, imbalance_method: str = 'smote', cv_folds: int = 5):
        """
        Initialize model trainer.
        
        Args:
            imbalance_method: Method to handle imbalance ('smote', 'adasyn', 'oversample', 
                            'undersample', 'smote_tomek', 'smote_enn', 'balanced_rf', 'none')
            cv_folds: Number of cross-validation folds
        """
        self.imbalance_method = imbalance_method
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        self.scaler = None
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using specified method.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Resampled (X, y)
        """
        if self.imbalance_method == 'none':
            return X, y
        
        print(f"\nHandling class imbalance using: {self.imbalance_method}")
        print(f"Original class distribution: {np.bincount(y)}")
        
        if self.imbalance_method == 'smote':
            sampler = SMOTE(random_state=42)
        elif self.imbalance_method == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif self.imbalance_method == 'oversample':
            sampler = RandomOverSampler(random_state=42)
        elif self.imbalance_method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif self.imbalance_method == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
        elif self.imbalance_method == 'smote_enn':
            sampler = SMOTEENN(random_state=42)
        else:
            return X, y
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"Resampled class distribution: {np.bincount(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Warning: Resampling failed: {e}. Using original data.")
            return X, y
    
    def get_models(self, n_classes: int, y: np.ndarray = None) -> tuple:
        """
        Get dictionary of models to evaluate.
        
        Args:
            n_classes: Number of classes
            y: Optional labels to compute class weights
            
        Returns:
            Tuple of (models_dict, class_weight_dict)
        """
        models = {}
        
        # Compute class weights if labels provided and not using SMOTE
        if y is not None and self.imbalance_method not in ['smote', 'adasyn', 'smote_tomek', 'smote_enn']:
            # Use square root weighting (dampened compared to linear)
            classes = np.unique(y)
            n_samples = len(y)
            n_classes = len(classes)
            
            # Square root weighting: weight = sqrt(n_samples / (n_classes * n_samples_class))
            class_weight_dict = {}
            for cls in classes:
                n_samples_class = np.sum(y == cls)
                linear_weight = n_samples / (n_classes * n_samples_class)
                sqrt_weight = np.sqrt(linear_weight)
                class_weight_dict[cls] = sqrt_weight
        else:
            class_weight_dict = 'balanced'
        
        # Random Forest
        # if self.imbalance_method == 'balanced_rf':
        #     models['BalancedRandomForest'] = BalancedRandomForestClassifier(
        #         n_estimators=200,
        #         max_depth=20,
        #         min_samples_split=5,
        #         min_samples_leaf=2,
        #         random_state=42,
        #         n_jobs=-1
        #     )
        # else:
        #     models['RandomForest'] = RandomForestClassifier(
        #         n_estimators=200,
        #         max_depth=20,
        #         min_samples_split=5,
        #         min_samples_leaf=2,
        #         random_state=42,
        #         n_jobs=-1,
        #         class_weight=class_weight_dict if isinstance(class_weight_dict, dict) else 'balanced'
        #     )
        
        # # Gradient Boosting - use sample_weight during fit
        # models['GradientBoosting'] = GradientBoostingClassifier(
        #     n_estimators=100,
        #     max_depth=10,
        #     learning_rate=0.1,
        #     random_state=42
        # )
        
        # # SVM (use linear kernel for large datasets)
        # models['SVM_Linear'] = SVC(
        #     kernel='linear',
        #     C=1.0,
        #     class_weight=class_weight_dict if isinstance(class_weight_dict, dict) else 'balanced',
        #     random_state=42,
        #     probability=True
        # )
        
        # XGBoost
        # xgb_available = False
        # xgb_model = None
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            models['XGBoost'] = xgb_model
            # xgb_available = True
        except ImportError:
            print("Warning: XGBoost not installed. Skipping XGBoost model.")
            print("Install with: pip install xgboost")
        
        # Ensemble: Voting Classifier (XGBoost + RandomForest)
        # Combines boosting (XGBoost) and bagging (RandomForest) for better stability
        # if xgb_available and self.imbalance_method != 'balanced_rf' and 'RandomForest' in models:
        #     try:
        #         rf_model = models['RandomForest']
        #         # Create voting ensemble with soft voting (uses probabilities)
        #         ensemble_estimators = [
        #             ('rf', rf_model),
        #             ('xgb', xgb_model)
        #         ]
        #         models['Ensemble_RF_XGB'] = VotingClassifier(
        #             estimators=ensemble_estimators,
        #             voting='soft',  # Use probability voting for better performance
        #             n_jobs=-1
        #         )
        #     except Exception as e:
        #         print(f"Warning: Could not create ensemble: {e}")
        
        # Logistic Regression
        # models['LogisticRegression'] = LogisticRegression(
        #     max_iter=1000,
        #     class_weight=class_weight_dict if isinstance(class_weight_dict, dict) else 'balanced',
        #     random_state=42,
        #     n_jobs=-1,
        #     multi_class='multinomial',
        #     solver='lbfgs'
        # )
        
        # # K-Nearest Neighbors - use sample_weight during fit
        # models['KNN'] = KNeighborsClassifier(
        #     n_neighbors=5,
        #     weights='distance',
        #     n_jobs=-1
        # )
        
        # # Neural Network - use sample_weight during fit
        # models['MLP'] = MLPClassifier(
        #     hidden_layer_sizes=(100, 50),
        #     max_iter=500,
        #     random_state=42,
        #     early_stopping=True,
        #     validation_fraction=0.1
        # )
        
        # # AdaBoost - use sample_weight during fit
        # models['AdaBoost'] = AdaBoostClassifier(
        #     n_estimators=100,
        #     learning_rate=1.0,
        #     random_state=42
        # )
        
        return models, class_weight_dict if isinstance(class_weight_dict, dict) else None
    
    def evaluate_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: dict = None,
        class_weights: dict = None
    ) -> dict:
        """
        Evaluate multiple models using cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            models: Dictionary of models (if None, uses default models)
            class_weights: Dictionary of class weights for sample_weight
            
        Returns:
            Dictionary of model names and their scores
        """
        if models is None:
            models, _ = self.get_models(len(np.unique(y)), y)
        
        print(f"\nEvaluating {len(models)} models using {self.cv_folds}-fold CV...")
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        results = {}
        
        # Compute sample weights if class_weights provided
        if class_weights is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weights, y)
        else:
            sample_weights = None
        
        for name, model in models.items():
            try:
                # For models that don't support class_weight, use sample_weight in fit
                # Ensemble models need special handling
                if name == 'Ensemble_RF_XGB':
                    # Ensemble: train each base estimator with sample weights
                    f1_scores = []
                    accuracy_scores = []
                    from sklearn.metrics import f1_score, accuracy_score
                    from sklearn.base import clone
                    
                    for train_idx, val_idx in cv.split(X, y):
                        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                        sample_weights_fold = sample_weights[train_idx] if sample_weights is not None else None
                        
                        # Clone ensemble
                        ensemble_fold = clone(model)
                        
                        # Train each estimator in the ensemble manually
                        # Get estimators list - handle both old and new sklearn versions
                        try:
                            # Try newer sklearn version first (estimators_ is a list)
                            if hasattr(ensemble_fold, 'estimators_') and len(ensemble_fold.estimators_) >= 2:
                                # Train RF (first estimator, index 0)
                                ensemble_fold.estimators_[0].fit(X_train_fold, y_train_fold)
                                # Train XGBoost (second estimator, index 1) with sample weights
                                if sample_weights_fold is not None:
                                    ensemble_fold.estimators_[1].fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                                else:
                                    ensemble_fold.estimators_[1].fit(X_train_fold, y_train_fold)
                            # Try older sklearn version (named_estimators_ is a dict)
                            elif hasattr(ensemble_fold, 'named_estimators_'):
                                for est_name, est in ensemble_fold.named_estimators_.items():
                                    if est_name == 'xgb' and sample_weights_fold is not None:
                                        est.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                                    else:
                                        est.fit(X_train_fold, y_train_fold)
                            else:
                                # Fallback: fit ensemble directly (won't use sample weights for XGBoost)
                                ensemble_fold.fit(X_train_fold, y_train_fold)
                        except Exception as e:
                            # If manual training fails, try fitting ensemble directly
                            ensemble_fold.fit(X_train_fold, y_train_fold)
                        
                        y_pred = ensemble_fold.predict(X_val_fold)
                        f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
                        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
                    
                    f1_scores = np.array(f1_scores)
                    accuracy_scores = np.array(accuracy_scores)
                elif name in ['GradientBoosting', 'KNN', 'MLP', 'AdaBoost', 'XGBoost'] and sample_weights is not None:
                    # Manual cross-validation with sample weights
                    f1_scores = []
                    accuracy_scores = []
                    from sklearn.metrics import f1_score, accuracy_score
                    
                    for train_idx, val_idx in cv.split(X, y):
                        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                        sample_weights_fold = sample_weights[train_idx] if sample_weights is not None else None
                        
                        # Clone model for each fold
                        from sklearn.base import clone
                        model_fold = clone(model)
                        model_fold.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
                        
                        y_pred = model_fold.predict(X_val_fold)
                        f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
                        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
                    
                    f1_scores = np.array(f1_scores)
                    accuracy_scores = np.array(accuracy_scores)
                else:
                    # Standard cross-validation
                    f1_scores = cross_val_score(
                        model, X, y,
                        cv=cv,
                        scoring='f1_macro',
                        n_jobs=-1
                    )
                    accuracy_scores = cross_val_score(
                        model, X, y,
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1
                    )
                
                f1_mean = f1_scores.mean()
                f1_std = f1_scores.std()
                acc_mean = accuracy_scores.mean()
                acc_std = accuracy_scores.std()
                
                results[name] = {
                    'mean': f1_mean,  # Keep F1 as primary for selection
                    'std': f1_std,
                    'f1_mean': f1_mean,
                    'f1_std': f1_std,
                    'accuracy_mean': acc_mean,
                    'accuracy_std': acc_std,
                    'scores': f1_scores
                }
                print(f"{name}: Accuracy = {acc_mean:.4f} (+/- {acc_std:.4f}), F1-macro = {f1_mean:.4f} (+/- {f1_std:.4f})")
            except Exception as e:
                print(f"{name}: Failed - {e}")
                results[name] = {'mean': 0, 'std': 0, 'f1_mean': 0, 'f1_std': 0, 'accuracy_mean': 0, 'accuracy_std': 0, 'scores': []}
        
        return results
    
    def select_best_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: dict = None,
        class_weights: dict = None
    ) -> tuple:
        """
        Select the best model based on cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            models: Dictionary of models
            class_weights: Dictionary of class weights for sample_weight
            
        Returns:
            Tuple of (best_model, best_model_name, best_score)
        """
        results = self.evaluate_models(X, y, models, class_weights)
        
        # Find best model
        best_name = max(results.keys(), key=lambda k: results[k]['mean'])
        best_score = results[best_name]['mean']
        
        print(f"\nBest model: {best_name} (F1-macro = {best_score:.4f})")
        
        # Train best model on full data
        if models is None:
            models, class_weights = self.get_models(len(np.unique(y)), y)
        
        best_model = models[best_name]
        
        # Use sample weights for models that need it
        if best_name == 'Ensemble_RF_XGB':
            # Ensemble: train each base estimator with sample weights
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weights, y) if class_weights is not None else None
            
            # Train each estimator in the ensemble
            if hasattr(best_model, 'named_estimators_'):
                # Older sklearn versions
                for est_name, est in best_model.named_estimators_.items():
                    if est_name == 'xgb' and sample_weights is not None:
                        est.fit(X, y, sample_weight=sample_weights)
                    else:
                        est.fit(X, y)
            else:
                # Newer sklearn versions - use estimators_ list
                # Estimators are in order: [rf, xgb]
                if len(best_model.estimators_) >= 2:
                    # Train RF (first estimator)
                    best_model.estimators_[0].fit(X, y)
                    # Train XGBoost (second estimator) with sample weights
                    if sample_weights is not None:
                        best_model.estimators_[1].fit(X, y, sample_weight=sample_weights)
                    else:
                        best_model.estimators_[1].fit(X, y)
        elif best_name in ['GradientBoosting', 'KNN', 'MLP', 'AdaBoost', 'XGBoost'] and class_weights is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weights, y)
            best_model.fit(X, y, sample_weight=sample_weights)
        else:
            best_model.fit(X, y)
        
        self.best_model = best_model
        self.best_model_name = best_name
        self.best_score = best_score
        
        return best_model, best_name, best_score
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None
    ) -> dict:
        """
        Complete training pipeline: handle imbalance, select best model, evaluate.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            
        Returns:
            Dictionary with results and metrics
        """
        # Handle class imbalance (or use original data with class weights)
        if self.imbalance_method != 'none':
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
            # Compute class weights from balanced data (though they should be equal after SMOTE)
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train_balanced)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
            class_weight_dict = dict(zip(classes, class_weights))
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            # Compute square root class weights (dampened compared to linear)
            classes = np.unique(y_train)
            n_samples = len(y_train)
            n_classes = len(classes)
            
            # Square root weighting: weight = sqrt(n_samples / (n_classes * n_samples_class))
            # This is less extreme than linear weights (1/frequency)
            class_weight_dict = {}
            for cls in classes:
                n_samples_class = np.sum(y_train == cls)
                # Linear weight would be: n_samples / (n_classes * n_samples_class)
                linear_weight = n_samples / (n_classes * n_samples_class)
                # Square root weight: sqrt of linear weight
                sqrt_weight = np.sqrt(linear_weight)
                class_weight_dict[cls] = sqrt_weight
            
            print(f"\nUsing square root class weights (no resampling) - dampened effect:")
            for cls, weight in sorted(class_weight_dict.items()):
                count = np.sum(y_train == cls)
                linear_weight = n_samples / (n_classes * count)
                print(f"  Class {cls}: {count:5d} samples, sqrt_weight = {weight:.4f} (linear would be {linear_weight:.4f})")
        
        # Select best model
        best_model, best_name, best_score = self.select_best_model(
            X_train_balanced, y_train_balanced, class_weights=class_weight_dict
        )
        
        # Evaluate on test set if provided
        results = {
            'best_model': best_model,
            'best_model_name': best_name,
            'cv_score': best_score,
            'test_metrics': None
        }
        
        if X_test is not None and y_test is not None:
            y_pred = best_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            
            results['test_metrics'] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print("\n" + "="*50)
            print("Test Set Results:")
            print("="*50)
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-macro: {f1_macro:.4f}")
            print(f"F1-weighted: {f1_weighted:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print("\nClassification Report:")
            print(results['test_metrics']['classification_report'])
        
        return results

