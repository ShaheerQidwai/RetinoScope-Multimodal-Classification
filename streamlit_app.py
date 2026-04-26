"""
Streamlit Dashboard for Fundus and OCT Image Classification
Provides overview, results visualization, and interactive prediction interface.
"""
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
from pathlib import Path
import pandas as pd

# Page config
st.set_page_config(
    page_title="Fundus & OCT Classification Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Model paths
FUNDUS_MODEL_PATH = 'models/best_model.pkl'
OCT_MODEL_PATH = 'models/oct_best_model.pkl'

# Class names (after excluding classes 2 and 5)
CLASS_NAMES = {
    0: 'Normal',
    1: 'Class 1',
    2: 'Class 3',
    3: 'Class 4',
    4: 'Class 6',
    5: 'Class 7',
    6: 'Class 8'
}

@st.cache_resource
def load_models():
    """Load Fundus and OCT models (cached)."""
    try:
        with open(FUNDUS_MODEL_PATH, 'rb') as f:
            fundus_data = pickle.load(f)
        with open(OCT_MODEL_PATH, 'rb') as f:
            oct_data = pickle.load(f)
        
        fundus_model = fundus_data['model'] if isinstance(fundus_data, dict) else fundus_data
        oct_model = oct_data['model'] if isinstance(oct_data, dict) else oct_data
        
        # Get selectors first
        fundus_selector = fundus_data.get('feature_selector') if isinstance(fundus_data, dict) else None
        oct_selector = oct_data.get('feature_selector') if isinstance(oct_data, dict) else None
        
        # Try to load original feature names from saved features files
        # The selector expects features BEFORE selection (139 features)
        fundus_original_features = None
        oct_original_features = None
        
        # Method 1: Try to load from features file (before selection) - this is the source of truth
        try:
            fundus_features_file = 'features/train_features.npz'
            if os.path.exists(fundus_features_file):
                fundus_feat_data = np.load(fundus_features_file, allow_pickle=True)
                if 'feature_names' in fundus_feat_data:
                    fundus_original_features = fundus_feat_data['feature_names'].tolist()
                    st.success(f"✓ Loaded {len(fundus_original_features)} original Fundus feature names from training data")
        except Exception as e:
            st.warning(f"Could not load Fundus original features from file: {e}")
        
        # Method 2: Try to get from feature selector if it has feature_names stored
        if fundus_original_features is None and fundus_selector is not None:
            if hasattr(fundus_selector, 'feature_names'):
                fundus_original_features = fundus_selector.feature_names
                st.info(f"✓ Got {len(fundus_original_features)} Fundus feature names from selector")
        
        try:
            oct_features_file = 'features/oct_train_features.npz'
            if os.path.exists(oct_features_file):
                oct_feat_data = np.load(oct_features_file, allow_pickle=True)
                if 'feature_names' in oct_feat_data:
                    oct_original_features = oct_feat_data['feature_names'].tolist()
                    st.success(f"✓ Loaded {len(oct_original_features)} original OCT feature names from training data")
        except Exception as e:
            st.warning(f"Could not load OCT original features from file: {e}")
        
        # Method 2: Try to get from feature selector if it has feature_names stored
        if oct_original_features is None and oct_selector is not None:
            if hasattr(oct_selector, 'feature_names'):
                oct_original_features = oct_selector.feature_names
                st.info(f"✓ Got {len(oct_original_features)} OCT feature names from selector")
        
        # Get test metrics if available
        fundus_test_metrics = fundus_data.get('test_metrics', {}) if isinstance(fundus_data, dict) else {}
        oct_test_metrics = oct_data.get('test_metrics', {}) if isinstance(oct_data, dict) else {}
        
        return {
            'fundus_model': fundus_model,
            'oct_model': oct_model,
            'fundus_selector': fundus_selector,
            'oct_selector': oct_selector,
            'fundus_cv_score': fundus_data.get('cv_score', 0) if isinstance(fundus_data, dict) else 0,
            'oct_cv_score': oct_data.get('cv_score', 0) if isinstance(oct_data, dict) else 0,
            'fundus_test_accuracy': fundus_test_metrics.get('accuracy', 0) if fundus_test_metrics else 0,
            'oct_test_accuracy': oct_test_metrics.get('accuracy', 0) if oct_test_metrics else 0,
            'fundus_test_f1_macro': fundus_test_metrics.get('f1_macro', 0) if fundus_test_metrics else 0,
            'oct_test_f1_macro': oct_test_metrics.get('f1_macro', 0) if oct_test_metrics else 0,
            'fundus_feature_names': fundus_data.get('selected_feature_names') if isinstance(fundus_data, dict) else None,
            'oct_feature_names': oct_data.get('selected_feature_names') if isinstance(oct_data, dict) else None,
            'fundus_original_feature_names': fundus_original_features,
            'oct_original_feature_names': oct_original_features,
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

@st.cache_resource
def load_feature_extractor():
    """Load feature extractor (cached)."""
    from radiomics_extractor import RadiomicsExtractor
    return RadiomicsExtractor()

def extract_features_from_image(image, extractor, expected_feature_names=None):
    """Extract features from a single image."""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            # Ensure RGB
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
        else:
            img_array = image
        
        # Extract features
        features_dict = extractor.extract_features_from_image(img_array)
        
        # Match feature order to training features if provided
        if expected_feature_names is not None:
            # Use expected feature order - this ensures we get all features in correct order
            # Missing features will be set to 0.0
            features = np.array([features_dict.get(name, 0.0) for name in expected_feature_names])
            feature_names = expected_feature_names
            
            # Warn if some features are missing
            missing_features = [name for name in expected_feature_names if name not in features_dict]
            if missing_features:
                st.warning(f"Missing {len(missing_features)} features, using 0.0 as default")
        else:
            # Use sorted feature names
            feature_names = sorted(features_dict.keys())
            features = np.array([features_dict[name] for name in feature_names])
        
        return features, feature_names
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def predict_single_image(image, model, selector, extractor, modality_name, expected_feature_names=None, original_feature_names=None):
    """Predict class for a single image using the same feature extraction as training."""
    # Extract features - use original feature names from training (before selection)
    # This ensures we use the exact same features in the same order as training
    features, feature_names = extract_features_from_image(image, extractor, original_feature_names)
    
    if features is None:
        return None, None
    
    # Reshape for single sample
    features = features.reshape(1, -1)
    
    # Verify feature count matches what selector expects
    if selector is not None:
        # Check what the scaler expects (it was fit on original features during training)
        expected_n_features = None
        if hasattr(selector, 'scaler') and hasattr(selector.scaler, 'n_features_in_'):
            expected_n_features = selector.scaler.n_features_in_
        elif hasattr(selector, 'feature_names'):
            expected_n_features = len(selector.feature_names)
        
        if expected_n_features is not None:
            if features.shape[1] != expected_n_features:
                st.warning(f"[{modality_name}] Feature count mismatch: got {features.shape[1]}, selector expects {expected_n_features}")
                
                # If we have original feature names, re-extract with proper ordering
                if original_feature_names is not None and len(original_feature_names) == expected_n_features:
                    # Convert image to array if needed
                    if isinstance(image, Image.Image):
                        img_array = np.array(image)
                        if img_array.ndim == 2:
                            img_array = np.stack([img_array] * 3, axis=-1)
                        elif img_array.shape[2] == 4:
                            img_array = img_array[:, :, :3]
                    else:
                        img_array = image
                    
                    # Re-extract features with proper ordering using training feature names
                    features_dict = extractor.extract_features_from_image(img_array)
                    # Create feature vector matching original feature order exactly
                    features = np.array([[features_dict.get(name, 0.0) for name in original_feature_names]])
                    st.success(f"[{modality_name}] ✓ Re-extracted {len(original_feature_names)} features using training feature names")
                else:
                    # Fallback: pad or truncate
                    if features.shape[1] < expected_n_features:
                        padding = np.zeros((features.shape[0], expected_n_features - features.shape[1]))
                        features = np.hstack([features, padding])
                        st.warning(f"[{modality_name}] Padded features (may affect accuracy)")
                    else:
                        features = features[:, :expected_n_features]
                        st.warning(f"[{modality_name}] Truncated features (may affect accuracy)")
    
    # Apply feature selection (transforms 139 features -> 50 features)
    if selector is not None:
        try:
            features = selector.transform(features)
            st.success(f"[{modality_name}] ✓ Applied feature selection: {features.shape[1]} features selected")
        except Exception as e:
            st.error(f"Could not apply feature selection for {modality_name}: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None, None
    
    # Get prediction from trained model
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        return prediction, probabilities
    except Exception as e:
        st.error(f"Error making prediction for {modality_name}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def fuse_predictions(fundus_proba, oct_proba, fundus_weight=0.3, oct_weight=0.7):
    """Fuse predictions from Fundus and OCT models."""
    p_fused = (fundus_weight * fundus_proba + oct_weight * oct_proba) / (fundus_weight + oct_weight)
    return p_fused

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Fundus & OCT Image Classification Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Overview", "📊 Results", "🔮 Predictions", "ℹ️ About"]
    )
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please check model files.")
        return
    
    extractor = load_feature_extractor()
    
    if page == "🏠 Overview":
        show_overview(models)
    elif page == "📊 Results":
        show_results(models)
    elif page == "🔮 Predictions":
        show_predictions(models, extractor)
    elif page == "ℹ️ About":
        show_about(models)

def show_overview(models):
    """Show project overview."""
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Description")
        st.markdown("""
        This project implements a **multi-modal medical image classification system** 
        for fundus and OCT (Optical Coherence Tomography) images.
        
        **Key Features:**
        - ✅ Radiomics feature extraction
        - ✅ Advanced feature selection (RFE)
        - ✅ Class imbalance handling (square root weighting)
        - ✅ Ensemble learning (Random Forest + XGBoost)
        - ✅ Simulated late fusion for unpaired data
        - ✅ 7-class classification (excluding rare classes)
        """)
    
    with col2:
        st.subheader("🔬 Methodology")
        st.markdown("""
        **Pipeline:**
        1. **Feature Extraction**: Radiomics features from images
        2. **Feature Selection**: RFE selects best 50 features
        3. **Model Training**: Multiple models evaluated
        4. **Best Model Selection**: Cross-validation based
        5. **Late Fusion**: Combines Fundus + OCT predictions
        """)
    
    st.divider()
    
    st.subheader("📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    # Use test accuracy if available, otherwise fall back to CV score
    fundus_score = models.get('fundus_test_accuracy', 0)
    fundus_label = "Test Accuracy"
    if fundus_score == 0:
        fundus_score = models.get('fundus_cv_score', 0)
        fundus_label = "CV F1-macro"
    
    oct_score = models.get('oct_test_accuracy', 0)
    oct_label = "Test Accuracy"
    if oct_score == 0:
        oct_score = models.get('oct_cv_score', 0)
        oct_label = "CV F1-macro"
    
    with col1:
        st.metric(
            "Fundus Accuracy",
            "70.55%",
            help="Validated Test Accuracy on 11,529 samples"
        )
    
    with col2:
        st.metric(
            "OCT Accuracy",
            "86.74%",
            help="Validated Test Accuracy on 8,710 samples"
        )
    
    with col3:
        # Calculate fusion performance (average of both models)
        fusion_score = (fundus_score + oct_score) / 2
        st.metric(
            "Fusion Accuracy",
            "88.26%",
            delta="+1.52%",
            help="Weighted Late Fusion (40% Fundus / 60% OCT)"
        )
    
    st.divider()
    
    st.subheader("📁 Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Fundus Dataset:**
        - Training: ~46,415 images
        - Test: ~11,601 images
        - Classes: 7 (0, 1, 3, 4, 6, 7, 8)
        - Features: 139 → 50 (selected)
        """)
    
    with col2:
        st.info("""
        **OCT Dataset:**
        - Training: Variable
        - Test: Variable
        - Classes: 7 (0, 1, 3, 4, 6, 7, 8)
        - Features: 139 → 50 (selected)
        """)

def show_results(models):
    """Show detailed results and visualizations."""
    st.header("📊 Results & Performance")
    
    # Load results if available
    results_file = 'fusion_results.png'
    if os.path.exists(results_file):
        st.subheader("Confusion Matrix")
        st.image(results_file, width='stretch')
    
    # Model comparison
    st.subheader("Model Comparison")
    # These are your FINAL validated numbers from latefusion.py
    fundus_acc = 0.7055
    oct_acc = 0.8674
    fusion_acc = 0.8826
    
    metric_label = 'Test Accuracy'
    
    # Create comparison chart
    comparison_data = {
        'Model': ['Fundus Only', 'OCT Only', 'Fused System'],
        metric_label: [fundus_acc, oct_acc, fusion_acc]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_comparison['Model'], df_comparison[metric_label], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_ylabel(metric_label)
    ax.set_title('Validated Model Performance')
    ax.set_ylim([0, 1.0]) # Set y-axis to 0-100%
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}',
                ha='center', va='bottom', fontweight='bold')
                
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    # # Use test accuracy if available, otherwise use CV scores
    # fundus_perf = models.get('fundus_test_accuracy', models.get('fundus_cv_score', 0))
    # oct_perf = models.get('oct_test_accuracy', models.get('oct_cv_score', 0))
    # fusion_perf = (fundus_perf + oct_perf) / 2
    
    # # Determine metric label
    # if models.get('fundus_test_accuracy', 0) > 0 or models.get('oct_test_accuracy', 0) > 0:
    #     metric_label = 'Test Accuracy'
    # else:
    #     metric_label = 'CV F1-macro'
    
    # # Create comparison chart
    # comparison_data = {
    #     'Model': ['Fundus Only', 'OCT Only', 'Fused System'],
    #     metric_label: [fundus_perf, oct_perf, fusion_perf]
    # }
    
    # df_comparison = pd.DataFrame(comparison_data)
    
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(df_comparison['Model'], df_comparison[metric_label], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    # ax.set_ylabel(metric_label)
    # ax.set_title('Model Performance Comparison')
    # ax.set_ylim([0, 1])
    # for i, v in enumerate(df_comparison[metric_label]):
    #     ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # st.pyplot(fig)
    
    # Feature information
    st.subheader("Feature Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Original Features", "139", help="Radiomics features extracted")
    
    with col2:
        st.metric("Selected Features", "50", help="Features after RFE selection")
    
    # Class distribution (if we can load it)
    st.subheader("Class Distribution")
    st.info("""
    Classes are balanced using square root weighting:
    - Majority classes get lower weights
    - Minority classes get higher weights
    - Prevents overfitting to majority class
    """)

def show_predictions(models, extractor):
    """Show prediction interface."""
    st.header("🔮 Image Classification")
    
    st.markdown("""
    Upload Fundus and/or OCT images to get predictions. 
    You can upload one or both modalities.
    """)
    
    # Fusion weights
    st.sidebar.subheader("Fusion Settings")
    fundus_weight = st.sidebar.slider(
        "Fundus Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Weight for Fundus model in fusion"
    )
    oct_weight = st.sidebar.slider(
        "OCT Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Weight for OCT model in fusion"
    )
    
    # Normalize weights
    total_weight = fundus_weight + oct_weight
    if total_weight > 0:
        fundus_weight = fundus_weight / total_weight
        oct_weight = oct_weight / total_weight
    
    # Image upload
    col1, col2 = st.columns(2)
    
    fundus_image = None
    oct_image = None
    
    with col1:
        st.subheader("📷 Fundus Image")
        fundus_file = st.file_uploader(
            "Upload Fundus Image",
            type=['png', 'jpg', 'jpeg'],
            key='fundus'
        )
        if fundus_file is not None:
            fundus_image = Image.open(fundus_file)
            st.image(fundus_image, caption="Uploaded Fundus Image", width='stretch')
    
    with col2:
        st.subheader("📷 OCT Image")
        oct_file = st.file_uploader(
            "Upload OCT Image",
            type=['png', 'jpg', 'jpeg'],
            key='oct'
        )
        if oct_file is not None:
            oct_image = Image.open(oct_file)
            st.image(oct_image, caption="Uploaded OCT Image", width='stretch')
    
    # Prediction button
    if st.button("🔍 Predict", type="primary"):
        if fundus_image is None and oct_image is None:
            st.warning("Please upload at least one image (Fundus or OCT)")
        else:
            with st.spinner("Processing images and making predictions..."):
                results = {}
                
                # Fundus prediction
                if fundus_image is not None:
                    try:
                        fundus_pred, fundus_proba = predict_single_image(
                            fundus_image,
                            models['fundus_model'],
                            models['fundus_selector'],
                            extractor,
                            "Fundus",
                            models.get('fundus_feature_names'),  # Selected feature names
                            models.get('fundus_original_feature_names')  # Original feature names (before selection)
                        )
                        if fundus_pred is not None:
                            results['fundus'] = {
                                'prediction': fundus_pred,
                                'probabilities': fundus_proba,
                                'class_name': CLASS_NAMES[fundus_pred]
                            }
                    except Exception as e:
                        st.error(f"Error predicting Fundus image: {e}")
                
                # OCT prediction
                if oct_image is not None:
                    try:
                        oct_pred, oct_proba = predict_single_image(
                            oct_image,
                            models['oct_model'],
                            models['oct_selector'],
                            extractor,
                            "OCT",
                            models.get('oct_feature_names'),  # Selected feature names
                            models.get('oct_original_feature_names')  # Original feature names (before selection)
                        )
                        if oct_pred is not None:
                            results['oct'] = {
                                'prediction': oct_pred,
                                'probabilities': oct_proba,
                                'class_name': CLASS_NAMES[oct_pred]
                            }
                    except Exception as e:
                        st.error(f"Error predicting OCT image: {e}")
                
                # Display results
                if results:
                    st.success("✅ Predictions Complete!")
                    st.divider()
                    
                    # Individual predictions
                    if 'fundus' in results:
                        st.subheader("📷 Fundus Prediction")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                "Predicted Class",
                                results['fundus']['class_name'],
                                help=f"Class {results['fundus']['prediction']}"
                            )
                        with col2:
                            # Probability bar chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            probs = results['fundus']['probabilities']
                            classes = [CLASS_NAMES[i] for i in range(len(probs))]
                            colors = ['green' if i == results['fundus']['prediction'] else 'gray' for i in range(len(probs))]
                            ax.barh(classes, probs, color=colors)
                            ax.set_xlabel('Probability')
                            ax.set_title('Fundus Prediction Probabilities')
                            ax.set_xlim([0, 1])
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    if 'oct' in results:
                        st.subheader("📷 OCT Prediction")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                "Predicted Class",
                                results['oct']['class_name'],
                                help=f"Class {results['oct']['prediction']}"
                            )
                        with col2:
                            # Probability bar chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            probs = results['oct']['probabilities']
                            classes = [CLASS_NAMES[i] for i in range(len(probs))]
                            colors = ['green' if i == results['oct']['prediction'] else 'gray' for i in range(len(probs))]
                            ax.barh(classes, probs, color=colors)
                            ax.set_xlabel('Probability')
                            ax.set_title('OCT Prediction Probabilities')
                            ax.set_xlim([0, 1])
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Fusion prediction (if both available)
                    if 'fundus' in results and 'oct' in results:
                        st.divider()
                        st.subheader("🔀 Fused Prediction")
                        
                        fused_proba = fuse_predictions(
                            results['fundus']['probabilities'],
                            results['oct']['probabilities'],
                            fundus_weight,
                            oct_weight
                        )
                        fused_pred = np.argmax(fused_proba)
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(
                                "Fused Prediction",
                                CLASS_NAMES[fused_pred],
                                help=f"Class {fused_pred} (Weighted: Fundus {fundus_weight:.1%}, OCT {oct_weight:.1%})"
                            )
                        
                        with col2:
                            # Fused probability chart
                            fig, ax = plt.subplots(figsize=(8, 4))
                            classes = [CLASS_NAMES[i] for i in range(len(fused_proba))]
                            colors = ['green' if i == fused_pred else 'gray' for i in range(len(fused_proba))]
                            ax.barh(classes, fused_proba, color=colors)
                            ax.set_xlabel('Probability')
                            ax.set_title('Fused Prediction Probabilities')
                            ax.set_xlim([0, 1])
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Agreement indicator
                        if results['fundus']['prediction'] == results['oct']['prediction']:
                            st.success("✅ Both models agree on the prediction!")
                        else:
                            st.info(f"ℹ️ Models disagree: Fundus={CLASS_NAMES[results['fundus']['prediction']]}, OCT={CLASS_NAMES[results['oct']['prediction']]}")

def show_about(models):
    """Show about page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## Project Information
    
    This dashboard provides an interactive interface for the Fundus and OCT image 
    classification system developed as part of a Final Year Project.
    
    ### Key Technologies
    - **Python** - Core programming language
    - **Scikit-learn** - Machine learning models
    - **XGBoost** - Gradient boosting
    - **scikit-image** - Feature extraction
    - **Streamlit** - Interactive dashboard
    
    ### Methodology
    1. **Radiomics Feature Extraction**: Comprehensive feature extraction from medical images
    2. **Feature Selection**: RFE (Recursive Feature Elimination) for optimal feature subset
    3. **Class Imbalance Handling**: Square root weighting for balanced learning
    4. **Model Training**: Multiple models evaluated via cross-validation
    5. **Late Fusion**: Simulated fusion for unpaired multi-modal data
    
    ### Contact
    For questions or issues, please refer to the project documentation.
    """)
    
    st.divider()
    
    st.subheader("📚 Model Information")
    col1, col2 = st.columns(2)
    
    # Get performance metrics
    fundus_test_acc = models.get('fundus_test_accuracy', 0)
    fundus_test_f1 = models.get('fundus_test_f1_macro', 0)
    fundus_cv = models.get('fundus_cv_score', 0)
    
    oct_test_acc = models.get('oct_test_accuracy', 0)
    oct_test_f1 = models.get('oct_test_f1_macro', 0)
    oct_cv = models.get('oct_cv_score', 0)
    
    with col1:
        fundus_info = "**Fundus Model:**\n- Best Model: XGBoost\n- Features: 50 selected\n"
        if fundus_test_acc > 0:
            fundus_info += f"- Test Accuracy: {fundus_test_acc:.4f}\n- Test F1-macro: {fundus_test_f1:.4f}\n"
        if fundus_cv > 0:
            fundus_info += f"- CV F1-macro: {fundus_cv:.4f}"
        st.info(fundus_info)
    
    with col2:
        oct_info = "**OCT Model:**\n- Features: 50 selected\n"
        if oct_test_acc > 0:
            oct_info += f"- Test Accuracy: {oct_test_acc:.4f}\n- Test F1-macro: {oct_test_f1:.4f}\n"
        if oct_cv > 0:
            oct_info += f"- CV F1-macro: {oct_cv:.4f}"
        st.info(oct_info)

if __name__ == "__main__":
    main()

