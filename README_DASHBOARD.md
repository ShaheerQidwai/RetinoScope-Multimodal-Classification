# Streamlit Dashboard - Fundus & OCT Classification

## Overview

This Streamlit dashboard provides an interactive interface for the Fundus and OCT image classification system. It includes:

- **Project Overview**: Methodology and dataset information
- **Results Visualization**: Performance metrics and confusion matrices
- **Interactive Predictions**: Upload images and get predictions
- **Late Fusion**: Combine predictions from both modalities

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure models are trained and saved:
   - `models/best_model.pkl` (Fundus model)
   - `models/oct_best_model.pkl` (OCT model)

3. Optional: Run late fusion to generate results:
```bash
python latefusion.py --optimize_weights
```
This will create `fusion_results.png` for visualization.

## Running the Dashboard

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Features

### 🏠 Overview Page
- Project description and methodology
- Model performance metrics
- Dataset information

### 📊 Results Page
- Confusion matrix visualization
- Model comparison charts
- Feature information

### 🔮 Predictions Page
- Upload Fundus and/or OCT images
- Get individual predictions
- View probability distributions
- Fused predictions (when both modalities available)
- Adjustable fusion weights

### ℹ️ About Page
- Project information
- Technology stack
- Model details

## Usage

1. **Single Modality Prediction**:
   - Upload either Fundus or OCT image
   - Click "Predict" to get classification

2. **Multi-Modality Fusion**:
   - Upload both Fundus and OCT images
   - Adjust fusion weights in sidebar (default: 30% Fundus, 70% OCT)
   - Get fused prediction combining both modalities

3. **View Results**:
   - Navigate to Results page to see performance metrics
   - View confusion matrix if available

## Troubleshooting

- **Model Loading Error**: Ensure model files exist in `models/` directory
- **Feature Extraction Error**: Check that `radiomics_extractor.py` and `feature_extractor_alternative.py` are available
- **Prediction Error**: Ensure uploaded images are valid (PNG, JPG, JPEG)

## Notes

- Feature extraction may take a few seconds per image
- Models are cached for faster subsequent predictions
- Feature names must match training feature order for accurate predictions

