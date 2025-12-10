"""
Simple script to run the OCT classification pipeline.
This is a convenience wrapper around oct_pipeline.py
"""
import sys
from oct_pipeline import main

if __name__ == '__main__':
    # You can modify these default parameters here
    # Or use command-line arguments with oct_pipeline.py directly
    
    # Set default arguments if running directly
    if len(sys.argv) == 1:
        # Default configuration for OCT
        sys.argv = [
            'run_oct_pipeline.py',
            '--data_dir', 'multieye_data/Oct',
            '--image_dir', 'multieye_data/Oct/ImageData/oct-filter-448x448',
            '--n_features', '50',
            '--feature_selection_method', 'rfe',
            '--imbalance_method', 'none',
            '--cv_folds', '5'
        ]
    
    main()

