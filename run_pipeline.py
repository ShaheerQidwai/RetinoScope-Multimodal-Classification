"""
Simple script to run the fundus classification pipeline.
This is a convenience wrapper around fundus_pipeline.py
"""
import sys
from fundus_pipeline import main

if __name__ == '__main__':
    # You can modify these default parameters here
    # Or use command-line arguments with fundus_pipeline.py directly
    
    # Set default arguments if running directly
    if len(sys.argv) == 1:
        # Default configuration
        sys.argv = [
            'run_pipeline.py',
            '--data_dir', 'multieye_data/Fundus',
            '--image_dir', 'multieye_data/Fundus/ImageData/cfp-clahe-224x224',
            '--n_features', '100',
            '--feature_selection_method', 'combined',
            '--imbalance_method', 'smote',
            '--cv_folds', '5'
        ]
    
    main()


