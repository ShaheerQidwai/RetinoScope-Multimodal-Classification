import numpy as np
import json
import os

def extract_and_save_names(npz_path, json_path):
    if not os.path.exists(npz_path):
        print(f"Error: Could not find '{npz_path}'")
        return
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        if 'feature_names' in data:
            names = data['feature_names'].tolist()
            with open(json_path, 'w') as f:
                json.dump(names, f, indent=4)
            print(f"Successfully extracted {len(names)} feature names and saved to '{json_path}'")
        else:
            print(f"Error: 'feature_names' key not found in '{npz_path}'")
    except Exception as e:
        print(f"Error processing '{npz_path}': {e}")

if __name__ == "__main__":
    # Ensure features directory exists before trying
    os.makedirs("features", exist_ok=True)
    
    print("Extracting feature names...")
    extract_and_save_names(
        "features/train_features.npz", 
        "features/train_features_names.json"
    )
    extract_and_save_names(
        "features/oct_train_features.npz", 
        "features/oct_train_features_names.json"
    )
    print("Done.")
