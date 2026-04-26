import pickle

# Specify the path to your .pkl file
file_path = 'models/oct_best_model.pkl' 

try:
    # Open the file in binary read mode ('rb')
    with open(file_path, 'rb') as f:
        # Load the object from the file
        data = pickle.load(f)
    
    # Now 'data' contains the Python object that was stored in the .pkl file
    print("File opened successfully. Loaded data:")
    print(data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred while opening or loading the file: {e}")