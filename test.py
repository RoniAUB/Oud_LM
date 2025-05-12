import os
import numpy as np

# --- Load all frequencies from .npy files ---
input_folder = "Clustered_Frequencies"
os.makedirs(output_folder, exist_ok=True)

frequencies = []
durations=[]
for filename in os.listdir(input_folder):
    if filename.endswith('.npz'):
        try:
            A=np.load(os.path.join(input_folder, filename))
            frequencies.append(A['frequencies'])
            durations.append(A['durations'])
        except Exception as e:
            print(f"Error loading {filename}: {e}")