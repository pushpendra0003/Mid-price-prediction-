import h5py
import pandas as pd
import numpy as np
import os


def inspect_h5_file(file_path):
    """Print the structure of the .h5 file to identify datasets."""
    with h5py.File(file_path, 'r') as f:
        print(f"Contents of file: {file_path}")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        f.visititems(print_structure)


def data_transform(file_paths):
    """Transform the data from the .h5 files into a single NumPy array."""
    all_data = []

    for file_path in file_paths:
        inspect_h5_file(file_path)  # Inspect the file structure
        dataset_name = 'STATES_DATA'  # Modify this based on the inspection result
        try:
            data = pd.read_hdf(file_path, dataset_name)
            all_data.append(data)
        except KeyError as e:
            print(f"Dataset '{dataset_name}' not found in file {file_path}: {e}")
            continue

        prefix = os.path.dirname(file_path)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data_array = combined_data.values
        np.save(prefix + '/combined_data_array', combined_data_array)

        # Save as a CSV for tabular inspection
        combined_data.to_csv(prefix + '/combined_data.csv', index=False)
        print("Combined data saved as NumPy array and CSV.")
    else:
        print("No data was processed. Please check dataset names and structure.")


# Define your file paths
stock_file_paths = [
    ('data_appl/15S091415-v50-APPL_OCT2.h5',
    'data_appl/15S091515-v50-APPL_OCT2.h5',
    'data_appl/15S091615-v50-APPL_OCT2.h5',
    'data_appl/15S091715-v50-APPL_OCT2.h5',)
]

# Process the files
for file_paths in stock_file_paths:
    data_transform(file_paths)
