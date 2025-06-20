import pandas as pd
import os
import subprocess # For running shell commands like git clone

def download_udacity_data(target_dir='P1_Facial_Keypoints'):
    """Clones the Udacity Facial Keypoints dataset if not already present."""
    if not os.path.exists(target_dir):
        print(f"'{target_dir}' not found. Cloning dataset...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/udacity/P1_Facial_Keypoints.git", target_dir],
                check=True, capture_output=True
            )
            print(f"Dataset cloned into '{target_dir}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning dataset: {e.stderr.decode()}")
            raise # Re-raise the exception to halt if data isn't available
    else:
        print(f"'{target_dir}' directory already exists.")

def rename_keypoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames the keypoint columns in the DataFrame for better readability."""
    df_copy = df.copy() # To Avoid modifying the original DataFrame passed in
    new_columns = df_copy.columns.tolist()
    keypoint_idx = 0
    # First column is image name, rest are x0,y0,x1,y1...
    for i in range(1, len(new_columns), 2):
        keypoint_idx += 1
        new_columns[i] = f'x{keypoint_idx}'
        new_columns[i+1] = f'y{keypoint_idx}'
    if new_columns: # Check if new_columns is not empty
        new_columns[0] = 'image_file_name'
    df_copy.columns = new_columns
    return df_copy