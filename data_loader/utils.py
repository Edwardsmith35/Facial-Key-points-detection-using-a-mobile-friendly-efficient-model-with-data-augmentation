import pandas as pd
import os
import subprocess

from sklearn.model_selection import train_test_split
import config
from data_loader.augmentations import get_training_augmentations
from data_loader.dataset import FacialKeypointsDataset

from torch.utils.data import DataLoader


def load_training_data():
    try:
        return pd.read_csv(config.TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Training CSV not found at {config.TRAIN_CSV_PATH}")
        print("Check dataset download and config paths.")
        exit(1)


def split_data(df):
    return train_test_split(
        df,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE_DATA_SPLIT,
    )


def download_udacity_data(target_dir="P1_Facial_Keypoints"):
    """Clones the Udacity Facial Keypoints dataset if not already present."""
    if not os.path.exists(target_dir):
        print(f"'{target_dir}' not found. Cloning dataset...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/udacity/P1_Facial_Keypoints.git",
                    target_dir,
                ],
                check=True,
                capture_output=True,
            )
            print(f"Dataset cloned into '{target_dir}'.")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning dataset: {e.stderr.decode()}")
            raise
    else:
        print(f"'{target_dir}' directory already exists.")


def rename_keypoint_columns(df):
    """Renames the keypoint columns in the DataFrame for better readability.

    Return:
        DataFrame
    """
    df_copy = df.copy()  # To Avoid modifying the original DataFrame passed in
    new_columns = df_copy.columns.tolist()
    keypoint_idx = 0
    # First column is image name, rest are x0,y0,x1,y1...
    for i in range(1, len(new_columns), 2):
        keypoint_idx += 1
        new_columns[i] = f"x{keypoint_idx}"
        new_columns[i + 1] = f"y{keypoint_idx}"
    if new_columns:
        new_columns[0] = "image_file_name"
    df_copy.columns = new_columns
    return df_copy


def create_dataloaders(train_df, val_df):
    train_augs = get_training_augmentations()
    train_dataset = FacialKeypointsDataset(
        train_df.reset_index(drop=True), config.TRAIN_IMG_DIR, augmentations=train_augs
    )
    val_dataset = FacialKeypointsDataset(
        val_df.reset_index(drop=True), config.TRAIN_IMG_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS_DATALOADER,
        drop_last=True,
        pin_memory=config.DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS_DATALOADER,
        pin_memory=config.DEVICE.type == "cuda",
    )

    return train_loader, val_loader
