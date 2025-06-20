import torch
import multiprocessing as mp 
import config # At the top
from utils.plotting import plot_loss_curves, display_predictions # At the top

# We set the start method for multiprocessing important for numworkers to work
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True) # 'spawn' is generally safer with CUDA
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if "context has already been set" not in str(e).lower(): # Only print if it's not the "already set" error
            print(f"Warning: Could not set multiprocessing start method: {e}")
            print("If using CUDA with num_workers > 0, this might lead to issues.")
        else:
            print("Multiprocessing context already set.")

import pandas as pd
import os
import glob # For finding test images
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config # Loads all configurations
from data_loader.utils import download_udacity_data, rename_keypoint_columns
from data_loader.augmentations import get_training_augmentations
from data_loader.dataset import FacialKeypointsDataset
from model.architecture import build_facial_keypoints_model, get_loss_function, get_optimizer
from training.trainer import fit
from training.checkpoint_manager import load_checkpoint
from utils.plotting import plot_loss_curves, display_predictions

def run_training_pipeline():
    """Orchestrates the full training and evaluation pipeline."""
    print(f"Using device: {config.DEVICE}")
    print("Debug Check Colab: ",config.IS_COLAB)
    if config.IS_COLAB:
        if not os.path.exists('/content/drive/MyDrive'):
            print("WARNING: Running in presumed Colab environment (IS_COLAB=True) but /content/drive/MyDrive not found.")
            print("Please ensure Google Drive is mounted in a preceding Colab cell.")

    download_udacity_data()

    # Load and Prepare Data
    try:
        raw_data_df = pd.read_csv(config.TRAIN_CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: Training CSV not found at {config.TRAIN_CSV_PATH}")
        print("Please ensure the P1_Facial_Keypoints dataset is correctly downloaded and paths in config.py are correct.")
        return

    processed_data_df = rename_keypoint_columns(raw_data_df)

    # Split data
    train_df, val_df = train_test_split(
        processed_data_df,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_STATE_DATA_SPLIT
    )
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Get augmentations
    train_augs = get_training_augmentations()

    # Create Datasets
    train_dataset = FacialKeypointsDataset(
        train_df.reset_index(drop=True),
        config.TRAIN_IMG_DIR,
        augmentations=train_augs
    )
    val_dataset = FacialKeypointsDataset(
        val_df.reset_index(drop=True),
        config.TRAIN_IMG_DIR
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS_DATALOADER, # Using config
        drop_last=True, # Important for consistent batch sizes because batchnorm layers depends on it
        pin_memory=True if config.DEVICE.type == 'cuda' else False # Speeds up CPU to GPU transfer
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS_DATALOADER,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    # Initialize Model, Loss, Optimizer
    model = build_facial_keypoints_model(pretrained=True)
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model, learning_rate=config.LEARNING_RATE)

    # Load Checkpoint
    model, optimizer, start_epoch, train_loss_history, val_loss_history = load_checkpoint(
        model=model,
        optimizer=optimizer,
        save_dir=config.CHECKPOINT_SAVE_PATH,
        learning_rate=config.LEARNING_RATE
    )
    
    # model summary (optional)
    # try:
    #     from torchsummary import summary
    #     summary(model, (3, config.IMG_HEIGHT, config.IMG_WIDTH))
    # except ImportError:
    #     print("torchsummary not installed. Skipping model summary.")


    # Training
    model, optimizer, train_loss_history, val_loss_history = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        start_epoch=start_epoch,
        num_epochs=config.NUM_EPOCHS,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        checkpoint_save_dir=config.CHECKPOINT_SAVE_PATH,
        checkpoint_interval=config.CHECKPOINT_EPOCH_INTERVAL
    )

    # Plotting Results
    print("\n--- Plotting Loss Curves ---")
    plot_loss_curves(
        train_loss_history,
        val_loss_history,
        save_dir=config.CHECKPOINT_SAVE_PATH # Pass the configured save directory
    )
    
    # Test on a few sample images from the original test set (or validation set)
    print("\n--- Displaying predictions on a few validation/test images ---")
    if not val_df.empty:
        num_test_samples = min(3, len(val_df))
        sample_image_info = val_df.sample(n=num_test_samples, random_state=config.RANDOM_STATE_DATA_SPLIT if hasattr(config, 'RANDOM_STATE_DATA_SPLIT') else None)
        
        for _, row in sample_image_info.iterrows():
            image_filename = row['image_file_name']
            image_path = os.path.join(config.TRAIN_IMG_DIR, image_filename) # Images from training split for validation
            print(f"Displaying prediction for: {image_path}")
            if os.path.exists(image_path):
                display_predictions(
                    image_path_str=image_path,
                    model=model,
                    device_to_use=config.DEVICE,
                    save_dir=config.CHECKPOINT_SAVE_PATH # Pass the save directory
                )
            else:
                print(f"Warning: Sample image for display not found: {image_path}")
    else:
        print("No validation images to display predictions on.")

if __name__ == '__main__':
    run_training_pipeline()
