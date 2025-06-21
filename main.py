import multiprocessing as mp
import config
from model.utils import setup_model
from utils.plotting import plot_training_curves
from data_loader.utils import (
    create_dataloaders,
    download_udacity_data,
    load_training_data,
    rename_keypoint_columns,
    split_data,
)
from training.trainer import train_model
from utils.utils import (
    check_colab_mount,
    display_sample_predictions,
    show_model_summary,
)


def run_training_pipeline():
    print(f"Using device: {config.DEVICE}")
    check_colab_mount()
    download_udacity_data()
    raw_data_df = load_training_data()
    processed_data_df = rename_keypoint_columns(raw_data_df)

    train_df, val_df = split_data(processed_data_df)
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    train_loader, val_loader = create_dataloaders(train_df, val_df)
    model, optimizer, start_epoch, train_history, val_history = setup_model()

    show_model_summary(model)

    model, optimizer, train_history, val_history = train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        start_epoch,
        train_history,
        val_history,
    )

    plot_training_curves(train_history, val_history)
    display_sample_predictions(val_df, model)


if __name__ == "__main__":
    # We set the start method for multiprocessing important for numworkers to work
    try:
        mp.set_start_method("spawn", force=True)  # 'spawn' is generally safer with CUDA
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if (
            "context has already been set" not in str(e).lower()
        ):  # Only print if it's not the "already set" error
            print(f"Warning: Could not set multiprocessing start method: {e}")
            print("If using CUDA with num_workers > 0, this might lead to issues.")
        else:
            print("Multiprocessing context already set.")
    run_training_pipeline()
