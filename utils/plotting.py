import matplotlib.pyplot as plt
import numpy as np
import os

from config import DEVICE, IMG_WIDTH, IMG_HEIGHT, NUM_KEYPOINTS, CHECKPOINT_SAVE_PATH
import config


def plot_loss_curves(
    train_losses, val_losses, save_dir=None, filename="loss_curves.png"
):
    """
    Plots training and validation loss curves and optionally saves the plot.
    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        save_dir (str, optional): Directory to save the plot. If None, plot is not saved.
                                  Defaults to None.
        filename (str, optional): Filename for the saved plot. Defaults to "loss_curves.png".
    """
    if not train_losses and not val_losses:
        print("No loss data to plot.")
        return

    if train_losses:
        epochs_to_plot = np.arange(len(train_losses)) + 1
    elif val_losses:
        epochs_to_plot = np.arange(len(val_losses)) + 1
    else:
        return

    plt.figure(figsize=(12, 6))
    if train_losses:
        plt.plot(
            epochs_to_plot,
            train_losses,
            "bo-",
            label="Training loss",
            markersize=4,
            linewidth=1.5,
        )
    if val_losses:
        plt.plot(
            epochs_to_plot,
            val_losses,
            "ro-",
            label="Validation loss",
            markersize=4,
            linewidth=1.5,
        )

    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (L1)")

    if train_losses or val_losses:
        plt.legend()

    plt.grid(True)
    plt.minorticks_on()

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plot_full_path = os.path.join(save_dir, filename)
        try:
            plt.savefig(plot_full_path)
            print(f"Loss curves saved to {plot_full_path}")
        except Exception as e:
            print(f"Error saving loss curves plot: {e}")

    plt.show()
    plt.close()


def plot_training_curves(train_history, val_history):
    print("\n--- Plotting Loss Curves ---")
    plot_loss_curves(train_history, val_history, save_dir=config.CHECKPOINT_SAVE_PATH)
