import os
import torch
from config import DEVICE, CHECKPOINT_FILENAME_PREFIX


def save_checkpoint(
    epoch,
    model,
    optimizer,
    train_loss_history,
    val_loss_history,
    save_dir,
    filename_prefix=CHECKPOINT_FILENAME_PREFIX,
):
    """Saves model checkpoint."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f"{filename_prefix}{epoch}.pth")
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer,
    save_dir,
    filename_prefix=CHECKPOINT_FILENAME_PREFIX,
    learning_rate=None,
):
    """
    Loads the latest model checkpoint if available.
    Returns the model, optimizer, starting epoch, and loss histories.
    """
    latest_epoch = -1
    latest_checkpoint_file = None

    if not os.path.isdir(save_dir):
        print(f"Checkpoint directory {save_dir} not found. Starting from scratch.")
        return model, optimizer, 0, [], []

    for f_name in os.listdir(save_dir):
        if f_name.startswith(filename_prefix) and f_name.endswith(".pth"):
            try:
                epoch_str = f_name.replace(filename_prefix, "").replace(".pth", "")
                epoch_num = int(epoch_str)
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint_file = os.path.join(save_dir, f_name)
            except ValueError:
                print(f"Warning: Could not parse epoch from filename: {f_name}")

    if latest_checkpoint_file:
        print(f"Loading checkpoint from {latest_checkpoint_file}...")
        try:
            checkpoint = torch.load(latest_checkpoint_file, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])

            # Re-initialize optimizer with potentially new LR from config BEFORE loading state
            # This is important if we change LR between runs but want to load optimizer state
            current_lr = (
                optimizer.param_groups[0]["lr"] if optimizer else learning_rate
            )
            if (
                optimizer is None and learning_rate is not None
            ):  # If optimizer not passed, create it
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=learning_rate,
                )
            elif (
                optimizer and learning_rate is not None and current_lr != learning_rate
            ):  # If LR changed in config
                print(f"Updating optimizer LR from {current_lr} to {learning_rate}")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint.get(
                "epoch", latest_epoch
            )  # epoch to resume FROM (so next epoch is start_epoch)
            train_loss_history = checkpoint.get("train_loss_history", [])
            val_loss_history = checkpoint.get("val_loss_history", [])
            print(f"Resuming training. Next epoch will be {start_epoch + 1}.")
            return model, optimizer, start_epoch, train_loss_history, val_loss_history
        except Exception as e:
            print(
                f"Error loading checkpoint {latest_checkpoint_file}: {e}. Starting from scratch."
            )
    else:
        print("No checkpoint found. Starting training from scratch.")

    # If starting from scratch or load failed, ensure optimizer is fresh if needed
    if optimizer is None and learning_rate is not None:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
        )

    return model, optimizer, 0, [], []
