import torch
from tqdm import tqdm

from config import DEVICE, CHECKPOINT_SAVE_PATH, CHECKPOINT_EPOCH_INTERVAL
import config
from model.architecture import get_loss_function
from training.checkpoint_manager import save_checkpoint


def train_one_epoch(model, dataloader, optimizer, loss_fn):
    # Set model to training mode this is important,
    # for dropout and batchnorm to work correctly
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Training Epoch", leave=False)
    for images, keypoints in progress_bar:
        # Skip if batch is problematic (e.g. from dataset returning None)
        if (
            images is None
            or keypoints is None
            or images.nelement() == 0
            or keypoints.nelement() == 0
        ):
            continue

        images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, keypoints)
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update model parameters

        total_loss += (
            loss.item()
        )  # Accumulate loss ( .item() gets Python number from tensor)
        num_batches += 1
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else float("nan")


@torch.no_grad() # Disable gradient calculations for validation
def validate_one_epoch(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Validating Epoch", leave=False)
    for images, keypoints in progress_bar:
        if (
            images is None
            or keypoints is None
            or images.nelement() == 0
            or keypoints.nelement() == 0
        ):
            continue

        images, keypoints = images.to(DEVICE), keypoints.to(DEVICE)
        predictions = model(images)
        loss = loss_fn(predictions, keypoints)

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else float("nan")


def fit(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    val_dataloader,
    start_epoch,
    num_epochs,
    train_loss_history,
    val_loss_history,
    checkpoint_save_dir,
    checkpoint_interval,
):
    """
    Main training loop
    """
    if start_epoch >= num_epochs:
        print(
            f"Model already trained for {start_epoch} epochs. Target was {num_epochs}. No further training needed."
        )
        return model, optimizer, train_loss_history, val_loss_history

    print(
        f"Starting training. Next epoch will be {start_epoch + 1}, training up to {num_epochs} epochs."
    )

    for epoch in range(start_epoch, num_epochs):
        current_epoch_display = epoch + 1  # User-friendly epoch number (1-based)
        print(f"\n--- Epoch: {current_epoch_display}/{num_epochs} ---")

        avg_train_loss = train_one_epoch(model, train_dataloader, optimizer, loss_fn)
        avg_val_loss = validate_one_epoch(model, val_dataloader, loss_fn)

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(
            f"Epoch {current_epoch_display} Summary: Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )

        # Checkpoint Saving
        if (current_epoch_display % checkpoint_interval == 0) or (
            current_epoch_display == num_epochs
        ):
            save_checkpoint(
                epoch=current_epoch_display,
                model=model,
                optimizer=optimizer,
                train_loss_history=train_loss_history,
                val_loss_history=val_loss_history,
                save_dir=checkpoint_save_dir,
            )

    print("Training finished.")
    return model, optimizer, train_loss_history, val_loss_history


def train_model(
    model, optimizer, train_loader, val_loader, start_epoch, train_history, val_history
):
    return fit(
        model=model,
        optimizer=optimizer,
        loss_fn=get_loss_function(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        start_epoch=start_epoch,
        num_epochs=config.NUM_EPOCHS,
        train_loss_history=train_history,
        val_loss_history=val_history,
        checkpoint_save_dir=config.CHECKPOINT_SAVE_PATH,
        checkpoint_interval=config.CHECKPOINT_EPOCH_INTERVAL,
    )
