
from model.architecture import build_facial_keypoints_model, get_loss_function, get_optimizer
import config
from training.checkpoint_manager import load_checkpoint

def setup_model():
    model = build_facial_keypoints_model(pretrained=True)
    loss_fn = get_loss_function()
    optimizer = get_optimizer(model, learning_rate=config.LEARNING_RATE)
    model, optimizer, start_epoch, train_loss_history, val_loss_history = load_checkpoint(
        model=model,
        optimizer=optimizer,
        save_dir=config.CHECKPOINT_SAVE_PATH,
        learning_rate=config.LEARNING_RATE
    )
    return model, optimizer, start_epoch, train_loss_history, val_loss_history

