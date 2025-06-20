import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
import os

from config import DEVICE, IMG_WIDTH, IMG_HEIGHT, NUM_KEYPOINTS, CHECKPOINT_SAVE_PATH

def plot_loss_curves(train_losses, val_losses, save_dir=None, filename="loss_curves.png"):
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
        plt.plot(epochs_to_plot, train_losses, 'bo-', label='Training loss', markersize=4, linewidth=1.5)
    if val_losses:
        plt.plot(epochs_to_plot, val_losses, 'ro-', label='Validation loss', markersize=4, linewidth=1.5) # Changed to 'ro-' for better visibility
    
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (L1)')
    
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
    
    plt.show() # Show plot in notebook environments / interactive sessions, may Fail that's why we're saving the plots
    plt.close() # Close the figure to free up memory

def display_predictions(
    image_path_str: str,
    model: torch.nn.Module,
    device_to_use=DEVICE,
    target_img_width: int = IMG_WIDTH,
    target_img_height: int = IMG_HEIGHT,
    save_dir=None,
    filename_prefix: str = "prediction_"
):
    """
    Loads an image, gets model predictions, displays image with keypoints,
    and optionally saves the plot.
    """
    img_bgr = cv2.imread(image_path_str)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path_str}")
        return

    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # HWC uint8 RGB

    # Image for display (resized)
    img_display_resized_rgb = cv2.resize(img_rgb_original, (target_img_width, target_img_height))

    # Image for model input (preprocessing the same as in dataset's __getitem__)
    img_for_model_resized_rgb = cv2.resize(img_rgb_original, (target_img_width, target_img_height))
    img_float_0_1 = img_for_model_resized_rgb / 255.0
    img_tensor_chw = torch.tensor(img_float_0_1, dtype=torch.float32).permute(2, 0, 1)
    
    imagenet_normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    img_tensor_normalized = imagenet_normalizer(img_tensor_chw)
    # Add batch dimension and move to device
    img_ready_for_model = img_tensor_normalized.unsqueeze(0).to(device_to_use)

    model.eval() # Set model to evaluation mode
    model.to(device_to_use) # Ensure model is on the correct device

    with torch.no_grad():
        # Output is expected to be [1, NUM_KEYPOINTS*2]
        predicted_kps_normalized_flat = model(img_ready_for_model).squeeze(0).detach().cpu().numpy() # squeeze removes the extra batch dimension
   
    # Plotting
    plt.figure(figsize=(8, 8)) 
    plt.imshow(img_display_resized_rgb)
    
    image_basename = os.path.basename(image_path_str)
    plt.title(f'Keypoint Predictions for {image_basename}')
    
    # Denormalize keypoints: predictions are in [0,1] relative to target_img_width/height
    # The first NUM_KEYPOINTS values are x, the next NUM_KEYPOINTS are y
    kp_x_pixels = predicted_kps_normalized_flat[:NUM_KEYPOINTS] * target_img_width
    kp_y_pixels = predicted_kps_normalized_flat[NUM_KEYPOINTS:] * target_img_height
    
    plt.scatter(kp_x_pixels, kp_y_pixels, c='lime', s=30, marker='o', edgecolors='black', linewidths=0.7)
    plt.axis('off') #  To turn off axis numbers and ticks

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        base_img_name_no_ext = os.path.splitext(image_basename)[0]
        pred_filename = f"{filename_prefix}{base_img_name_no_ext}.png"
        plot_full_path = os.path.join(save_dir, pred_filename)
        try:
            plt.savefig(plot_full_path)
            print(f"Prediction plot saved to {plot_full_path}")
        except Exception as e:
            print(f"Error saving prediction plot: {e}")

    plt.show() 
    plt.close()
