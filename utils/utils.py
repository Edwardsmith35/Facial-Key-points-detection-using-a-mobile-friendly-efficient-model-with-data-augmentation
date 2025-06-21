import os

import cv2
from matplotlib import pyplot as plt, transforms
import torch
import config


def check_colab_mount():
    if config.IS_COLAB and not os.path.exists("/content/drive/MyDrive"):
        print("WARNING: IS_COLAB=True but /content/drive/MyDrive not found.")
        print("Ensure Google Drive is mounted.")


def display_predictions(
    image_path_str: str,
    model: torch.nn.Module,
    device_to_use=config.DEVICE,
    target_img_width: int = config.IMG_WIDTH,
    target_img_height: int = config.IMG_HEIGHT,
    save_dir=None,
    filename_prefix: str = "prediction_",
):
    """
    Loads an image, gets model predictions, displays image with keypoints,
    and optionally saves the plot.
    """
    img_bgr = cv2.imread(image_path_str)
    if img_bgr is None:
        print(f"Error: Could not load image from {image_path_str}")
        return

    img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HWC uint8 RGB

    img_display_resized_rgb = cv2.resize(
        img_rgb_original, (target_img_width, target_img_height)
    )

    # preprocessing the same as in dataset's __getitem__
    img_for_model_resized_rgb = cv2.resize(
        img_rgb_original, (target_img_width, target_img_height)
    )
    img_float_0_1 = img_for_model_resized_rgb / 255.0
    img_tensor_chw = torch.tensor(img_float_0_1, dtype=torch.float32).permute(2, 0, 1)

    imagenet_normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    img_tensor_normalized = imagenet_normalizer(img_tensor_chw)

    # Add batch dimension and move to device
    img_ready_for_model = img_tensor_normalized.unsqueeze(0).to(device_to_use)

    model.eval()
    model.to(device_to_use)  # Ensure model is on the correct device

    with torch.no_grad():
        # Output is expected to be [1, NUM_KEYPOINTS*2]
        predicted_kps_normalized_flat = (
            model(img_ready_for_model).squeeze(0).detach().cpu().numpy()
        )

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(img_display_resized_rgb)

    image_basename = os.path.basename(image_path_str)
    plt.title(f"Keypoint Predictions for {image_basename}")

    # Denormalize keypoints: predictions are in [0,1] relative to target_img_width/height
    # The first NUM_KEYPOINTS values are x, the next NUM_KEYPOINTS are y
    kp_x_pixels = (
        predicted_kps_normalized_flat[: config.NUM_KEYPOINTS] * target_img_width
    )
    kp_y_pixels = (
        predicted_kps_normalized_flat[config.NUM_KEYPOINTS :] * target_img_height
    )

    plt.scatter(
        kp_x_pixels,
        kp_y_pixels,
        c="lime",
        s=30,
        marker="o",
        edgecolors="black",
        linewidths=0.7,
    )
    plt.axis("off")

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


def display_sample_predictions(val_df, model):
    print("\n--- Displaying predictions on a few validation/test images ---")
    if val_df.empty:
        print("No validation images to display predictions on.")
        return

    num_samples = min(3, len(val_df))
    sample_df = val_df.sample(
        n=num_samples, random_state=getattr(config, "RANDOM_STATE_DATA_SPLIT", None)
    )

    for _, row in sample_df.iterrows():
        image_path = os.path.join(config.TRAIN_IMG_DIR, row["image_file_name"])
        print(f"Displaying prediction for: {image_path}")
        if os.path.exists(image_path):
            display_predictions(
                image_path_str=image_path,
                model=model,
                device_to_use=config.DEVICE,
                save_dir=config.CHECKPOINT_SAVE_PATH,
            )
        else:
            print(f"Warning: Image not found: {image_path}")


def show_model_summary(model):
    try:
        from torchsummary import summary

        summary(model, (3, config.IMG_HEIGHT, config.IMG_WIDTH))
    except ImportError:
        print("torchsummary not installed. Skipping model summary.")
