import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
# from torchsummary import summary
import numpy as np, pandas as pd, os, cv2, glob
import random
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#%matplotlib inline # Typically needed for Jupyter, but can be commented out if causing issues in other envs
from torch.utils.data import DataLoader, Dataset

# --- Import Albumentations ---
import albumentations as A
# from albumentations.pytorch import ToTensorV2 # We are doing manual tensor conversion for now

# --- Configuration for Saving/Loading ---
from google.colab import drive
drive.mount('/content/drive')

MODEL_SAVE_PATH = '/content/drive/MyDrive/Facial_KeyPoints_Detection_MobileNetV3_Aug/'
# OPTIMIZER_SAVE_PATH = 'optimizer_mobilenet.pth' # Combined into MODEL_SAVE_PATH
# METADATA_SAVE_PATH = 'training_metadata_mobilenet.json' # Combined into MODEL_SAVE_PATH
CHECKPOINT_EPOCH_INTERVAL = 10
# --- End Configuration ---

# Get Data and read it
if not os.path.exists('P1_Facial_Keypoints'):
    !git clone https://github.com/udacity/P1_Facial_Keypoints.git
else:
    print("'P1_Facial_Keypoints' directory already exists.")

root_dir = '/content/P1_Facial_Keypoints/data/training/'
test_data_dir_for_val = '/content/P1_Facial_Keypoints/data/test/' # Used for validation set
path_for_inspect = '/content/P1_Facial_Keypoints/data/training/' # Path for initial inspection

data = pd.read_csv('/content/P1_Facial_Keypoints/data/training_frames_keypoints.csv')

# inspect an image
image_path_list_inspect = glob.glob(os.path.join(path_for_inspect, 'Luis_Fonsi_21.jpg'))
if image_path_list_inspect:
    print(f"Inspecting: {image_path_list_inspect[0]}")
    img = cv2.imread(image_path_list_inspect[0])
    if img is not None:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_RGB)
        plt.title("Sample Image for Inspection")
        plt.show()
    else:
        print(f"Failed to load inspection image: {image_path_list_inspect[0]}")
else:
    print("Inspection image not found.")

# Change the Dataframe Column Names
new_columns = data.columns.tolist()
idx_col_rename = 0 # Use a different variable name than global 'index'
for i in range(1,len(new_columns),2):
    idx_col_rename +=1
    new_columns[i] = f'x{idx_col_rename}'
    new_columns[i+1] = f'y{idx_col_rename}'
new_columns[0] = 'image_file_name'
data.columns = new_columns

# Creating Face_Dataset Class:
class Faces_Dataset(Dataset):
    def __init__(self, df, image_base_path, augmentations=None): # Added augmentations
        super(Faces_Dataset).__init__()
        self.image_base_path = image_base_path
        self.normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
        self.augmentations = augmentations

        self.image_filenames = []
        self.keypoints_list_of_arrays = [] # Store keypoints as numpy arrays

        print(f"Initializing Dataset. Verifying {len(df)} image paths from DataFrame...")
        for i in range(len(df)):
            filename = df.iloc[i, 0]
            fpath_full = os.path.join(self.image_base_path, filename)
            if os.path.exists(fpath_full):
                self.image_filenames.append(filename)
                self.keypoints_list_of_arrays.append(df.iloc[i, 1:].values.astype('float32'))
            else:
                print(f"Warning: Image file not found {fpath_full}. Skipping row {i} (filename: {filename}).")
        self.df_is_now_filtered_implicitly_by_using_lists = True # Info only
        print(f"Dataset initialized with {len(self.image_filenames)} valid samples.")


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, ix):
        image_filename = self.image_filenames[ix]
        image_path_full = os.path.join(self.image_base_path, image_filename)

        img_cv2_original_bgr = cv2.imread(image_path_full)

        if img_cv2_original_bgr is None:
            print(f"ERROR in __getitem__: Could not read image {image_path_full}. Returning zeros.")
            return torch.zeros((3,224,224), device=device), torch.zeros((136), device=device)

        img_original_rgb = cv2.cvtColor(img_cv2_original_bgr, cv2.COLOR_BGR2RGB) # HWC, RGB, uint8

        # Get keypoints as list of [x,y] pixel coordinates for Albumentations
        kp_values_flat_pixels = self.keypoints_list_of_arrays[ix] # Already a float32 numpy array
        keypoints_for_aug = []
        for i in range(0, len(kp_values_flat_pixels), 2):
            keypoints_for_aug.append([kp_values_flat_pixels[i], kp_values_flat_pixels[i+1]])

        # --- Apply Augmentations (if any) ---
        img_augmented_rgb = img_original_rgb # HWC, uint8
        keypoints_augmented_pixels_list_of_lists = keypoints_for_aug # List of [x,y]

        if self.augmentations:
            try:
                augmented_data = self.augmentations(image=img_original_rgb, keypoints=keypoints_for_aug)
                img_augmented_rgb = augmented_data['image']
                keypoints_augmented_pixels_list_of_lists = augmented_data['keypoints']
            except Exception as e:
                print(f"Error during augmentation for {image_filename}: {e}. Using original.")
                # Fallback to original if augmentation fails

        # --- Image Processing for Model Input ---
        # Resize the (potentially augmented) image
        # img_augmented_rgb is HWC, uint8 (or float if aug changed it, albumentations usually keeps uint8)
        img_resized_rgb = cv2.resize(img_augmented_rgb, (224, 224))
        img_normalized_0_1 = img_resized_rgb / 255.0 # HWC, RGB, float (0-1)
        img_tensor_chw = torch.tensor(img_normalized_0_1).float().permute(2,0,1)
        img_final_for_model = self.normalize_imagenet(img_tensor_chw)

        # --- Keypoint Post-Processing (after augmentation) ---
        # Keypoints from Albumentations ('keypoints_augmented_pixels_list_of_lists')
        # are pixel coordinates on `img_augmented_rgb`.
        # We need to normalize them based on the dimensions of `img_augmented_rgb`
        # *before* it was resized to 224x224. These normalized 0-1 keypoints
        # will then be valid for the 224x224 image fed to the model.
        
        aug_h, aug_w, _ = img_augmented_rgb.shape # Dimensions of the image after augmentation

        kp_x_normalized_list = []
        kp_y_normalized_list = []

        num_expected_keypoints = 68
        if not keypoints_augmented_pixels_list_of_lists or len(keypoints_augmented_pixels_list_of_lists) < num_expected_keypoints:
            # print(f"Warning: Not enough keypoints after augmentation for {image_filename}. Padding with 0.5.")
            # This padding is a simple strategy. Better might be to skip such samples if too many KPs are lost.
            for _ in range(num_expected_keypoints):
                kp_x_normalized_list.append(0.5)
                kp_y_normalized_list.append(0.5)
        else:
            for i in range(num_expected_keypoints): # Iterate up to expected number
                if i < len(keypoints_augmented_pixels_list_of_lists):
                    kp_pair_aug = keypoints_augmented_pixels_list_of_lists[i]
                    x_norm = kp_pair_aug[0] / aug_w
                    y_norm = kp_pair_aug[1] / aug_h
                    kp_x_normalized_list.append(np.clip(x_norm, 0.0, 1.0))
                    kp_y_normalized_list.append(np.clip(y_norm, 0.0, 1.0))
                else: # Should not happen if remove_invisible=False and we handle empty list
                    kp_x_normalized_list.append(0.5)
                    kp_y_normalized_list.append(0.5)
        
        kp_concatenated_normalized = np.concatenate((kp_x_normalized_list, kp_y_normalized_list))
        keypoints_tensor = torch.tensor(kp_concatenated_normalized).float()

        return img_final_for_model.to(device), keypoints_tensor.to(device)

# Split data
train_df, test_df = train_test_split(data, test_size=0.2, random_state=101)

# --- Define Augmentations ---
train_augs = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.06, scale_limit=0.1, rotate_limit=20, p=0.7, # Increased rotate limit
        border_mode=cv2.BORDER_CONSTANT, value=0 # Fill new pixels with black
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.3),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)) # remove_invisible=False is important for fixed number of keypoints

# Create Datasets using the correct base paths and augmentations
train_dataset = Faces_Dataset(train_df.reset_index(drop=True), root_dir, augmentations=train_augs)
test_dataset = Faces_Dataset(test_df.reset_index(drop=True), root_dir, augmentations=None) # No augmentations for test/validation

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True) # Added num_workers
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False) # Added num_workers

def get_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True # Fine-tuning all parameters

    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256,136),
        nn.Sigmoid()
      )
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Using the LR that worked well for you
    return model.to(device) , optimizer, loss_function

# --- train_batch and validate_batch remain the same ---
def train_batch(model, images, kps, optimizer, loss_func):
    model.train()
    _kps = model(images)
    loss = loss_func(_kps, kps)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

@torch.no_grad()
def validate_batch(model, images, kps, loss_func):
    model.eval()
    _kps = model(images)
    loss = loss_func(_kps, kps)
    return loss.item()

# --- Initialize model, optimizer, loss ---
model, optimizer, loss_func = get_model()

# --- torchsummary (optional, install if needed) ---
# !pip install -q torchsummary
# from torchsummary import summary
# summary(model, (3,224,224))

# --- Attempt to Load Checkpoint ---
start_epoch = 0
train_epochs_losses = []
val_epochs_losses = []

# Ensure the save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Find the latest checkpoint file
latest_checkpoint_epoch = -1
latest_checkpoint_path = None
for filename in os.listdir(MODEL_SAVE_PATH):
    if filename.startswith('facial_keypoints_mobilenet_') and filename.endswith('.pth'):
        try:
            # Extract epoch number from filename (e.g., 'facial_keypoints_mobilenet_10.pth')
            epoch_str = filename.replace('facial_keypoints_mobilenet_', '').replace('.pth', '')
            epoch_num = int(epoch_str)
            if epoch_num > latest_checkpoint_epoch:
                latest_checkpoint_epoch = epoch_num
                latest_checkpoint_path = os.path.join(MODEL_SAVE_PATH, filename)
        except ValueError:
            print(f"Warning: Could not parse epoch from filename: {filename}")

if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
    print(f"Loading checkpoint from {latest_checkpoint_path}...")
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Re-initialize optimizer with current model parameters THEN load state
        # Make sure parameters passed to Adam are those that have requires_grad=True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) # Match LR
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', latest_checkpoint_epoch) # Use epoch from checkpoint if available, fallback to filename
        train_epochs_losses = checkpoint.get('train_loss_history', [])
        val_epochs_losses = checkpoint.get('val_loss_history', [])
        print(f"Resuming training from epoch {start_epoch}") # Display epoch number resuming *from*
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 0
        train_epochs_losses = []
        val_epochs_losses = []
        # Ensure model and optimizer are freshly initialized if load fails badly
        model, optimizer, loss_func = get_model()
else:
    print("No checkpoint found or latest checkpoint path invalid. Starting training from scratch.")


# Training Loop:
n_epochs_total = 100 # Define total desired epochs (adjust as needed)

if start_epoch >= n_epochs_total:
    print(f"Model already trained for {start_epoch} epochs. Target was {n_epochs_total}.")
else:
    for epoch in range(start_epoch, n_epochs_total):
        current_epoch_display = epoch + 1
        print(f"Epoch: {current_epoch_display}/{n_epochs_total}")
        
        model.train()
        train_batch_losses = []
        for ix, batch_data in enumerate(iter(train_dataloader)):
            if batch_data is None : continue
            images, kps = batch_data
            if images is None or kps is None : continue
            if images.nelement() == 0 or kps.nelement() == 0: continue # Skip empty tensors
            
            batch_loss = train_batch(model, images, kps, optimizer, loss_func)
            train_batch_losses.append(batch_loss)
        
        epoch_train_loss = np.mean(train_batch_losses) if train_batch_losses else float('nan') # Use nan for empty
        
        model.eval()
        val_batch_losses = []
        for ix, batch_data in enumerate(iter(test_dataloader)):
            if batch_data is None : continue
            images, kps = batch_data
            if images is None or kps is None : continue
            if images.nelement() == 0 or kps.nelement() == 0: continue # Skip empty tensors

            batch_val_loss = validate_batch(model, images, kps, loss_func)
            val_batch_losses.append(batch_val_loss)
        
        epoch_val_loss = np.mean(val_batch_losses) if val_batch_losses else float('nan')

        train_epochs_losses.append(epoch_train_loss)
        val_epochs_losses.append(epoch_val_loss)

        print(f"Epoch [{current_epoch_display}/{n_epochs_total}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        if (epoch + 1) % CHECKPOINT_EPOCH_INTERVAL == 0 or (epoch + 1) == n_epochs_total:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss_history': train_epochs_losses,
                'val_loss_history': val_epochs_losses,
            }
            # Use the constructed save_path which includes the filename
            save_path = os.path.join(MODEL_SAVE_PATH, f'facial_keypoints_mobilenet_{epoch+1}.pth') # Use epoch+1 for filename consistency
            # Ensure the directory exists before saving - already done before the loop, but harmless here
            os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
            torch.save(checkpoint_data, save_path) # Corrected the save path
            print("Checkpoint saved.")


# Plotting
# Correct the epochs_to_plot to match the number of recorded losses
epochs_to_plot = np.arange(len(train_epochs_losses)) + 1

# import matplotlib.ticker as mtick # Not strictly needed again if imported once
# import matplotlib.ticker as mticker # Not strictly needed again

plt.figure(figsize=(12,6)) # Adjusted figure size
plt.plot(epochs_to_plot, train_epochs_losses, 'bo-', label='Training loss', markersize=4, linewidth=1.5)
plt.plot(epochs_to_plot, val_epochs_losses, 'r-', label='Validation loss', markersize=4, linewidth=1.5)
plt.title('Training and Validation Loss Over Epochs (with Augmentation)')
plt.xlabel('Epochs')
plt.ylabel('Loss (L1)')
plt.legend()
plt.grid(True)
plt.minorticks_on() # Add minor ticks for better readability
plt.show()


# test_model function (using your latest improved version, with slight mods)
def test_model(image_path_str, model_to_test, test_device):
    img_original_bgr = cv2.imread(image_path_str)
    if img_original_bgr is None:
        print(f"Error loading {image_path_str}")
        return

    img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB) # HWC uint8 RGB

    # For display (resized to 224x224, keep as uint8 for direct imshow if not scaling pixels)
    img_display_resized_rgb = cv2.resize(img_original_rgb, (224,224))

    # For model input
    # 1. Resize original RGB
    img_for_model_resized_rgb = cv2.resize(img_original_rgb, (224,224))
    # 2. Normalize pixels to 0-1 float
    img_for_model_0_1_float = img_for_model_resized_rgb / 255.0
    # 3. To Tensor CHW
    img_tensor_chw = torch.tensor(img_for_model_0_1_float).float().permute(2, 0, 1)
    # 4. ImageNet Normalize
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    img_ready_for_model = normalize_transform(img_tensor_chw).to(test_device)

    plt.figure(figsize=(6,6))
    plt.imshow(img_display_resized_rgb) # Displaying uint8 RGB image
    plt.title('Image Keypoints Prediction')

    model_to_test.eval()
    model_to_test.to(test_device)

    with torch.no_grad():
        predicted_kps_normalized = model_to_test(img_ready_for_model.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    # Keypoints are predicted in [0,1] range relative to 224x224 input
    # Scale them to the display image size (which is also 224x224)
    kp_x_pixels = predicted_kps_normalized[:68] * 224
    kp_y_pixels = predicted_kps_normalized[68:] * 224

    plt.scatter(kp_x_pixels, kp_y_pixels, c='lime', s=25, marker='o', edgecolors='black', linewidths=0.5)
    plt.axis('off')
    plt.show()

# Test the model after training:
path_for_final_test = '/content/P1_Facial_Keypoints/data/test/'

test_image_files = glob.glob(os.path.join(path_for_final_test, '*.jpg'))
if test_image_files:
    # Test a few random images
    for _ in range(min(3, len(test_image_files))): # Test up to 3 images
        random_image_path = random.choice(test_image_files)
        print(f"Testing model on: {random_image_path}")
        test_model(random_image_path, model, device)
else:
    print(f"No jpg images found in {path_for_final_test} for testing.")