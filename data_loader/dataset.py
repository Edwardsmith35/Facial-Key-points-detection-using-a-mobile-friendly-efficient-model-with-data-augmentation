import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_WIDTH, IMG_HEIGHT, NUM_KEYPOINTS

class FacialKeypointsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_base_path: str, augmentations=None, 
                 target_img_width: int = IMG_WIDTH, target_img_height: int = IMG_HEIGHT):
        super().__init__()
        self.df = df
        self.image_base_path = image_base_path
        self.augmentations = augmentations
        self.target_img_width = target_img_width
        self.target_img_height = target_img_height
        
        # Essential for Transfer learning
        self.imagenet_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self._filter_valid_samples()

    def _filter_valid_samples(self):
        """Filters the DataFrame to keep only rows with existing image files"""
        self.image_filenames = []
        self.keypoints_data = []

        print(f"Initializing Dataset. Verifying {len(self.df)} image paths from DataFrame...")
        valid_indices = []
        for i, row in self.df.iterrows():
            filename = row['image_file_name']
            fpath_full = os.path.join(self.image_base_path, filename)
            if os.path.exists(fpath_full):
                valid_indices.append(i)
                self.image_filenames.append(filename)
                # keypoints starts from the second column in this Sequence: (x1, y1, x2, y2...)
                self.keypoints_data.append(row.iloc[1:].values.astype('float32'))
            else:
                print(f"Warning: Image file not found {fpath_full}. Skipping (filename: {filename}).")
        
        print(f"Dataset initialized with {len(self.image_filenames)} valid samples.")


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, ix: int):
        image_filename = self.image_filenames[ix]
        image_path_full = os.path.join(self.image_base_path, image_filename)

        img_bgr = cv2.imread(image_path_full)
        if img_bgr is None:
            print(f"ERROR in __getitem__: Could not read image {image_path_full}. Returning zeros.")
            # Return tensors of correct shape and type, moving to the correct device is handled by the data_loader
            return (torch.zeros((3, self.target_img_height, self.target_img_width), dtype=torch.float32),
                    torch.full((NUM_KEYPOINTS * 2,), 0.5, dtype=torch.float32)) # Pad with 0.5

        img_rgb_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Shape: (height, width, Channels), RGB, uint8

        # Keypoints are flat [x1,y1,x2,y2,...] from self.keypoints_data
        keypoints_flat_pixels = self.keypoints_data[ix]
        keypoints_for_albumentations = []
        for i in range(0, len(keypoints_flat_pixels), 2):
            keypoints_for_albumentations.append([keypoints_flat_pixels[i], keypoints_flat_pixels[i+1]])

        img_processed_rgb = img_rgb_original
        keypoints_processed_pixels = keypoints_for_albumentations

        if self.augmentations:
            try:
                augmented = self.augmentations(image=img_rgb_original, keypoints=keypoints_for_albumentations)
                img_processed_rgb = augmented['image']
                keypoints_processed_pixels = augmented['keypoints']
            except Exception as e:
                print(f"Warning: Augmentation failed for {image_filename}: {e}, Using original image and keypoints")
                # Fallback handled by using img_rgb_original and keypoints_for_albumentations

        # Image to Tensor and Normalize for Model
        # img_processed_rgb is HWC
        img_resized_rgb = cv2.resize(img_processed_rgb, (self.target_img_width, self.target_img_height))
        img_float_0_1 = img_resized_rgb / 255.0  # HWC, RGB, float (0-1)
        img_tensor_chw = torch.tensor(img_float_0_1, dtype=torch.float32).permute(2, 0, 1) # permute from HWC to CHW
        img_tensor_normalized = self.imagenet_normalizer(img_tensor_chw)

        # Keypoints to Tensor and Normalize

        # Keypoints from Albumentations are pixel coordinates on img_processed_rgb
        # We Normalize them based on the dimensions of img_processed_rgb (image after augmentation but before final resize)
        aug_h, aug_w, _ = img_processed_rgb.shape
        
        normalized_keypoints_flat = []
        
        # Ensure we have the correct number of keypoints, padding if necessary after augmentation
        num_visible_keypoints = len(keypoints_processed_pixels)

        for i in range(NUM_KEYPOINTS):
            if i < num_visible_keypoints and keypoints_processed_pixels[i] is not None:
                kp_pair_aug = keypoints_processed_pixels[i]
                # Clip keypoints to be within image bounds before normalization
                x_clipped = np.clip(kp_pair_aug[0], 0, aug_w -1) # -1 to avoid being exactly on boundary
                y_clipped = np.clip(kp_pair_aug[1], 0, aug_h -1)
                
                x_norm = x_clipped / aug_w
                y_norm = y_clipped / aug_h
                normalized_keypoints_flat.extend([x_norm, y_norm])
            else:
                # Pad with 0.5 (center) if keypoint is missing or invisible after augmentation
                normalized_keypoints_flat.extend([0.5, 0.5]) 
        
        keypoints_tensor = torch.tensor(normalized_keypoints_flat, dtype=torch.float32)

        return img_tensor_normalized, keypoints_tensor