import torch
import os

# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Paths ---
DATA_ROOT_DIR = 'P1_Facial_Keypoints/data/'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT_DIR, 'training/')
TRAIN_CSV_PATH = os.path.join(DATA_ROOT_DIR, 'training_frames_keypoints.csv')

# --- Model & Training Configuration ---
MODEL_ARCHITECTURE = 'mobilenet_v3_small'
PRETRAINED_WEIGHTS = 'MobileNet_V3_Small_Weights.IMAGENET1K_V1' 
IMG_WIDTH = 224
IMG_HEIGHT = 224
NUM_KEYPOINTS = 68 # 68 (x,y) pairs = 136 output values

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 11
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE_DATA_SPLIT = 101 # to replicate the same training run multiple times

# --- Checkpoint Configuration ---
IS_COLAB = False

BASE_SAVE_DIR = '/content/drive/MyDrive/' if IS_COLAB else 'outputs/'
MODEL_SAVE_DIR_NAME = 'Facial_KeyPoints_Detection_MobileNetV3_Aug'
CHECKPOINT_SAVE_PATH = os.path.join(BASE_SAVE_DIR, MODEL_SAVE_DIR_NAME)
CHECKPOINT_FILENAME_PREFIX = 'facial_keypoints_mobilenet_'
CHECKPOINT_EPOCH_INTERVAL = 10 # Determins the Number of Epochs to wait before saving a checkpoint

# --- Augmentation Parameters ---
AUG_HORIZONTAL_FLIP_PROB = 0.5 # Probability of Flipping the imaegeS
AUG_SHIFT_LIMIT = 0.06  # Example., 0.06 for +/- 6% shift
AUG_SCALE_LIMIT = 0.1   # Example., 0.1 for scales between 0.9 and 1.1
AUG_ROTATE_LIMIT = 20   # Example., 20 for +/- 20 degrees rotation
AUG_SSR_PROB = 0.7      # ShiftScaleRotate probability
AUG_BRIGHTNESS_CONTRAST_PROB = 0.5 # Probability of Changing the Brightness and Contrast 50% 
AUG_GAUSS_NOISE_PROB = 0.3 # Probability of Adding Gaussian Noise
BRIGHTNESS_LIMIT = 0.2  # Limits for Brightness Adjustment
CONTRAST_LIMIT = 0.2 #  Limits for Contrast Adjustment
# Gaussian Noise Limits:
GAUSS_VAR_LIMIT_MIN = 10.0
GAUSS_VAR_LIMIT_MAX = 50.0

# --- Other ---
NUM_WORKERS_DATALOADER = os.cpu_count() // 2 if os.cpu_count() else 2