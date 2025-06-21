import albumentations as A

from config import (
    AUG_HORIZONTAL_FLIP_PROB,
    AUG_SHIFT_LIMIT,
    AUG_SCALE_LIMIT,
    AUG_ROTATE_LIMIT,
    AUG_SSR_PROB,
    AUG_BRIGHTNESS_CONTRAST_PROB,
    AUG_GAUSS_NOISE_PROB,
    BRIGHTNESS_LIMIT,
    CONTRAST_LIMIT,
    GAUSS_VAR_LIMIT_MIN,
    GAUSS_VAR_LIMIT_MAX,
)


def get_training_augmentations():
    """Returns the Albumentations compose Augmentation object for training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=AUG_HORIZONTAL_FLIP_PROB),
            A.Affine(
                scale=(
                    1.0 - AUG_SCALE_LIMIT,
                    1.0 + AUG_SCALE_LIMIT,
                ),  # Example: if limit is 0.1, range is (0.9, 1.1)
                translate_percent={
                    "x": (-AUG_SHIFT_LIMIT, AUG_SHIFT_LIMIT),
                    "y": (-AUG_SHIFT_LIMIT, AUG_SHIFT_LIMIT),
                },  # Translate independently on x and y
                rotate=(
                    -AUG_ROTATE_LIMIT,
                    AUG_ROTATE_LIMIT,
                ),  # Rotation range in degrees
                p=AUG_SSR_PROB,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=BRIGHTNESS_LIMIT,
                contrast_limit=CONTRAST_LIMIT,
                p=AUG_BRIGHTNESS_CONTRAST_PROB,
            ),
            A.GaussNoise(
                p=AUG_GAUSS_NOISE_PROB,
                var_limit=(GAUSS_VAR_LIMIT_MIN, GAUSS_VAR_LIMIT_MAX),
            ),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
