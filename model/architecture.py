import torch
import torch.nn as nn
from torchvision import models
from config import DEVICE, PRETRAINED_WEIGHTS, NUM_KEYPOINTS


def build_facial_keypoints_model(pretrained: bool = True):
    """
    Builds and returns the MobileNetV3 Small model configured for facial keypoint detection.
    """
    actual_weights = None
    if pretrained:
        if PRETRAINED_WEIGHTS == "MobileNet_V3_Small_Weights.IMAGENET1K_V1":
            actual_weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            # TODO: Handle unknown weight string or raise an error
            print(
                f"Warning: Unknown PRETRAINED_WEIGHTS string in config: '{PRETRAINED_WEIGHTS}'. Using no weights."
            )

    model = models.mobilenet_v3_small(weights=actual_weights)

    # Allow fine-tuning of all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Replacing the classifier:
    num_input_features_to_classifier = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_input_features_to_classifier, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_KEYPOINTS * 2),  # Output 136 values (68 x,y pairs)
        nn.Sigmoid(),  # Squashes output values between 0 and 1 for normalized keypoints
    )
    return model.to(DEVICE)


def get_loss_function():
    """Returns the loss function for the task."""
    return nn.L1Loss()  # Mean Absolute Error (MAE)


def get_optimizer(model: nn.Module, learning_rate: float):
    """Returns the optimizer for the model."""
    # only pass the model parameters that needs to be trained:
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate
    )
