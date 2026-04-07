"""
Lightweight CNN model for Federated Learning on CIFAR-10.

Designed to simulate edge device constraints with fewer than 500K
total parameters, making it suitable for resource-constrained
deployment scenarios in edge AI.
"""

import logging

import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EdgeCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        - Conv2d(3, 32, 3) -> ReLU -> MaxPool2d(2)
        - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
        - Flatten
        - Linear(64 * 6 * 6, 256) -> ReLU -> Dropout(0.25)
        - Linear(256, 10)

    Input: 3x32x32 CIFAR-10 images
    Output: 10-class logits

    Spatial dimension trace:
        32x32 -> Conv(3,pad=0) -> 30x30 -> MaxPool(2) -> 15x15
        -> Conv(3,pad=0) -> 13x13 -> MaxPool(2) -> 6x6
    """

    def __init__(self):
        """Initialize the EdgeCNN model layers."""
        super(EdgeCNN, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=0
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected classifier
        # After 2x (conv + pool): 64 channels x 6 x 6 spatial = 2304
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

        # Regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))   # -> (B, 32, 15, 15)
        x = self.pool(F.relu(self.conv2(x)))   # -> (B, 64, 6, 6)
        x = x.view(x.size(0), -1)              # -> (B, 2304)
        x = self.dropout(F.relu(self.fc1(x)))   # -> (B, 128)
        x = self.fc2(x)                         # -> (B, 10)
        return x


def get_model_size(model):
    """
    Compute the total number of trainable parameters and model size in MB.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        tuple: (total_params, size_mb)
            - total_params (int): Total number of trainable parameters.
            - size_mb (float): Model size in megabytes (float32).
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes

    logger.info("Model parameters: %d (%.3f MB)", total_params, size_mb)
    return total_params, size_mb


def create_model(device="cpu"):
    """
    Factory function to create and initialize an EdgeCNN model.

    Args:
        device (str): Device to place the model on ('cpu' or 'cuda').

    Returns:
        EdgeCNN: The initialized model on the specified device.
    """
    model = EdgeCNN().to(device)
    total_params, size_mb = get_model_size(model)

    if total_params > 500_000:
        logger.warning(
            "Model has %d params, exceeding the 500K edge target!", total_params
        )
    else:
        logger.info(
            "Model within edge constraints: %d params (< 500K)", total_params
        )

    return model
