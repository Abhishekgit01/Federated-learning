"""
Evaluation module for the federated model.

Provides functions to test model accuracy and generate
confusion matrices from the CIFAR-10 test set.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate the model on the test dataset.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to use for evaluation ('cpu' or 'cuda').

    Returns:
        tuple:
            - overall_accuracy (float): Overall test accuracy (0.0 to 1.0).
            - class_accuracies (dict): Mapping of class name to accuracy.
    """
    model.eval()
    correct = 0
    total = 0

    class_correct = {c: 0 for c in range(10)}
    class_total = {c: 0 for c in range(10)}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Per-class accuracy
            for t, p in zip(target.view(-1), predicted.view(-1)):
                class_total[t.item()] += 1
                if t.item() == p.item():
                    class_correct[t.item()] += 1

    overall_accuracy = correct / total if total > 0 else 0.0

    class_accuracies = {}
    for i in range(10):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        class_accuracies[CIFAR10_CLASSES[i]] = acc

    logger.info(
        "Test Eval - Overall Acc: %.2f%%", overall_accuracy * 100
    )

    return overall_accuracy, class_accuracies


def plot_confusion_matrix(model, test_loader, results_dir="results", device="cpu"):
    """
    Generate and save a confusion matrix plot for the model.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        results_dir (str): Directory to save the plot.
        device (str): Device to use ('cpu' or 'cuda').
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CIFAR10_CLASSES, yticklabels=CIFAR10_CLASSES,
    )
    plt.title("CIFAR-10 Confusion Matrix (Federated Model)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save
    plot_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    logger.info("Confusion matrix saved to %s", plot_path)
