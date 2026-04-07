"""
Data partitioning module for Federated Learning simulation.

Provides IID and non-IID data splitting strategies to simulate
heterogeneous edge device data distributions using CIFAR-10.
"""

import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# CIFAR-10 class names for reference
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_cifar10_transforms():
    """
    Return standard CIFAR-10 data transforms for training and testing.

    Returns:
        tuple: (train_transform, test_transform)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    return train_transform, test_transform


def load_cifar10(data_dir="./cifar10_data"):
    """
    Download and load the CIFAR-10 dataset.

    Args:
        data_dir (str): Directory to store/load the dataset.

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_transform, test_transform = get_cifar10_transforms()

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    logger.info(
        "CIFAR-10 loaded: %d training samples, %d test samples",
        len(train_dataset), len(test_dataset),
    )
    return train_dataset, test_dataset


def partition_iid(dataset, num_clients, batch_size=64):
    """
    Split the dataset into IID (Independent and Identically Distributed)
    shards across clients. Each client gets a uniformly random subset.

    Args:
        dataset: The full training dataset.
        num_clients (int): Number of federated clients.
        batch_size (int): Batch size for each client's DataLoader.

    Returns:
        dict[int, DataLoader]: Mapping from client_id to DataLoader.
        dict[int, int]: Mapping from client_id to number of samples.
    """
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)
    shard_size = num_samples // num_clients

    client_loaders = {}
    client_sample_counts = {}

    for client_id in range(num_clients):
        start = client_id * shard_size
        # Last client gets remaining samples to handle uneven division
        if client_id == num_clients - 1:
            end = num_samples
        else:
            end = start + shard_size

        client_indices = indices[start:end].tolist()
        subset = Subset(dataset, client_indices)
        client_loaders[client_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
        client_sample_counts[client_id] = len(client_indices)

        logger.info(
            "IID Client %d: %d samples", client_id, len(client_indices)
        )

    return client_loaders, client_sample_counts


def partition_non_iid(dataset, num_clients, batch_size=64, dominant_ratio=0.7):
    """
    Split the dataset into non-IID shards where each client predominantly
    receives data from 2-3 classes, simulating real-world heterogeneity.

    Each client gets ~70% of their data from 2 dominant classes and ~30%
    from the remaining classes to prevent complete label absence.

    Args:
        dataset: The full training dataset.
        num_clients (int): Number of federated clients.
        batch_size (int): Batch size for each client's DataLoader.
        dominant_ratio (float): Fraction of data from dominant classes.

    Returns:
        dict[int, DataLoader]: Mapping from client_id to DataLoader.
        dict[int, int]: Mapping from client_id to number of samples.
    """
    num_classes = 10
    targets = np.array(dataset.targets)

    # Group indices by class label
    class_indices = {
        c: np.where(targets == c)[0].tolist() for c in range(num_classes)
    }

    # Shuffle indices within each class
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    # Assign 2 dominant classes per client (cycling through all 10 classes)
    dominant_classes = {}
    for client_id in range(num_clients):
        c1 = (client_id * 2) % num_classes
        c2 = (client_id * 2 + 1) % num_classes
        dominant_classes[client_id] = [c1, c2]

    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients

    # Track consumption pointers per class
    class_pointers = {c: 0 for c in range(num_classes)}

    client_loaders = {}
    client_sample_counts = {}

    for client_id in range(num_clients):
        client_indices = []
        dominant = dominant_classes[client_id]
        num_dominant = int(samples_per_client * dominant_ratio)
        num_other = samples_per_client - num_dominant

        # Collect dominant class samples (split evenly between 2 classes)
        per_dominant_class = num_dominant // len(dominant)
        for cls in dominant:
            ptr = class_pointers[cls]
            available = len(class_indices[cls]) - ptr
            take = min(per_dominant_class, available)
            client_indices.extend(class_indices[cls][ptr:ptr + take])
            class_pointers[cls] += take

        # Collect other class samples (spread across remaining classes)
        other_classes = [c for c in range(num_classes) if c not in dominant]
        per_other_class = max(1, num_other // len(other_classes))
        for cls in other_classes:
            ptr = class_pointers[cls]
            available = len(class_indices[cls]) - ptr
            take = min(per_other_class, available)
            client_indices.extend(class_indices[cls][ptr:ptr + take])
            class_pointers[cls] += take

        np.random.shuffle(client_indices)
        subset = Subset(dataset, client_indices)
        client_loaders[client_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
        client_sample_counts[client_id] = len(client_indices)

        # Log per-client class distribution
        label_counts = {}
        for idx in client_indices:
            label = int(targets[idx])
            label_counts[label] = label_counts.get(label, 0) + 1

        logger.info(
            "Non-IID Client %d: %d samples, dominant classes=%s, distribution=%s",
            client_id, len(client_indices), dominant,
            {CIFAR10_CLASSES[k]: v for k, v in sorted(label_counts.items())},
        )

    return client_loaders, client_sample_counts


def get_test_loader(test_dataset, batch_size=128):
    """
    Create a DataLoader for the test dataset.

    Args:
        test_dataset: The CIFAR-10 test dataset.
        batch_size (int): Batch size for the test DataLoader.

    Returns:
        DataLoader: Test data loader.
    """
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
