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


def partition_non_iid(dataset, num_clients, batch_size=64, alpha=0.5):
    """
    Split the dataset into non-IID shards heavily using a Dirichlet distribution.
    This simulates real-world heterogeneity better than just slicing the dataset.

    Args:
        dataset: The full training dataset.
        num_clients (int): Number of federated clients.
        batch_size (int): Batch size for each client's DataLoader.
        alpha (float): Parameter for Dirichlet distribution (lower = more heterogeneous).

    Returns:
        dict[int, DataLoader]: Mapping from client_id to DataLoader.
        dict[int, int]: Mapping from client_id to number of samples.
    """
    num_classes = 10
    targets = np.array(dataset.targets)

    # Setup client assignments
    client_indices = {i: [] for i in range(num_clients)}
    
    # We iterate over each class and distribute its samples across clients 
    # based on the Dirichlet distribution proportions.
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        
        # Dirichlet gives a distribution of probabilities across clients for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Ensure we don't accidentally create zero samples by normalizing properly 
        proportions = np.array([p * (len(idx_j) < len(dataset) / num_clients) for p, idx_j in zip(proportions, client_indices.values())])
        proportions = proportions / proportions.sum()
        
        # Split indices
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, proportions)
        
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())

    # Build Loaders
    client_loaders = {}
    client_sample_counts = {}

    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
        subset = Subset(dataset, client_indices[client_id])
        client_loaders[client_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
        client_sample_counts[client_id] = len(client_indices[client_id])

        # Log per-client class distribution roughly
        label_counts = {}
        for idx in client_indices[client_id]:
            label = int(targets[idx])
            label_counts[label] = label_counts.get(label, 0) + 1

        logger.info(
            "Non-IID Client %d: %d samples, distribution=%s",
            client_id, len(client_indices[client_id]),
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
