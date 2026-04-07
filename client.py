"""
Federated Learning client module.

Implements the local training logic for each edge client in the
federated learning simulation. Each client trains a local copy of
the global model on its private data shard.
"""

import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Represents an edge device in the Federated Learning system.

    Each client holds a local data shard and trains the global model
    locally for a specified number of epochs before returning updated
    weights to the central server.

    Args:
        client_id (int): Unique identifier for this client.
        data_loader (DataLoader): Local training data loader.
        num_samples (int): Number of training samples this client has.
        device (str): Device for training ('cpu' or 'cuda').
        lr (float): Learning rate for local SGD optimizer.
    """

    def __init__(self, client_id, data_loader, num_samples, device="cpu", lr=0.01):
        """Initialize the federated client with its local data."""
        self.client_id = client_id
        self.data_loader = data_loader
        self.num_samples = num_samples
        self.device = device
        self.lr = lr
        self.is_active = True  # Can be set to False for dropout simulation

    def train(self, global_model_state, local_epochs):
        """
        Train a local copy of the global model on this client's data.

        Receives the global model weights, creates a local copy,
        trains for the specified number of local epochs, and returns
        the updated model weights along with training metrics.

        Args:
            global_model_state (dict): State dict of the current global model.
            local_epochs (int): Number of local training epochs.

        Returns:
            dict: Updated model state_dict after local training.
            float: Average training loss across all local epochs.
        """
        # Import here to avoid circular dependency
        from models.cnn import create_model

        # Create a local model copy and load global weights
        local_model = create_model(device=self.device)
        local_model.load_state_dict(copy.deepcopy(global_model_state))
        local_model.train()

        # Setup optimizer and loss
        optimizer = optim.SGD(
            local_model.parameters(), lr=self.lr, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            total_loss += avg_epoch_loss
            total_batches += 1

            logger.debug(
                "Client %d - Epoch %d/%d - Loss: %.4f",
                self.client_id, epoch + 1, local_epochs, avg_epoch_loss,
            )

        avg_loss = total_loss / max(total_batches, 1)
        logger.info(
            "Client %d finished training: avg_loss=%.4f (%d samples, %d epochs)",
            self.client_id, avg_loss, self.num_samples, local_epochs,
        )

        return local_model.state_dict(), avg_loss

    def __repr__(self):
        """String representation of the client."""
        status = "active" if self.is_active else "dropped"
        return (
            f"FederatedClient(id={self.client_id}, "
            f"samples={self.num_samples}, status={status})"
        )
