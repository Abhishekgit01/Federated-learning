"""
Federated Averaging (FedAvg) server module.

Implements the central aggregation server for the Federated Learning
simulation. Collects model updates from participating clients and
computes a weighted average to produce the new global model.

Reference:
    McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data" (2017). https://arxiv.org/abs/1602.05629
"""

import copy
import logging
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


class FedAvgServer:
    """
    Central server implementing the Federated Averaging algorithm.

    Maintains the global model and aggregates client updates using
    weighted averaging based on each client's sample count.

    Args:
        global_model (nn.Module): The global model to be trained.
        device (str): Device for model operations ('cpu' or 'cuda').
    """

    def __init__(self, global_model, device="cpu"):
        """Initialize the FedAvg server with the global model."""
        self.global_model = global_model
        self.device = device
        self.round_number = 0

    def get_global_weights(self):
        """
        Get a deep copy of the current global model's state dict.

        Returns:
            dict: Deep copy of the global model state_dict.
        """
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, client_updates, client_sample_counts):
        """
        Perform Federated Averaging aggregation.

        Computes the new global model as a weighted average of client
        model updates: w_global = sum(n_k / n_total * w_k)

        where n_k is client k's number of samples and n_total is the
        total number of samples across all participating clients.

        Args:
            client_updates (list[dict]): List of client state_dicts.
            client_sample_counts (list[int]): Number of samples per client
                (must correspond to entries in client_updates).

        Returns:
            dict: The new aggregated global model state_dict.
        """
        if not client_updates:
            logger.warning("No client updates received, skipping aggregation.")
            return self.get_global_weights()

        self.round_number += 1

        # Total samples across all participating clients
        total_samples = sum(client_sample_counts)

        # Compute weighted sum of parameters
        aggregated_state = OrderedDict()
        for key in client_updates[0].keys():
            aggregated_state[key] = torch.zeros_like(
                client_updates[0][key], dtype=torch.float32
            )
            for i, client_state in enumerate(client_updates):
                weight = client_sample_counts[i] / total_samples
                aggregated_state[key] += weight * client_state[key].float()

        # Update global model
        self.global_model.load_state_dict(aggregated_state)

        logger.info(
            "Round %d: Aggregated %d client updates (%d total samples)",
            self.round_number, len(client_updates), total_samples,
        )

        return aggregated_state

    @staticmethod
    def compute_comm_cost(model, num_participating_clients):
        """
        Compute the communication cost for one round of federated learning.

        Each participating client downloads the global model (1x) and
        uploads their updated model (1x), so cost = 2 * num_clients * model_size.

        Args:
            model (nn.Module): The model being trained.
            num_participating_clients (int): Number of clients participating
                in this round (excludes dropped-out clients).

        Returns:
            float: Communication cost in megabytes (MB).
        """
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # float32

        # Each client: 1 download + 1 upload = 2x model size
        round_cost_mb = 2 * num_participating_clients * model_size_mb

        logger.debug(
            "Comm cost: %d clients × 2 × %.3f MB = %.3f MB",
            num_participating_clients, model_size_mb, round_cost_mb,
        )

        return round_cost_mb
