"""
Main orchestrator for the Federated Learning simulation.

Responsible for setting up clients and data, running the FedAvg
training loop, simulating client dropouts, computing a centralized
baseline, and outputting metrics for plotting.
"""

import argparse
import copy
import json
import logging
import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from client import FederatedClient
from data.partition import get_test_loader, load_cifar10, partition_iid, partition_non_iid
from evaluate import evaluate_model, plot_confusion_matrix
from models.cnn import create_model
from plot_results import generate_all_plots
from server import FedAvgServer

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def debug_print(msg):
    # just a quick helper I used during early debugging
    print(f"[DEBUG] {msg}")


def run_centralized_baseline(dataset, test_loader, config, device, file_path="results/metrics.json"):
    """
    Train a standard centralized model to serve as a baseline.
    Trains for exactly (rounds * local_epochs) total epochs for a fair comparison.
    """
    logger.info("-" * 50)
    logger.info("STARTING CENTRALIZED BASELINE TRAINING")
    logger.info("-" * 50)

    # Use a combined Dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    model = create_model(device=device)
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_epochs = config.rounds * config.local_epochs
    history = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate periodically or at the end
        if epoch % config.local_epochs == 0 or epoch == total_epochs:
            acc, _ = evaluate_model(model, test_loader, device=device)
            logger.info("Centralized Epoch %d/%d - Acc: %.2f%%", epoch, total_epochs, acc * 100)
            history.append({"epoch": epoch, "test_accuracy": acc})

    return history


def main():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation for Edge AI")
    parser.add_argument("--num_clients", type=int, default=5, help="Number of edge clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of communication rounds")
    parser.add_argument("--local_epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=64, help="Client batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Client learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Probability a client drops out per round")
    parser.add_argument("--iid", action="store_true", help="Use IID data split instead of non-IID")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save plots and metrics")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device.upper())

    # 1. Load Data
    train_dataset, test_dataset = load_cifar10(data_dir="./cifar10_data")
    test_loader = get_test_loader(test_dataset)

    # 2. Partition Data
    if args.iid:
        logger.info("Performing IID data split...")
        client_loaders, client_samples = partition_iid(train_dataset, args.num_clients, args.batch_size)
    else:
        logger.info("Performing Non-IID data split...")
        client_loaders, client_samples = partition_non_iid(train_dataset, args.num_clients, args.batch_size)

    # 3. Initialize Model and Server
    global_model = create_model(device=device)
    server = FedAvgServer(global_model, device=device)

    # 4. Initialize Clients
    clients = []
    for i in range(args.num_clients):
        c = FederatedClient(
            client_id=i,
            data_loader=client_loaders[i],
            num_samples=client_samples[i],
            device=device,
            lr=args.lr
        )
        clients.append(c)

    metrics = {
        "config": vars(args),
        "timestamp": datetime.now().isoformat(),
        "total_params": sum(p.numel() for p in global_model.parameters()),
        "fed_rounds": []
    }

    logger.info("-" * 50)
    logger.info("STARTING FEDERATED TRAINING")
    logger.info("Clients: %d | Rounds: %d | Local Epochs: %d", args.num_clients, args.rounds, args.local_epochs)
    logger.info("-" * 50)

    # 5. Federated Training Loop
    for r in range(1, args.rounds + 1):
        logger.info("==== Communication Round %d/%d ====", r, args.rounds)

        global_state = server.get_global_weights()
        client_updates = []
        participating_samples = []
        round_losses = {}
        
        # debug_print(f"Starting round {r} with global weights pulled")

        # Simulate dropout (randomly select clients that participate this round)
        participating_clients = []
        for client in clients:
            if random.random() >= args.dropout_rate:
                participating_clients.append(client)
            else:
                logger.warning("Client %d DROPPED OUT this round (simulated network failure).", client.client_id)

        # Clients perform local training
        for client in participating_clients:
            local_state, avg_loss = client.train(global_state, args.local_epochs)

            client_updates.append(local_state)
            participating_samples.append(client.num_samples)
            round_losses[client.client_id] = avg_loss

        # Collect updates and aggregate on server
        _ = server.aggregate(client_updates, participating_samples)

        # Calculate comm cost
        comm_cost = server.compute_comm_cost(global_model, len(participating_clients))

        # Evaluate new global model
        acc, _ = evaluate_model(global_model, test_loader, device=device)

        round_metrics = {
            "round": r,
            "participating_clients": [c.client_id for c in participating_clients],
            "client_losses": round_losses,
            "comm_cost_mb": comm_cost,
            "test_accuracy": acc
        }
        metrics["fed_rounds"].append(round_metrics)

    # 6. Final Evaluation (Confusion Matrix)
    logger.info("Generating confusion matrix for final federated model...")
    plot_confusion_matrix(global_model, test_loader, args.results_dir, device)

    # 7. Run Centralized Baseline
    baseline_history = run_centralized_baseline(train_dataset, test_loader, args, device)
    metrics["centralized_history"] = baseline_history

    # 8. Save Metrics
    metrics_path = os.path.join(args.results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved all metrics to %s", metrics_path)

    # 9. Generate Plots
    generate_all_plots(metrics_path)


if __name__ == "__main__":
    main()
