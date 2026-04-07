"""
Plotting module to visualize Federated Learning simulation results.

Loads metrics from a JSON file and generates comparison plots for
accuracy, loss, communication costs, and convergence.
"""

import json
import logging
import os

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Apply a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_accuracy_curve(metrics, results_dir="results"):
    """Plot global model test accuracy over rounds."""
    rounds = [r["round"] for r in metrics["fed_rounds"]]
    acc = [r["test_accuracy"] * 100 for r in metrics["fed_rounds"]]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, acc, marker="o", linestyle="-", color="#2ca02c", linewidth=2)
    plt.title("Federated Global Model Accuracy per Round")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, "accuracy_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_client_loss(metrics, results_dir="results"):
    """Plot local training loss for each client over rounds."""
    plt.figure(figsize=(10, 6))

    num_clients = metrics["config"]["num_clients"]
    rounds = [r["round"] for r in metrics["fed_rounds"]]

    for client_id in range(num_clients):
        # Extract loss for this client across all rounds
        client_losses = []
        valid_rounds = []

        for r_data in metrics["fed_rounds"]:
            str_id = str(client_id)
            if str_id in r_data["client_losses"]:
                client_losses.append(r_data["client_losses"][str_id])
                valid_rounds.append(r_data["round"])

        if valid_rounds:
            plt.plot(
                valid_rounds, client_losses,
                marker=".", linestyle="-", alpha=0.7,
                label=f"Client {client_id}"
            )

    plt.title("Per-Client Local Training Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Average Local Loss")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    path = os.path.join(results_dir, "client_loss_curve.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_communication_cost(metrics, results_dir="results"):
    """Plot cumulative communication cost over rounds."""
    rounds = [r["round"] for r in metrics["fed_rounds"]]

    # Calculate cumulative cost
    cumulative_cost = []
    current_cap = 0
    for r in metrics["fed_rounds"]:
        current_cap += r["comm_cost_mb"]
        cumulative_cost.append(current_cap)

    plt.figure(figsize=(8, 6))
    plt.bar(rounds, [r["comm_cost_mb"] for r in metrics["fed_rounds"]],
            color="#1f77b4", alpha=0.7, label="Per Round")
    plt.plot(rounds, cumulative_cost, color="#ff7f0e", marker="o",
             linewidth=2, label="Cumulative")

    plt.title("Communication Cost per Round")
    plt.xlabel("Communication Round")
    plt.ylabel("Data Transferred (MB)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(results_dir, "comm_cost.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_convergence_comparison(metrics, results_dir="results"):
    """Plot federated vs centralized accuracy comparison."""
    if "centralized_history" not in metrics:
        logger.warning("No centralized history found for comparison plot.")
        return

    # Federated data (x-axis = equivalent epochs)
    fed_rounds = [r["round"] for r in metrics["fed_rounds"]]
    local_epochs = metrics["config"]["local_epochs"]
    # Equivalent centralized epochs = round * local_epochs
    fed_epochs = [r * local_epochs for r in fed_rounds]
    fed_acc = [r["test_accuracy"] * 100 for r in metrics["fed_rounds"]]

    # Centralized data
    cent_epochs = [h["epoch"] for h in metrics["centralized_history"]]
    cent_acc = [h["test_accuracy"] * 100 for h in metrics["centralized_history"]]

    plt.figure(figsize=(10, 6))
    plt.plot(fed_epochs, fed_acc, marker="o", linestyle="-",
             color="#2ca02c", linewidth=2.5, label="Federated (FedAvg)")
    plt.plot(cent_epochs, cent_acc, linestyle="--",
             color="#d62728", linewidth=2, alpha=0.8, label="Centralized Baseline")

    plt.title("Convergence Comparison: Federated vs Centralized")
    plt.xlabel("Equivalent Training Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.4)

    # Note about non-IID
    iid_status = "IID" if metrics["config"]["iid"] else "Non-IID"
    plt.annotate(f"Data Partition: {iid_status}",
                 xy=(0.02, 0.05), xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()

    path = os.path.join(results_dir, "convergence_comparison.png")
    plt.savefig(path, dpi=300)
    plt.close()


def generate_all_plots(metrics_file="results/metrics.json"):
    """Load metrics JSON and generate all available plots."""
    if not os.path.exists(metrics_file):
        logger.error("Metrics file %s not found!", metrics_file)
        return

    results_dir = os.path.dirname(metrics_file)

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    logger.info("Generating plots based on %s...", metrics_file)

    plot_accuracy_curve(metrics, results_dir)
    plot_client_loss(metrics, results_dir)
    plot_communication_cost(metrics, results_dir)
    plot_convergence_comparison(metrics, results_dir)

    logger.info("All plots saved to %s/", results_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_all_plots()
