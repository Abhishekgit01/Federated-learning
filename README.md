# Federated Learning Edge AI Simulation

A PyTorch implementation of a Federated Learning system. I built this simulation to explore how we can train models on resource-constrained edge devices without compromising user data privacy.

This project was developed as part of my application for the IISc Bengaluru ACM India Summer School 2026.

## Background

In typical machine learning setups, we send huge amounts of raw data from devices to a central server. This uses up a lot of bandwidth and creates obvious privacy risks.

Federated Learning flips this around. Instead of sending data to the server, we send the model to the data.

To make this realistic for edge environments, I implemented three constraints in this project:
1. **Model Size:** The EdgeCNN model has under 300K parameters so it fits into tight memory limits.
2. **Data Heterogeneity:** Real world edge devices don't have nicely balanced datasets. I used a custom "Non-IID" partitioner to force each client to train mostly on just 2 to 3 classes.
3. **Network Drops:** Edge devices lose connection all the time. The simulation randomly drops a percentage of clients during each round to test if the aggregator can still recover and learn.

## System Architecture

```text
                 [ Central Server ]
                 |  Runs FedAvg   |
                 | Global Weights |
                 +-------+--------+
                         |
            +------------+-----------+
            |                        |
      (Weights Up/Down)        (Weights Up/Down) 
            |                        |
        [Client 1]               [Client 2] 
      Local SGD on             Local SGD on
      local data               local data
      (e.g., Cats, Dogs)       (e.g., Cars, Planes)
```

## How Federated Averaging Works

The Federated Averaging (FedAvg) loop works like this:
1. The server broadcasts the current global model weights to the clients.
2. Each client trains the model on its local dataset for a few epochs.
3. The clients send their updated weights back. The raw data never leaves the device.
4. The server averages these weights (adjusting based on how many samples each client trained on) to create the next global model.

## Running the Code

### Setup

I recommend creating a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Starting the simulation

To run the training loop with default settings (5 clients, 10 rounds, 5 local epochs, 20% dropout rate, non-IID data):

```bash
python train.py
```

You can tweak the parameters using arguments. For example, to run an IID benchmark with 10 clients:
```bash
python train.py --num_clients 10 --rounds 5 --iid
```

### Outputs

The script trains both the federated model and a centralized baseline model for isolated comparison. The results are dumped into the `results/` folder:
* `metrics.json`: JSON dump of all the underlying calculations.
* `accuracy_curve.png`: Test accuracy over communication rounds.
* `client_loss_curve.png`: How local loss drops per client.
* `comm_cost.png`: The exact MB size of weights transferred over the network.
* `convergence_comparison.png`: Federated accuracy vs Centralized accuracy.
* `confusion_matrix.png`: Final evaluation heatmap.

## Results Summary

* **Privacy:** Federated Learning keeps all training data on the edge device. The centralized baseline requires all data to be aggregated.
* **Accuracy:** The centralized model provides a theoretical upper bound. The federated model converges slightly slower due to the Non-IID data split, but still reaches comparable accuracy given enough rounds.
* **Bandwidth:** The communication cost is fixed to the model size (around 2.3 MB per client round) rather than scaling with the dataset size.
