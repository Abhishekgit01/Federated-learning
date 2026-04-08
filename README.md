# Federated Learning Edge AI Simulation

Hi! I'm an undergraduate student passionate about the intersection of Machine Learning and Edge Computing. This is a PyTorch implementation of a Federated Learning system that I built to explore how we can train models on resource-constrained edge devices without compromising user data privacy. 

This project was developed as part of my application for the **IISc Bengaluru ACM India Summer School 2026**.

## Why did I build this?

I've always been fascinated by how AI can be deployed in the real world. Growing up and traveling around smaller towns in Karnataka, I noticed how spotty internet connectivity can be. We can't always rely on sending huge amounts of raw data from mobile devices to a giant central server in Bengaluru or Mumbai—it burns bandwidth, kills battery life, and honestly, creates massive privacy risks.

I started reading about Federated Learning [^1] and it completely flipped things around for me. Instead of sending the data to the model, what if we send the model to the data? 

To make this simulation realistic for edge environments, I tried to implement constraints you'd actually see in the wild:
- **Model Size:** The `EdgeCNN` model I built has under 300K parameters so it fits into tight memory limits.
- **Data Heterogeneity:** Real-world edge devices don't have perfectly balanced, IID datasets (my phone sees mostly dogs, yours might see mostly cats). I used a Dirichlet distribution to force realistic Non-IID partitions.
- **Network Drops:** Edge devices lose connection all the time in rural areas. The simulation randomly drops a percentage of clients during each round to test if the central server can still recover and learn.

## How it works (The FedAvg Loop)

The core algorithm is Federated Averaging (FedAvg). It's incredibly elegant:
1. The central server broadcasts the current global model weights to the participating clients.
2. Each client trains the model on its *local dataset* for just a few epochs.
3. The clients send their updated weights back. The raw data *never* leaves the device!
4. The server averages these weights—adjusting based on how many samples each client trained on—to create the next global model.

## Running the Code

I recommend creating a virtual environment so nothing conflicts:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run the training loop with my default settings (5 clients, 10 rounds, 5 local epochs, 20% dropout rate, non-IID data):
```bash
python train.py
```

You can tweak the parameters using arguments. If you want to run an IID benchmark with 10 clients:
```bash
python train.py --num_clients 10 --rounds 5 --iid
```

### Outputs
The script trains both the federated model and a centralized baseline model for isolated comparison. The results are dumped into the `results/` folder, including accuracy curves, communication cost plots, and a final confusion matrix.

## Reflections

### What I struggled with
Getting the math right for the Federated Averaging was surprisingly tricky. At first, I was just taking a simple mean of the weights, which failed miserably when clients had different amounts of data! I also spent way too long tuning the Dirichlet parameter (`alpha`) to get a non-IID split that was hard enough to be realistic, but not impossible to learn from.

### What surprised me
Honestly, I didn't expect non-IID data to hurt accuracy this much compared to a centralized baseline. When a device only sees 2 or 3 classes, it overfits locally so fast. It was super cool to watch the global aggregation slowly correct those local biases over multiple communication rounds.

---
[^1]: McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS).*
