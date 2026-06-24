# Raspberry Pi Federated Client Deployment Guide

## 1. Overview

This guide covers the deployment and operation of the federated learning clients (`client1.py` and `client2.py`) on separate Raspberry Pi devices. Each Raspberry Pi hosts one client node and connects to a central Flower server for federated training.

## 2. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Raspberry Pi Model | 4B (4 GB) | 5 (8 GB) |
| Storage | 32 GB microSD | 64 GB microSD (A2 rated) |
| Power Supply | Official 3A USB-C | Official 3A USB-C |
| Cooling | Heatsink | Active cooler/fan |
| Network | Wi-Fi | Ethernet (Cat5e+) |
| CPU | Quad-core Cortex-A72 | Similar/better |
| RAM | 4 GB LPDDR4 | More preferred |

**Multi-client deployment:**
- 2 x Raspberry Pi units (one per client)
- 1 x Host machine (server)

## 3. Network Topology

```
Client 1 (Raspberry Pi A)
       \
        Server (Host Machine / Cloud)
       /
Client 2 (Raspberry Pi B)
```

Configuration:
- All nodes must be on the same network.
- Configure the server IP in `client*.py`.
- Default gateway and DNS should be reachable for dependency downloads.
- Port **8080** TCP must be allowed through the host and client firewalls.

## 4. Preparing the Host (Control Machine)

The host clones the repository, runs the Flower server, and later consumes the results. Steps:

```bash
git clone https://github.com/your-org/fed-project.git
cd fed-project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
pip install -r requirements.txt  # sleep/wake tools if needed
```

Write down the host machine IP (e.g., `192.168.1.100`) for client configuration.

## 5. Preparing Each Raspberry Pi

### 5.1 Install Raspberry Pi OS

- Raspberry Pi Imager:
  - OS: Raspberry Pi OS Lite (64-bit)
  - Enable SSH
  - Set hostname and credentials
  - Wireless LAN credentials (if applicable)

Boot the Pi and update:
```bash
sudo apt update && sudo apt upgrade -y
```

### 5.2 Install Python Dependencies

```bash
sudo apt install -y python3 python3-pip python3-venv
```

Create and activate a virtual environment:
```bash
python3 -m venv federated_client
source federated_client/bin/activate
```

Install ML dependencies:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn numpy pandas Pillow flwr==1.7.0
```

> Note: CPU-only PyTorch is recommended unless Pi has a USB accelerator.

## 6. Repository Sync

Copy project files to each Pi:

```bash
# From Mac/Linux host
scp -r fed_project pi@raspberrypi-a:/home/pi/fed_project_client1
scp -r fed_project pi@raspberrypi-b:/home/pi/fed_project_client2
```

Or use `git`:
```bash
git clone https://github.com/your-org/fed-project.git /home/pi/fed_project
```

Place only the required dataset in each Pi:

```
/home/pi/fed_project/
  ├── cnn_model.py
  ├── svm_model.py
  ├── evaluation.py
  ├── requirements.txt  # optional
  ├── client1.py        # Pi A
  ├── client1/          # Pi A dataset
  └── client2.py        # Pi B
      client2/          # Pi B dataset
```

> **Dataset considerations:**
> - Store local data directory under `/home/pi/fed_project/client*`.
> - For storage-limited Pis, use SD cards with high endurance ratings.

## 7. Environment Configuration

### 7.1 Edit `client1.py` (Raspberry Pi A)

Update the server address and dataset path:

```python
# Original: server_address = "10.143.202.174:8080"
server_address = "192.168.1.100:8080"

DATASET_PATH = "/home/pi/fed_project/client1"
```

### 7.2 Edit `client2.py` (Raspberry Pi B)

```python
server_address = "192.168.1.100:8080"
DATASET_PATH = "/home/pi/fed_project/client2"
```

### 7.3 Verify Connectivity

```bash
ping 192.168.1.100
curl http://192.168.1.100:8080/healthz   # adapt to backend routes
```

Enable TCP keepalive on Pis to avoid silent drops:
```bash
sudo sysctl -w net.ipv4.tcp_keepalive_time=60
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=10
sudo sysctl -w net.ipv4.tcp_keepalive_probes=6
```

## 8. Running the Flower Server (Host)

Install dependencies on the host:
```bash
pip install flwr==1.7.0
pip install -r backend/requirements.txt
```

Run the Flower server:
```bash
python server.py
```

Server logs indicate when clients connect.

## 9. Starting Federated Clients

On each Raspberry Pi:
```bash
cd /home/pi/fed_project
source federated_client/bin/activate
python3 client1.py
```

```bash
cd /home/pi/fed_project
source federated_client/bin/activate
python3 client2.py
```

Both clients should appear connected in the server console output.

## 10. Monitoring and Metrics

### 10.1 Server-Side Check

After each round, clients upload metrics. Backend dashboard exposes:
- `GET /metrics` -> combined charts
- `GET /metrics/clients` -> per-client metrics

### 10.2 Client-Side Metrics

Clients save locally:
- `/home/pi/fed_project/client1_metrics.csv`
- `/home/pi/fed_project/client2_metrics.csv`

Periodically retrieve metrics:
```bash
scp pi@raspberrypi-a:/home/pi/fed_project/client1_metrics.csv ./backup_client1_round_$(date +%F).csv
```

### 10.3 Log Aggregation

Keep a lightweight journal:
```bash
journalctl -u fl-client-1 --no-pager -n 100
```

## 11. Updating the Server IP Over SSH

If the server IP changes, avoid hardcoding. Create a small ssh wrapper:

```bash
# On Pi A
export SERVER_IP=192.168.1.100
sed -i "s/server_address=.*/server_address=\"${SERVER_IP}:8080\"/" client1.py
```

## 12. Performance Tuning

- Reduce `BATCH_SIZE` to 8 if memory is constrained.
- Disable logging on training loops if disk space is scarce.
- Use `taskset` to limit CPU affinity if the Pi shares resources.
- Pre-warm PyTorch with a dummy forward pass.

```bash
python3 - <<'PY'
import torch, numpy as np
x = torch.randn(1, 3, 64, 64)
m = torch.nn.Conv2d(3, 16, 3)
with torch.no_grad(): m(x)
print("Warmup complete")
PY
```

## 13. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Connection refused | Wrong server IP/port | Verify `server_address` |
| OOM killed | Large model/batch | Reduce `BATCH_SIZE` |
| Slow gradients | CPU torch, no CUDA | Use lightweight feat. dim or quantize |
| Client silence | Firewall blocks 8080 | `sudo ufw allow 8080` |
| Dataset not found | Incorrect `DATASET_PATH` | Verify directory structure |
| Diverging metrics | Data skew | Add FedAdam / server-side LR schedule |

## 14. Post-Training Inference

When emitting predictions from the central system, ensure:
- Model weights are downloaded to a shared location.
- The Pi can reach `localhost:8000` if running local inference.
- Use the backend `/predict` endpoint for off-device inference.