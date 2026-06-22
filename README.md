# Federated Learning Based Tomato Leaf Disease Detection using CNN-SVM Ensemble

A privacy-preserving distributed deep learning system for detecting diseases in tomato leaves using Federated Learning, Convolutional Neural Networks (CNN), and Support Vector Machines (SVM).

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technologies Used](#technologies-used)
4. [Dataset Structure](#dataset-structure)
5. [Component Breakdown](#component-breakdown)
   - [CNN Model (cnn_model.py)](#cnn-model)
   - [SVM Model (svm_model.py)](#svm-model)
   - [Evaluation Module (evaluation.py)](#evaluation-module)
   - [Flower Client (client1.py / client2.py)](#flower-client)
   - [Flower Server (server.py)](#flower-server)
6. [Federated Learning Workflow](#federated-learning-workflow)
7. [Client Training Pipeline (Detailed)](#client-training-pipeline)
8. [Ensemble Prediction Mechanism](#ensemble-prediction-mechanism)
9. [Installation & Setup](#installation--setup)
10. [Running the Project](#running-the-project)
11. [Evaluation Metrics](#evaluation-metrics)
12. [Results](#results)
13. [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a **Federated Learning** framework for **Tomato Leaf Disease Detection** using the **Flower** federated learning framework. The system enables multiple clients to collaboratively train a global CNN-SVM ensemble model without sharing their local tomato leaf image datasets, thereby preserving data privacy while improving model performance through distributed training.

**Disease Classes Detected (5):**
1. **Healthy** — Disease-free tomato leaves
2. **Bacterial Spot** — Bacterial infection causing dark, water-soaked spots
3. **Late Blight** — Fungal disease causing large, dark lesions
4. **Septoria Leaf Spot** — Fungal infection with circular spots and pycnidia
5. **Yellow Leaf** — Chlorosis/yellowing caused by nutrient deficiency or viral infection

**Key Features:**
- Federated Learning using Flower Framework
- FedAvg (Federated Averaging) Aggregation Strategy
- CNN-based Feature Extraction (128-dimensional feature vectors)
- SVM-based Classification on extracted features
- Ensemble prediction (60% CNN + 40% SVM weighted average)
- Privacy-Preserving Distributed Training
- Per-client granular monitoring (accuracy, precision, recall, F1-score)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SERVER (Laptop/PC)                        │
│                                                                  │
│   Flower Server (port 8080)                                      │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  FedAvg Strategy                                          │   │
│   │  - Aggregates CNN weights from all clients               │   │
│   │  - Weighted average based on client dataset sizes        │   │
│   │  - Distributes updated global model back to clients      │   │
│   └─────────────────────────────────────────────────────────┘    │
│                       ▲                    ▼                     │
│                       │                    │                     │
│         CNN weights ▲ │                    │ ▼ CNN weights       │
│         (client → server)                  (server → client)    │
│                       │                    │                     │
└───────────────────────┼────────────────────┼─────────────────────┘
                        │                    │
         ┌──────────────┴─────┐    ┌────────┴──────────────┐
         │   Client 1 (RPi)   │    │   Client 2 (RPi)      │
         │                    │    │                        │
         │ Input: client1/    │    │ Input: client2/        │
         │         dataset    │    │         dataset        │
         │                    │    │                        │
         │ ┌──────────────┐   │    │ ┌──────────────┐      │
         │ │ CNN Feature  │   │    │ │ CNN Feature  │      │
         │ │ Extractor    │   │    │ │ Extractor    │      │
         │ │ → 128-dim    │   │    │ │ → 128-dim    │      │
         │ └──────┬───────┘   │    │ └──────┬───────┘      │
         │        │           │    │        │              │
         │        ▼           │    │        ▼              │
         │ ┌──────────────┐   │    │ ┌──────────────┐      │
         │ │ SVM Trainer  │   │    │ │ SVM Trainer  │      │
         │ │ (local only) │   │    │ │ (local only) │      │
         │ └──────────────┘   │    │ └──────────────┘      │
         │                    │    │                        │
         │ ┌──────────────┐   │    │ ┌──────────────┐      │
         │ │ Ensemble     │   │    │ │ Ensemble     │      │
         │ │ (0.6 CNN +   │   │    │ │ (0.6 CNN +   │      │
         │ │  0.4 SVM)    │   │    │ │  0.4 SVM)    │      │
         │ └──────────────┘   │    │ └──────────────┘      │
         └────────────────────┘    └────────────────────────┘
```

**Important Architectural Note:** Only the **CNN weights** are shared with the server. The **SVM models remain local** to each client, adding an extra layer of data privacy since the SVM is trained on locally extracted features and never leaves the client device.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.x | Core programming language |
| Flower (flwr) | 1.30.0 | Federated Learning framework |
| PyTorch | 2.12.0 | CNN model definition & training |
| torchvision | 0.27.0 | Image transforms & dataset handling |
| scikit-learn | 1.8.0 | SVM classifier, evaluation metrics |
| NumPy | 2.4.2 | Numerical operations |
| Pandas | 3.0.0 | Metrics CSV export |
| Pillow (PIL) | 12.1.1 | Image loading & preprocessing |

---

## Dataset Structure

The dataset is split across two clients, each having their own private data partition:

```
Federated-Learning-Project/
│
├── client1/
│   ├── healthy/              # Disease-free tomato leaf images
│   ├── bacterial spot/       # Bacterial spot infected images
│   ├── late blight/          # Late blight infected images
│   ├── septoria leaf spot/   # Septoria leaf spot infected images
│   └── yellow leaf/          # Yellow leaf (chlorosis) images
│
├── client2/
│   ├── healthy/
│   ├── bacterial spot/
│   ├── late blight/
│   ├── septoria leaf spot/
│   └── yellow leaf/
│
├── client1.py
├── client2.py
├── server.py
├── cnn_model.py
├── svm_model.py
├── evaluation.py
├── client1_metrics.csv
└── client2_metrics.csv
```

Each leaf image is loaded via Pillow, converted to RGB, and resized to **64×64 pixels** before being converted to PyTorch tensors.

---

## Component Breakdown

### CNN Model (cnn_model.py)

The CNN serves dual purposes: (1) feature extraction for the SVM and (2) standalone classification for ensemble voting.

**Architecture:**

```
Input Image (3×64×64 RGB)
        │
        ▼
┌─────────────────────────────────────┐
│         CONV BLOCK 1                │
│  Conv2d(3→16, 3×3, pad=1)          │
│  BatchNorm2d(16)                    │
│  ReLU                               │
│  MaxPool2d(2×2, stride=2)           │
│  Output: 16×32×32                   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│         CONV BLOCK 2                │
│  Conv2d(16→32, 3×3, pad=1)         │
│  BatchNorm2d(32)                    │
│  ReLU                               │
│  MaxPool2d(2×2, stride=2)           │
│  Output: 32×16×16                   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│         CONV BLOCK 3                │
│  Conv2d(32→64, 3×3, pad=1)         │
│  BatchNorm2d(64)                    │
│  ReLU                               │
│  MaxPool2d(2×2, stride=2)           │
│  Output: 64×8×8                     │
└─────────────────────────────────────┘
        │
        ▼ (Flatten: 64×8×8 = 4096)
        │
┌─────────────────────────────────────┐
│      FEATURE LAYER                  │
│  Linear(4096 → 128)                 │
│  ReLU                               │
│  Output: 128-dimensional vector     │
└─────────────────────────────────────┘
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
  (for SVM training)      Dropout(0.5)
        │                       │
        │                       ▼
        │              ┌────────────────┐
        │              │  Classifier    │
        │              │  Linear(128→5) │
        │              └────────────────┘
        │                       │
        │                       ▼
        │              Softmax → Class Probabilities
```

**Key Methods:**
- `extract_features(x)` — Forward pass through conv blocks + feature layer, returning 128-dimensional feature vectors (used for SVM training)
- `forward(x)` — Full forward pass through feature extraction, dropout, and classifier, returning raw logits
- `predict_proba(x)` — Returns softmax probabilities over the 5 disease classes

---

### SVM Model (svm_model.py)

A federated linear SVM implemented using scikit-learn's `SGDClassifier` with log loss (enabling probability outputs).

```python
self.model = SGDClassifier(
    loss="log_loss",        # Logistic regression-style loss for probability outputs
    learning_rate="optimal", # Adaptive learning rate schedule
    random_state=42          # Reproducibility
)
```

**Key Methods:**
- `initialize(X, y)` — First-time partial fit with explicit class array `[0, 1, 2, 3, 4]`
- `train(X, y)` — Incremental training via `partial_fit()` (supports online/federated learning)
- `predict(X)` — Class predictions
- `predict_proba(X)` — Class probability estimates (used in ensemble)
- `get_parameters()` — Returns SVM coefficients and intercept (for potential future server-side aggregation)
- `set_parameters(parameters)` — Loads pre-trained coefficients and intercept

**Critical Design Decision:** SVM remains **local to each client** and is retrained each round on features extracted by the (potentially updated) global CNN. This means:
- The SVM adapts to newly extracted features from the aggregated CNN
- Different clients may have different SVM decision boundaries
- Only CNN weights are federated/aggregated

---

### Evaluation Module (evaluation.py)

Provides comprehensive model evaluation using scikit-learn metrics.

**Metrics Calculated:**
| Metric | Function | Averaging |
|---|---|---|
| Accuracy | `accuracy_score()` | Global |
| Precision | `precision_score()` | Weighted |
| Recall | `recall_score()` | Weighted |
| F1-Score | `f1_score()` | Weighted |
| Log Loss | `log_loss()` | N/A |
| Confusion Matrix | `confusion_matrix()` | Full 5×5 |
| Classification Report | `classification_report()` | Per-class |

**Functions:**
- `evaluate_model(y_true, y_pred, y_prob=None)` — Returns dictionary with all metrics
- `print_metrics(round_number, results, client_name)` — Formatted console output

---

### Flower Client (client1.py / client2.py)

Both clients are structurally identical except for:
- Dataset path (`client1/` vs `client2/`)
- Metric history variable names (`client1_*` vs `client2_*`)
- Metrics CSV output filename (`client1_metrics.csv` vs `client2_metrics.csv`)

**Data Pipeline (per client):**

```
Image Files (JPG)
    │
    ▼
PIL Image.open() → RGB conversion
    │
    ▼
Transform Pipeline:
    ├── Resize(64, 64)
    └── ToTensor()
    │
    ▼
PyTorch Dataset (TomatoDataset)
    │
    ▼
random_split(70% train, 30% test)
    │
    ├── trainloader (batch_size=16, shuffle=True)
    └── testloader (batch_size=16, shuffle=False)
```

**TomatoDataset Class:**
- Maps folder names to integer labels via `CLASS_NAMES` list
- Skips missing class directories gracefully with `continue`
- Lazy-loads images on `__getitem__` call (not preloaded in memory)

**FlowerClient Class (extends `fl.client.NumPyClient`):**

```python
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config)    # → CNN weights as numpy arrays
    def set_parameters(self, parameters) # ← Global CNN weights from server
    def fit(self, parameters, config)    # Train + Evaluate → return metrics
    def evaluate(self, parameters, config) # Evaluate only → return metrics
```

**Weight exchange format (get/set_parameters):**
```
CNN state_dict keys: conv1.weight, conv1.bias, bn1.weight, bn1.bias,
                     bn1.running_mean, bn1.running_var, conv2.weight, ...
                     → Flattened to list of numpy arrays
```

---

### Flower Server (server.py)

The central aggregation server using Flower's gRPC infrastructure.

**Configuration:**
```python
strategy = FedAvg(
    fraction_fit=1.0,             # All clients participate in training
    fraction_evaluate=1.0,        # All clients participate in evaluation
    min_fit_clients=2,            # Minimum 2 clients required to start training
    min_evaluate_clients=2,       # Minimum 2 clients required for evaluation
    min_available_clients=2,      # Wait until 2 clients are connected
    evaluate_metrics_aggregation_fn=weighted_average
)
```

**Weighted Average Aggregation:**
```python
def weighted_average(metrics):
    # metrics = [(num_examples, {accuracy, precision, recall, f1}), ...]
    # Returns weighted average based on client dataset sizes
    weighted_metric = sum(num_examples * metric_value) / total_examples
```

**Server Parameters:**
- Address: `0.0.0.0:8080` (listens on all interfaces)
- Rounds: 20 (configurable via `ServerConfig(num_rounds=20)`)

---

## Federated Learning Workflow

The complete workflow for one communication round:

```
ROUND START
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  SERVER distributes global CNN weights to all clients    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  CLIENT i (for each client in parallel):                 │
│                                                          │
│  1. set_parameters(global_weights)                       │
│     → Load server's aggregated CNN weights               │
│                                                          │
│  2. train():                                             │
│     a. Forward pass: images → CNN → logits               │
│     b. CrossEntropyLoss backpropagation                  │
│     c. Adam optimizer step (lr=0.001)                    │
│     d. Extract 128-dim features via cnn.extract_features │
│     e. Train local SVM on extracted features             │
│                                                          │
│  3. evaluate_local(round_number):                        │
│     a. CNN inference on test set                         │
│     b. SVM inference on CNN-extracted features           │
│     c. Ensemble: 0.6 × CNN_probs + 0.4 × SVM_probs      │
│     d. Compute accuracy, precision, recall, F1, loss     │
│     e. Save metrics to history lists                     │
│     f. Save per-round metrics to CSV                     │
│                                                          │
│  4. Return (CNN_weights, num_samples, metrics_dict)      │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  SERVER aggregates results:                              │
│                                                          │
│  1. FedAvg:                                             │
│     global_weights = Σ(client_weights × client_size)     │
│                      ────────────────────────────        │
│                          total_samples                   │
│                                                          │
│  2. Weighted average of metrics for logging              │
│                                                          │
│  3. Store aggregated global model                        │
│                                                          │
│  4. Repeat for num_rounds iterations                     │
└──────────────────────────────────────────────────────────┘
    │
    ▼
ROUND END → Next round
```

---

## Client Training Pipeline (Detailed)

### 1. Dataset Loading
```python
dataset = TomatoDataset("client1", transform)
# transform = Compose([Resize(64,64), ToTensor()])
# Dataset splits: 70% train, 30% test
# Batch size: 16
```

### 2. CNN Training per Epoch
```python
for images, labels in trainloader:
    optimizer.zero_grad()      # Reset gradients
    outputs = cnn(images)      # Forward pass
    loss = CrossEntropyLoss(outputs, labels)  # Compute loss
    loss.backward()             # Backpropagation
    optimizer.step()            # Adam optimizer update
    
    # Extract features for SVM
    features = cnn.extract_features(images)
    all_features.append(features)
    all_labels.append(labels)

# Train SVM on extracted features
X = np.array(all_features)     # Shape: (num_samples, 128)
y = np.array(all_labels)       # Shape: (num_samples,)
svm.train(X, y)                # partial_fit()
```

**Training Hyperparameters:**
- Optimizer: Adam (default betas: 0.9, 0.999)
- Learning rate: 0.001
- Loss function: CrossEntropyLoss
- Batch size: 16
- Epochs: 1 per federated round

### 3. Evaluation Pipeline
```python
cnn.eval()  # Set to evaluation mode

with torch.no_grad():
    for images, labels in testloader:
        # 1. CNN predictions
        cnn_outputs = cnn(images)
        _, cnn_preds = torch.max(cnn_outputs, 1)
        cnn_correct += (cnn_preds == labels).sum().item()
        
        # 2. CNN probabilities for ensemble
        cnn_probs = cnn.predict_proba(images)
        
        # 3. Extract features for SVM
        features = cnn.extract_features(images)
        
        # 4. SVM predictions
        svm_preds = svm.predict(features)
        svm_probs = svm.predict_proba(features)
        
        # 5. Ensemble (weighted average)
        ensemble_probs = 0.6 * cnn_probs + 0.4 * svm_probs
        predictions = np.argmax(ensemble_probs, axis=1)
```

---

## Ensemble Prediction Mechanism

The ensemble combines CNN and SVM predictions using a **weighted probability averaging** strategy:

```
For each test sample:
    CNN_probs  = softmax(CNN_logits)     # Shape: (5,) — 5 disease classes
    SVM_probs  = SVM.predict_proba(feats) # Shape: (5,)
    
    Ensemble_probs = 0.6 × CNN_probs  +  0.4 × SVM_probs
    
    Final_Class = argmax(Ensemble_probs)
```

**Weights:**
- CNN contribution: **60%** (higher weight due to superior feature learning)
- SVM contribution: **40%** (complementary linear classifier on deep features)

**Rationale:** CNNs excel at learning hierarchical spatial features from raw pixels, while SVMs provide robust classification in high-dimensional spaces. The ensemble leverages both strengths — the CNN's representational power and the SVM's generalization on fixed features.

**Per-round monitoring includes:**
- CNN Accuracy (standalone)
- SVM Accuracy (standalone)
- Ensemble Accuracy (combined)
- Precision, Recall, F1-Score (ensemble)
- Log Loss (ensemble)

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies
```bash
pip install flwr torch torchvision numpy pandas pillow scikit-learn
```

### Dataset Preparation
Ensure the dataset is organized as shown in the [Dataset Structure](#dataset-structure) section, with images placed in the appropriate class subdirectories under `client1/` and `client2/`.

---

## Running the Project

### 1. Start the Flower Server
```bash
python server.py
```
Expected output:
```
===================================
Starting Flower Server
CNN + SVM Ensemble
FedAvg Aggregation
===================================
INFO :      Starting Flower server, config: num_rounds=20, no round_timeout
INFO :      Flower ECE: gRPC server running (20 rounds), SSL is disabled
INFO :      [INIT]
INFO :      Requesting initial parameters from one random client
```

### 2. Start Client 1 (in a new terminal)
```bash
python client1.py
```

### 3. Start Client 2 (in a new terminal)
```bash
python client2.py
```

### 4. Monitor Training Progress
Once both clients connect, the server will:
1. Request initial CNN parameters from one random client
2. Distribute global model to all clients
3. Begin federated rounds

Each client outputs per-round metrics:
```
==================================================
Client1 - ROUND 5
==================================================
Accuracy  : 0.8721
Loss      : 0.0000
Precision : 0.9043
Recall    : 0.8721
F1 Score  : 0.8595

Confusion Matrix:
[[...]]

Classification Report:
              precision    recall  f1-score   support
           0     0.xxxx    0.xxxx    0.xxxx        xx
           ...

=========================================
CNN Accuracy      : 0.8679
SVM Accuracy      : 0.8595
Ensemble Accuracy : 0.8721
=========================================
```

### 5. Training Completion
After all rounds complete:
- `client1_metrics.csv` and `client2_metrics.csv` are generated with per-round metrics
- The server shuts down gracefully

---

## Evaluation Metrics

Metrics are computed **per client** at the end of each federated round:

| Metric | Description | Formula |
|---|---|---|
| **Accuracy** | Proportion of correct predictions | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | Weighted average of per-class precision | Σ (support × precision_class) / total_samples |
| **Recall** | Weighted average of per-class recall | Σ (support × recall_class) / total_samples |
| **F1-Score** | Weighted harmonic mean of precision & recall | 2 × (precision × recall) / (precision + recall) |
| **Log Loss** | Cross-entropy loss on ensemble probabilities | -Σ y_true × log(y_prob) |
| **CNN Accuracy** | Accuracy of CNN classifier alone | CNN_correct / CNN_total |
| **SVM Accuracy** | Accuracy of SVM classifier alone | SVM_correct / SVM_total |

Per-class metrics are also available in the printed classification report.

---

## Results

### Client 1 Final Metrics (Round 40)
| Metric | Value |
|---|---|
| CNN Accuracy | 0.9832 (98.32%) |
| SVM Accuracy | 0.9853 (98.53%) |
| Ensemble Accuracy | 0.9811 (98.11%) |
| Precision | 0.9813 |
| Recall | 0.9811 |
| F1-Score | 0.9811 |

### Client 2 Final Metrics (Round 40)
| Metric | Value |
|---|---|
| CNN Accuracy | 0.9686 (96.86%) |
| SVM Accuracy | 0.9644 (96.44%) |
| Ensemble Accuracy | 0.9748 (97.48%) |
| Precision | 0.9754 |
| Recall | 0.9748 |
| F1-Score | 0.9749 |

### Training Convergence
Both clients achieved >95% ensemble accuracy within the first 12-15 rounds, with steady improvement toward ~97-98% by round 40. Occasional accuracy dips (e.g., Client 2 round 13: 46.9%) indicate sensitivity to CNN weight updates from the server, where newly aggregated weights may temporarily reduce feature quality before the SVM adapts.

---

## Future Improvements

1. **Increase Federated Clients** — Scale to more than 2 clients for more robust aggregation
2. **Deploy on Raspberry Pi** — Target edge devices for real-time field deployment
3. **Advanced Backbones** — Replace simple CNN with MobileNetV2 or EfficientNet for better feature extraction
4. **Real-time Dashboard** — Web-based monitoring of training progress across clients
5. **Secure Aggregation** — Implement differential privacy and secure multi-party computation
6. **Adaptive Ensemble Weights** — Dynamically adjust CNN/SVM weighting based on validation performance
7. **SVM Aggregation** — Extend federation to SVM parameters for fully distributed ensemble
8. **Cross-Validation** — Add k-fold cross-validation within each client for more robust evaluation
9. **Data Augmentation** — Add random flips, rotations, and color jitter to improve generalization

---

## License

This project is developed as part of an academic research initiative for BTech Computer Science & Engineering.