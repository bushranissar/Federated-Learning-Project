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
- Raspberry Pi Edge Deployment
- Multi-Class Tomato Leaf Disease Detection
- Performance Evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- 
Client Classes:
1. Healthy
2. Bacterial Spot
3. Late Blight
4. Septoria Leaf Spot
5. Yellow Leaf

System Architecture:
Client Side (Raspberry Pi)
Image Dataset
↓
CNN Feature Extractor
↓
128-Dimensional Feature Vector
↓
SVM Classifier
↓
Prediction & Evaluation
↓
Send CNN Weights to Server

Server Side (Laptop)
Receive CNN Weights
↓
FedAvg Aggregation
↓
Generate Global CNN Model
↓
Send Updated Global Model Back to Clients

Technologies Used:
- Python
- Flower (FL Framework)
- PyTorch
- Scikit-Learn
- NumPy
- Raspberry Pi
- Federated Averaging (FedAvg)

Dataset Structure
client1/
├── healthy/
├── bacterial spot/
├── late blight/
├── septoria leaf spot/
└── yellow leaf/

client2/
├── healthy/
├── bacterial spot/
├── late blight/
├── septoria leaf spot/
└── yellow leaf/

Evaluation Metrics:
The model performance is evaluated using:
- Accuracy
- Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Federated Learning Workflow:
1. Local CNN training on each client.
2. CNN extracts deep features from leaf images.
3. SVM trains on extracted features.
4. CNN weights are sent to the Flower server.
5. FedAvg aggregates client models.
6. Updated global CNN is distributed back to clients.
7. Process repeats for multiple communication rounds.

Future Improvements:
- Increase number of federated clients
- Deploy on larger agricultural IoT networks
- Integrate MobileNet/EfficientNet feature extractors
- Real-time disease monitoring dashboard
- Secure aggregation and differential privacy
