# Federated-Learning-Project
Federated Learning Based Tomato Leaf Disease Detection using CNN-SVM:
This project implements a Federated Learning framework for Tomato Leaf Disease Detection using Flower, Raspberry Pi devices, Convolutional Neural Networks (CNN), and Support Vector Machines (SVM). The system enables multiple clients to collaboratively train a global model without sharing their local datasets, preserving data privacy while improving model performance.

Features:
- Federated Learning using Flower Framework
- FedAvg Aggregation Strategy
- CNN-based Feature Extraction
- SVM-based Classification
- Privacy-Preserving Distributed Training
- Raspberry Pi Edge Deployment
- Multi-Class Tomato Leaf Disease Detection
- Performance Evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix

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
