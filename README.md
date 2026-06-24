# Federated Learning Tomato Leaf Disease Detection

> **Federated Learning based Tomato Leaf Disease Detection using CNN-SVM Ensemble**
> *A decentralized deep learning system for detecting tomato leaf diseases across multiple clients*

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [API Documentation](#api-documentation)
- [How It Works](#how-it-works)
- [Disease Classes](#disease-classes)
- [Model Performance](#model-performance)
- [Federated Learning Analytics](#federated-learning-analytics)
- [Demo Mode](#demo-mode)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project implements a **Federated Learning (FL) based Tomato Leaf Disease Detection** system using a **CNN-SVM Ensemble** architecture. The system enables collaborative model training across distributed clients without sharing raw data, preserving privacy while improving model accuracy.

### Key Highlights

- **Federated Learning**: Uses the Flower Framework with FedAvg aggregation strategy
- **CNN Feature Extractor**: Custom CNN architecture extracting 128-dimensional feature vectors
- **SVM Classifier**: Federated Linear SVM using SGDClassifier with log-loss
- **Ensemble Prediction**: Weighted combination (60% CNN + 40% SVM) for robust classification
- **5 Disease Classes**: Healthy, Bacterial Spot, Late Blight, Septoria Leaf Spot, Yellow Leaf
- **2 Federated Clients**: Distributed training across multiple data sources
- **40 Communication Rounds**: Achieved 98.11% final ensemble accuracy

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐                                    │
│  │ Client 1 │    │ Client 2 │              FEDERATED             │
│  │ Dataset  │    │ Dataset  │              CLIENTS               │
│  └────┬─────┘    └────┬─────┘                                    │
│       │               │                                          │
│  ┌────▼─────┐    ┌────▼─────┐                                    │
│  │    CNN   │    │    CNN   │         CNN FEATURE                │
│  │ Extractor│    │ Extractor│         EXTRACTOR                  │
│  └────┬─────┘    └────┬─────┘         (128-D vectors)            │
│       │               │                                          │
│  ┌────▼─────┐    ┌────▼─────┐                                    │
│  │    SVM   │    │    SVM   │         SVM CLASSIFIER             │
│  │ Classifier│   │ Classifier│        (Linear SGD)               │
│  └────┬─────┘    └────┬─────┘                                    │
│       │               │                                          │
│  ┌────▼─────┐    ┌────▼─────┐         ENSEMBLE                  │
│  │ Ensemble │    │ Ensemble │         60% CNN + 40% SVM          │
│  └────┬─────┘    └────┬─────┘                                    │
│       │               │                                          │
│  ┌────▼───────────────▼─────┐                                    │
│  │     FLOWER CLIENTS       │        FLOWER FRAMEWORK            │
│  │  (Send model parameters) │                                    │
│  └────┬─────────────────────┘                                    │
│       │                                                          │
│  ┌────▼─────────────────────┐                                    │
│  │   FEDAVG SERVER          │        FEDERATED                  │
│  │   (Aggregate models)     │        AVERAGING                   │
│  └────┬─────────────────────┘                                    │
│       │                                                          │
│  ┌────▼─────────────────────┐                                    │
│  │   GLOBAL MODEL           │        GLOBAL MODEL               │
│  └────┬─────────────────────┘     (Distributed to clients)       │
│       │                                                          │
│  ┌────▼─────────────────────┐                                    │
│  │   PREDICTION DASHBOARD   │        RESEARCH                   │
│  │  (Next.js + FastAPI)     │        DASHBOARD                  │
│  └──────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **PyTorch** | CNN Deep Learning Model |
| **Scikit-Learn** | SVM Classifier (SGDClassifier) |
| **Flower Framework** | Federated Learning (FedAvg) |
| **FastAPI** | Backend REST API Server |
| **Next.js 15** | Frontend Dashboard (React) |
| **TypeScript** | Type-safe Frontend Code |
| **TailwindCSS** | UI Styling |
| **Recharts** | Data Visualization |
| **Lucide React** | Icons |
| **Pandas** | CSV Metrics Parsing |
| **Pillow** | Image Processing |

---

## Project Structure

```
Federated-Learning-Project/
│
├── backend/                          # FastAPI Backend
│   ├── app.py                        # Server entry point (port 8000)
│   ├── config.py                     # Environment configuration
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── routers/
│   │   ├── predict.py                # POST /predict - Image inference
│   │   ├── metrics.py                # GET /metrics - Model performance
│   │   └── federated_status.py       # GET /federated-status - FL analytics
│   │
│   ├── models/
│   │   ├── cnn_inference.py          # CNN wrapper (reuses cnn_model.py)
│   │   ├── svm_inference.py          # SVM wrapper (reuses svm_model.py)
│   │   └── ensemble.py               # Weighted ensemble logic (0.6/0.4)
│   │
│   ├── services/
│   │   ├── image_processor.py        # Image validation & preprocessing
│   │   ├── model_loader.py           # Auto-load / fallback to demo mode
│   │   ├── mock_predictor.py         # Realistic demo predictions
│   │   └── metrics_aggregator.py     # Parse CSV files for real metrics
│   │
│   └── utils/
│       └── constants.py              # Disease metadata & descriptions
│
├── frontend/                         # Next.js 15 Frontend
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── package.json
│   │
│   └── src/
│       ├── app/
│       │   ├── layout.tsx            # Root layout with Header
│       │   ├── page.tsx              # Dashboard Homepage
│       │   └── predict/
│       │       └── page.tsx          # Prediction Page
│       │
│       ├── components/
│       │   ├── layout/
│       │   │   └── Header.tsx        # Navy header with navigation
│       │   ├── dashboard/
│       │   │   └── SystemArchitecture.tsx  # Interactive FL pipeline
│       │   ├── metrics/
│       │   │   └── MetricsCards.tsx       # Accuracy, Precision, etc.
│       │   ├── fl-analytics/
│       │   │   └── AccuracyVsRound.tsx    # Recharts visualization
│       │   ├── predict/
│       │   │   ├── ImageUploader.tsx      # Drag-and-drop upload
│       │   │   └── PredictionResult.tsx   # Results + disease info
│       │   └── common/
│       │       ├── LoadingSpinner.tsx
│       │       └── ErrorBoundary.tsx
│       │
│       ├── api-services/
│       │   ├── client.ts             # Axios instance
│       │   ├── predictor.ts          # Prediction API calls
│       │   ├── metrics.ts            # Metrics API calls
│       │   └── federated.ts          # FL analytics API calls
│       │
│       ├── types/
│       │   └── index.ts              # TypeScript interfaces
│       │
│       └── lib/
│           └── utils.ts              # Utility functions
│
├── client1_metrics.csv               # Client 1 training metrics (40 rounds)
├── client2_metrics.csv               # Client 2 training metrics (40 rounds)
├── cnn_model.py                      # Original CNN model architecture
├── svm_model.py                      # Original SVM model (SGDClassifier)
├── evaluation.py                     # Original evaluation functions
├── server.py                         # Original Flower server (FedAvg)
├── client1.py                        # Original Flower client 1
├── client2.py                        # Original Flower client 2
│
├── client1/                          # Client 1 dataset
│   ├── bacterial spot/
│   ├── healthy/
│   ├── late blight/
│   ├── septoria leaf spot/
│   └── yellow leaf/
│
├── client2/                          # Client 2 dataset
│   ├── bacterial spot/
│   ├── healthy/
│   ├── late blight/
│   ├── septoria leaf spot/
│   └── yellow leaf/
│
└── image.png                         # Sample test image
```

---

## Features

### 1. Dashboard Homepage (`/`)
- Project title, subtitle, and overview description
- Technology stack cards
- Federated Learning overview with metrics
- Interactive system architecture visualization (9-step FL pipeline)
- Model status indicators (CNN, SVM, Ensemble)
- Quick stats (128-D features, 5 classes, ensemble weights)
- Model performance metrics cards (Accuracy, Precision, Recall, F1)
- Federated Learning analytics chart (Accuracy & Loss vs Round)

### 2. Disease Prediction (`/predict`)
- Drag-and-drop image upload
- Image preview with clear button
- CNN, SVM, and Ensemble predictions side-by-side
- Top-5 class probabilities bar chart
- Disease information:
  - Description
  - Symptoms
  - Severity level
  - Treatment recommendations
  - Prevention tips
- Demo mode indicator (when models unavailable)
- Loading, error, and empty states

### 3. Model Metrics
- Accuracy: 98.11%
- Precision: 98.13%
- Recall: 98.11%
- F1 Score: 98.11%
- Real data from training CSV files

### 4. Federated Learning Analytics
- 2 Clients active
- 40 Communication rounds completed
- FedAvg aggregation strategy
- Accuracy progression over rounds (line chart)
- Loss progression over rounds (line chart)
- Client performance comparison

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
# Navigate to the project root
cd Federated-Learning-Project

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Start the FastAPI server
python app.py
```

The backend server starts at **http://localhost:8000**

### Frontend Setup

```bash
# Open a new terminal
cd Federated-Learning-Project

# Install frontend dependencies
cd frontend
npm install

# Start the development server
npm run dev
```

The frontend dashboard starts at **http://localhost:3000**

### Environment Variables

#### Backend (`backend/.env`)

```env
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000
CLIENT1_METRICS=client1_metrics.csv
CLIENT2_METRICS=client2_metrics.csv
CNN_MODEL_PATH=models/trained/cnn_model.pth
SVM_MODEL_PATH=models/trained/svm_model.pkl
DEMO_MODE=auto
```

#### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## API Documentation

### `GET /`

Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Federated Learning Tomato Leaf Disease Detection API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": [
    "POST /predict - Upload image for disease prediction",
    "GET /metrics - Get model performance metrics",
    "GET /federated-status - Get FL training status",
    "GET / - API health check"
  ]
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "healthy" }
```

### `POST /predict`

Upload a tomato leaf image and get disease prediction.

**Request:**
- `file`: Image file (multipart/form-data, PNG/JPG/JPEG/WEBP, max 10MB)

**Response:**
```json
{
  "success": true,
  "demo_mode": false,
  "prediction": "Late Blight",
  "confidence": 0.9421,
  "cnn_prediction": "Late Blight",
  "cnn_confidence": 0.9350,
  "cnn_probabilities": { "Healthy": 0.02, "Bacterial Spot": 0.01, ... },
  "svm_prediction": "Late Blight",
  "svm_confidence": 0.9180,
  "svm_probabilities": { "Healthy": 0.03, "Bacterial Spot": 0.02, ... },
  "ensemble_prediction": "Late Blight",
  "ensemble_confidence": 0.9282,
  "ensemble_probabilities": { "Healthy": 0.025, "Bacterial Spot": 0.015, ... },
  "probabilities": { "Healthy": 0.025, "Bacterial Spot": 0.015, ... },
  "top5": [
    { "class": "Late Blight", "probability": 0.9282 },
    { "class": "Healthy", "probability": 0.0250 },
    { "class": "Bacterial Spot", "probability": 0.0150 },
    { "class": "Septoria Leaf Spot", "probability": 0.0120 },
    { "class": "Yellow Leaf", "probability": 0.0100 }
  ],
  "disease_info": {
    "name": "Late Blight",
    "description": "...",
    "symptoms": ["..."],
    "severity": "Very High",
    "treatment": "...",
    "prevention": ["..."]
  }
}
```

### `GET /metrics`

Get model performance metrics.

**Response:**
```json
{
  "accuracy": 0.9811,
  "precision": 0.9813,
  "recall": 0.9811,
  "f1_score": 0.9811,
  "loss": 0.0,
  "demo_mode": false,
  "confusion_matrix": [[96, 1, 1, 1, 1], [1, 96, 1, 1, 1], ...],
  "classification_report": {
    "Healthy": { "precision": 0.9686, "recall": 0.9486, "f1_score": 0.9586, "support": 94 },
    "Bacterial Spot": { ... },
    "Late Blight": { ... },
    "Septoria Leaf Spot": { ... },
    "Yellow Leaf": { ... }
  },
  "rounds_data": [
    { "round": 1, "cnn_accuracy": 0.8407, "svm_accuracy": 0.7652, ... },
    ...
  ],
  "client_comparison": [
    { "round": 1, "client1_accuracy": 0.8407, "client2_accuracy": 0.8092, ... },
    ...
  ],
  "total_rounds": 40
}
```

### `GET /federated-status`

Get federated learning training status and analytics.

**Response:**
```json
{
  "num_clients": 2,
  "clients": ["Client 1", "Client 2"],
  "rounds_completed": 40,
  "aggregation_strategy": "FedAvg (Federated Averaging)",
  "status": "completed",
  "global_model_status": "trained",
  "client_status": [
    { "name": "Client 1", "status": "completed", "accuracy": 0.9811, "loss": 0, "samples": 1000 },
    { "name": "Client 2", "status": "completed", "accuracy": 0.9748, "loss": 0.0997, "samples": 1000 }
  ],
  "accuracy_progression": [
    { "round": 1, "accuracy": 0.8407, "loss": 0 },
    ...
  ],
  "demo_mode": false
}
```

---

## How It Works

### Prediction Pipeline

```
Tomato Leaf Image
        ↓
 [Preprocessing]
   Resize (64×64)
   Normalize
        ↓
 [CNN Feature Extractor]
   ↓ 128-D Feature Vector
   ↓ CNN Class Probabilities
        ↓
 [SVM Classifier]
   ↓ SVM Class Probabilities
        ↓
 [Weighted Ensemble]
   0.6 × CNN + 0.4 × SVM
        ↓
 [Final Prediction]
```

### Federated Learning Workflow

1. **Initialization**: Global CNN and SVM models initialized on server
2. **Distribution**: Server distributes global model to clients
3. **Local Training**: Each client trains on private local data
4. **Parameter Upload**: Clients send updated model parameters to server
5. **Aggregation**: Server uses FedAvg to aggregate client parameters
6. **Distribution**: Updated global model sent back to clients
7. **Repeat**: Steps 3-6 for 40 communication rounds

---

## Disease Classes

| Class | Description | Severity | Treatment |
|-------|-------------|----------|-----------|
| **Healthy** | Normal tomato leaf with no disease | None | Standard care |
| **Bacterial Spot** | Xanthomonas campestris infection causing dark spots with yellow halos | Moderate-High | Copper-based bactericides |
| **Late Blight** | Phytophthora infestans causing rapid wilting and collapse | Very High | Chlorothalonil, mancozeb |
| **Septoria Leaf Spot** | Septoria lycopersici fungus causing circular spots with gray centers | Moderate | Chlorothalonil, copper fungicides |
| **Yellow Leaf** | Nutrient deficiency or TYLCV virus causing leaf yellowing | Moderate-High | Fertilizer or remove infected plants |

---

## Model Performance

### Final Metrics (Round 40)

| Metric | Value |
|--------|-------|
| **Accuracy** | **98.11%** |
| **Precision** | **98.13%** |
| **Recall** | **98.11%** |
| **F1 Score** | **98.11%** |

### Performance Progression

```
Accuracy (%)    ████████████████████████████████████████ 98.11%
Precision (%)   ████████████████████████████████████████ 98.13%
Recall (%)      ████████████████████████████████████████ 98.11%
F1 Score (%)    ████████████████████████████████████████ 98.11%

Training improved from 84.07% (Round 1) to 98.11% (Round 40)
```

---

## Federated Learning Analytics

- **Clients**: 2 (Client 1, Client 2)
- **Rounds**: 40 communication rounds
- **Aggregation**: FedAvg (Federated Averaging)
- **Status**: Completed (trained)
- **Client 1 Final Accuracy**: 98.11%
- **Client 2 Final Accuracy**: 97.48%

---

## Demo Mode

The system automatically switches to **Demo Mode** when trained model files (`.pth`, `.pkl`) are unavailable.

**Demo Mode Features:**
- Realistic mock predictions with 70-95% confidence
- CNN and SVM probability distributions
- Weighted ensemble predictions
- Full disease information display
- Clear "Demo Mode" indicator
- Dashboard remains fully functional

**Model Priority:**
1. Load trained `.pth` and `.pkl` files if available
2. Fall back to Demo Mode if models are missing
3. Dashboard never breaks due to missing models

---

## Future Improvements

1. **Model Training Pipeline**
   - Train and save CNN/SVM models for real predictions
   - Add model checkpointing during FL training

2. **Enhanced Visualizations**
   - Confusion matrix heatmap
   - Client performance comparison chart
   - Real-time training monitoring via WebSocket

3. **Deployment**
   - Docker containers for easy deployment
   - CI/CD pipeline configuration
   - Cloud deployment (AWS/GCP/Azure)

4. **Additional Features**
   - Batch prediction for multiple images
   - Model version comparison
   - Export results as PDF/CSV
   - User authentication and history

5. **Edge Deployment**
   - Raspberry Pi client optimization
   - Model quantization for edge devices
   - Real-time camera feed integration

---

## License

This project is developed for academic research purposes.

## Contributors

- [Aqibabass](https://github.com/Aqibabass)
- [Bushranissar](https://github.com/bushranissar)