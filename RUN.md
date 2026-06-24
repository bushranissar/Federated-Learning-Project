# How to Run This Application

> **Federated Learning Tomato Leaf Disease Detection Dashboard**

---

## Quick Start

### 1. Backend (FastAPI Server)

```bash
# Open Terminal 1 (from project root)
cd backend

# Install dependencies (one-time)
pip install -r requirements.txt

# Start the server
python app.py
```

**Server runs at:** http://localhost:8000

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

### 2. Frontend (Next.js Dashboard)

```bash
# Open Terminal 2 (from project root)
cd frontend

# Install dependencies (one-time)
npm install

# Start the development server
npm run dev
```

**Dashboard opens at:** http://localhost:3000

**Expected output:**
```
▲ Next.js 15.2.4
- Local: http://localhost:3000
✓ Ready in 3.2s
```

---

## Verify Everything Works

### Check Backend APIs

Open these URLs in your browser or use a tool like curl:

| Endpoint | URL | Expected Result |
|----------|-----|-----------------|
| Health Check | http://localhost:8000/health | `{"status":"healthy"}` |
| API Info | http://localhost:8000/ | JSON with API details |
| Model Metrics | http://localhost:8000/metrics | Accuracy, Precision, etc. |
| FL Status | http://localhost:8000/federated-status | Client count, rounds, etc. |

### Quick Test with Python (optional)

```bash
# Test health endpoint
python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"

# Test metrics
python -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://localhost:8000/metrics').read().decode()); print(f'Accuracy: {d[\"accuracy\"]*100:.2f}%')"

# Test federated status
python -c "import urllib.request,json; d=json.loads(urllib.request.urlopen('http://localhost:8000/federated-status').read().decode()); print(f'Clients: {d[\"num_clients\"]}, Rounds: {d[\"rounds_completed\"]}')"
```

---

## Test the Prediction Feature

### Using the Dashboard UI

1. Open http://localhost:3000 in your browser
2. Click **"Predict"** in the navigation bar
3. Drag and drop a tomato leaf image (PNG/JPG) onto the upload area
4. Click **"Run Prediction"**
5. View the prediction result with:
   - Predicted disease name
   - Confidence score
   - CNN vs SVM vs Ensemble comparison
   - Top-5 class probabilities chart
   - Disease information (symptoms, treatment, prevention)

### Using the API Directly

```bash
# Using curl (replace with actual image path)
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/tomato-leaf.jpg" \
  -w "\n"

# Using Python
python -c "
import urllib.request

# Read image file
with open('image.png', 'rb') as f:
    image_data = f.read()

# Create multipart request
boundary = '----Boundary'
body = (
    ('--' + boundary + '\r\n'
     'Content-Disposition: form-data; name=\"file\"; filename=\"leaf.png\"\r\n'
     'Content-Type: image/png\r\n\r\n').encode() +
    image_data +
    ('\r\n--' + boundary + '--\r\n').encode()
)

req = urllib.request.Request(
    'http://localhost:8000/predict',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)

response = urllib.request.urlopen(req)
print(response.read().decode()[:500])
"
```

---

## Dashboard Pages

### Homepage (http://localhost:3000)

The dashboard homepage displays:

- **Hero Section**: Project title and description
- **Tech Stack**: Technology cards (PyTorch, Scikit-Learn, Flower, etc.)
- **FL Overview**: Client count, rounds, aggregation strategy, status
- **System Architecture**: Interactive 9-step FL pipeline visualization
- **Model Status**: CNN, SVM, Ensemble status indicators
- **Quick Stats**: Feature dimension, classes, ensemble weights
- **Model Performance**: Accuracy, Precision, Recall, F1 score cards
- **FL Analytics**: Accuracy & Loss vs Communication Round chart

### Prediction Page (http://localhost:3000/predict)

- Upload tomato leaf image via drag-and-drop
- View prediction with CNN, SVM, Ensemble results
- See disease information with symptoms and treatment

---

## Running in Demo Mode

By default, the system runs in **Demo Mode** if no trained model files (.pth, .pkl) are found. The dashboard will:

1. Display a yellow warning banner: *"Running in Demo Mode"*
2. Generate realistic mock predictions (70-95% confidence)
3. Show CNN, SVM, and Ensemble predictions side-by-side
4. Display disease information from the metadata database
5. Keep all charts and metrics fully functional

**Demo mode is automatic** — you don't need to configure anything.

---

## Load Real Trained Models

To switch from Demo Mode to real predictions:

1. Train the models using the federated learning pipeline:
   ```bash
   python server.py   # Start Flower server
   python client1.py  # Start Client 1 training
   python client2.py  # Start Client 2 training
   ```

2. Save the trained models in one of these locations:
   ```
   models/trained/cnn_model.pth
   models/trained/svm_model.pkl
   ```

3. Restart the backend — it will automatically detect and load the models.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Backend won't start | Run `pip install -r requirements.txt` again |
| Frontend shows errors | Run `npm install` then `npm run dev` |
| Cannot connect to backend | Ensure backend is running on port 8000 |
| Images not uploading | Check file is PNG/JPG under 10MB |
| Charts not displaying | Ensure backend is running (frontend fetches data on load) |
| Port 8000 already in use | Change PORT in `backend/.env` |
| Port 3000 already in use | Next.js will automatically use 3001 |

---

## Project Structure Overview

```
Federated-Learning-Project/
├── backend/         # Python FastAPI server (port 8000)
├── frontend/        # Next.js dashboard (port 3000)
├── client1_metrics.csv   # Real training data
├── client2_metrics.csv   # Real training data
├── cnn_model.py          # CNN architecture (reused)
├── svm_model.py          # SVM model (reused)
├── image.png             # Sample test image
├── client1/              # Client 1 dataset
├── client2/              # Client 2 dataset
├── README.md             # Full documentation
└── RUN.md                # This file