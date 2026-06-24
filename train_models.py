"""
Train and Save CNN + SVM Models for Tomato Leaf Disease Detection
Trains the model using all data from client1 and client2
Saves to models/trained/ directory
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import joblib

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cnn_model import CNNModel
from svm_model import FederatedSVM

# Configuration
DATA_DIRS = ["client1", "client2"]
CLASSES = ["healthy", "bacterial spot", "late blight", "septoria leaf spot", "yellow leaf"]
CLASS_TO_IDX = {cls.lower(): i for i, cls in enumerate(CLASSES)}
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models/trained"

print(f"Using device: {DEVICE}")
print(f"Classes: {CLASSES}")
os.makedirs(MODEL_DIR, exist_ok=True)


class TomatoLeafDataset(Dataset):
    """Dataset for loading tomato leaf images"""
    
    def __init__(self, data_dirs, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"Warning: {data_dir} not found, skipping")
                continue
                
            for class_name in os.listdir(data_dir):
                class_path = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                    
                # Find class index (case-insensitive partial match)
                class_lower = class_name.lower()
                class_idx = None
                for known_class, idx in CLASS_TO_IDX.items():
                    if known_class in class_lower or class_lower in known_class:
                        class_idx = idx
                        break
                
                if class_idx is None:
                    print(f"Warning: Unknown class directory: {class_name}")
                    continue
                
                # Load images
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        img_path = os.path.join(class_path, img_file)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images from {data_dirs}")
        for i, cls in enumerate(CLASSES):
            count = self.labels.count(i)
            print(f"  {cls}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", IMG_SIZE, (0, 0, 0))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """Get image transforms for training"""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform():
    """Get image transforms for evaluation"""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def train_cnn():
    """Train CNN model and return trained model"""
    print("\n" + "="*60)
    print("TRAINING CNN MODEL")
    print("="*60)
    
    transform = get_transforms()
    dataset = TomatoLeafDataset(DATA_DIRS, transform=transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use eval transform for validation
    if hasattr(val_dataset, 'dataset'):
        val_dataset.dataset.transform = get_eval_transform()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    # Initialize model
    model = CNNModel(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "cnn_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved best CNN model (val_acc={val_acc:.2f}%) to {model_path}")
    
    # Load best model
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "cnn_model.pth"),
        map_location=DEVICE
    ))
    model.eval()
    print(f"\nCNN Training Complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model


def train_svm(cnn_model):
    """Train SVM model using CNN-extracted features"""
    print("\n" + "="*60)
    print("TRAINING SVM MODEL (on CNN 128-D features)")
    print("="*60)
    
    # Load all data for feature extraction
    transform = get_eval_transform()
    dataset = TomatoLeafDataset(DATA_DIRS, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    cnn_model.eval()
    
    all_features = []
    all_labels = []
    
    print(f"Extracting 128-D features from {len(dataset)} images...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            features = cnn_model.extract_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    # Train SVM
    svm = FederatedSVM()
    
    print("Training SVM...")
    svm.initialize(X_train, y_train)
    
    # Additional training passes
    for epoch in range(1, 6):
        svm.train(X_train, y_train)
        
        # Evaluate
        train_pred = svm.predict(X_train)
        val_pred = svm.predict(X_val)
        
        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"SVM Epoch {epoch}: Train Acc={train_acc*100:.2f}%, Val Acc={val_acc*100:.2f}%")
    
    # Evaluate final model
    y_pred = svm.predict(X_val)
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f"\nSVM Validation Results:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    return svm


def evaluate_ensemble(cnn_model, svm_model):
    """Evaluate CNN-SVM ensemble performance"""
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION (60% CNN + 40% SVM)")
    print("="*60)
    
    transform = get_eval_transform()
    dataset = TomatoLeafDataset(DATA_DIRS, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    cnn_model.eval()
    
    all_cnn_probs = []
    all_svm_probs = []
    all_labels = []
    
    print("Evaluating ensemble on all data...")
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            
            # CNN prediction
            outputs = cnn_model(images)
            cnn_probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            # SVM prediction from features
            features = cnn_model.extract_features(images).cpu().numpy()
            svm_probs = svm_model.predict_proba(features)
            
            all_cnn_probs.append(cnn_probs)
            all_svm_probs.append(svm_probs)
            all_labels.append(labels.numpy())
    
    cnn_probs = np.concatenate(all_cnn_probs, axis=0)
    svm_probs = np.concatenate(all_svm_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    
    # Ensemble: 0.6 * CNN + 0.4 * SVM
    ensemble_probs = 0.6 * cnn_probs + 0.4 * svm_probs
    y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
    y_pred_cnn = np.argmax(cnn_probs, axis=1)
    y_pred_svm = np.argmax(svm_probs, axis=1)
    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    
    cnn_acc = accuracy_score(y_true, y_pred_cnn)
    svm_acc = accuracy_score(y_true, y_pred_svm)
    ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
    
    print(f"\nResults:")
    print(f"  CNN Accuracy:      {cnn_acc*100:.2f}%")
    print(f"  SVM Accuracy:      {svm_acc*100:.2f}%")
    print(f"  Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_ensemble)
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred_ensemble, target_names=CLASSES, digits=4))
    
    return ensemble_acc


if __name__ == "__main__":
    print("="*60)
    print("TRAINING TOMATO LEAF DISEASE DETECTION MODELS")
    print("="*60)
    print(f"Data directories: {DATA_DIRS}")
    print(f"Model save path: {os.path.abspath(MODEL_DIR)}")
    
    # Step 1: Train CNN
    cnn_model = train_cnn()
    
    # Step 2: Train SVM
    svm_model = train_svm(cnn_model)
    
    # Step 3: Save SVM model
    svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
    # Save the internal sklearn model
    import joblib as jl
    jl.dump(svm_model.model, svm_path)
    print(f"\nSVM model saved to {svm_path}")
    
    # Also save FederatedSVM wrapper
    full_svm_path = os.path.join(MODEL_DIR, "svm_model_full.pkl")
    jl.dump(svm_model, full_svm_path)
    print(f"Full SVM wrapper saved to {full_svm_path}")
    
    # Step 4: Evaluate ensemble
    ensemble_acc = evaluate_ensemble(cnn_model, svm_model)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFiles saved:")
    print(f"  {os.path.join(MODEL_DIR, 'cnn_model.pth')}")
    print(f"  {os.path.join(MODEL_DIR, 'svm_model.pkl')}")
    print(f"  {os.path.join(MODEL_DIR, 'svm_model_full.pkl')}")
    print(f"\nFinal Ensemble Accuracy: {ensemble_acc*100:.2f}%")
    print(f"To use these models, restart the backend:")
    print(f"  cd backend && python app.py")