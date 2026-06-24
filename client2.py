# =====================================================
# client2.py
# Part A
# CNN + SVM Ensemble Federated Learning
# Client 2
# =====================================================

import os
import torch
import numpy as np
import flwr as fl

from PIL import Image

from torch import nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

from torchvision import transforms

from cnn_model import CNNModel
from svm_model import FederatedSVM

from evaluation import (
    evaluate_model,
    print_metrics
)

# =====================================================
# CONFIGURATION
# =====================================================

DATASET_PATH = "client2"

NUM_CLASSES = 5

BATCH_SIZE = 16

LEARNING_RATE = 0.001

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [

    "healthy",

    "bacterial spot",

    "late blight",

    "septoria leaf spot",

    "yellow leaf"

]

# =====================================================
# DATASET CLASS
# =====================================================

class TomatoDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.images = []
        self.labels = []
        self.transform = transform

        for label, class_name in enumerate(
            CLASS_NAMES
        ):

            class_path = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.exists(
                class_path
            ):
                continue

            for img_name in os.listdir(
                class_path
            ):

                img_path = os.path.join(
                    class_path,
                    img_name
                )

                self.images.append(
                    img_path
                )

                self.labels.append(
                    label
                )

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        idx
    ):

        image = Image.open(
            self.images[idx]
        ).convert(
            "RGB"
        )

        if self.transform:

            image = self.transform(
                image
            )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return image, label


# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize(
        (64, 64)
    ),

    transforms.ToTensor()

])

# =====================================================
# LOAD DATASET
# =====================================================

dataset = TomatoDataset(
    DATASET_PATH,
    transform
)

print(
    "Total Images:",
    len(dataset)
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

train_size = int(
    0.7 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
)

train_dataset, test_dataset = random_split(
    dataset,
    [
        train_size,
        test_size
    ]
)

print(
    "Train Images:",
    len(train_dataset)
)

print(
    "Test Images:",
    len(test_dataset)
)

# =====================================================
# DATALOADER
# =====================================================

trainloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

testloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =====================================================
# MODEL INITIALIZATION
# =====================================================

cnn = CNNModel(
    num_classes=NUM_CLASSES
)

svm = FederatedSVM()

# =====================================================
# LOSS FUNCTION
# =====================================================

criterion = nn.CrossEntropyLoss()

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = torch.optim.Adam(
    cnn.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# HISTORY
# =====================================================

client2_accuracy_history = []
client2_loss_history = []

client2_precision_history = []
client2_recall_history = []
client2_f1_history = []

client2_cnn_accuracy_history = []
client2_svm_accuracy_history = []

client2_cnn_loss_history = []

client2_confusion_matrix_history = []

# =====================================================
# CNN + SVM TRAINING
# =====================================================

def train():

    cnn.train()

    total_loss = 0.0

    all_features = []

    all_labels = []

    for images, labels in trainloader:

        optimizer.zero_grad()

        outputs = cnn(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

        with torch.no_grad():

            features = cnn.extract_features(
                images
            )

            all_features.extend(
                features.cpu()
                .numpy()
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
            )

    X = np.array(
        all_features,
        dtype=np.float32
    )

    y = np.array(
        all_labels,
        dtype=np.int64
    )

    print(
        "Feature Shape:",
        X.shape
    )

    print(
        "Label Shape:",
        y.shape
    )

    print(
        "Classes:",
        np.unique(y)
    )

    svm.train(
        X,
        y
    )

    avg_loss = (
        total_loss
        / len(trainloader)
    )

    return avg_loss

# =====================================================
# EVALUATION
# =====================================================

def evaluate_local(
    round_number=0
):

    cnn.eval()

    y_true = []
    y_pred = []
    y_prob = []

    # -------------------------------------
    # CNN Accuracy
    # -------------------------------------

    cnn_correct = 0
    cnn_total = 0

    # -------------------------------------
    # SVM Accuracy
    # -------------------------------------

    svm_correct = 0
    svm_total = 0

    with torch.no_grad():

        for images, labels in testloader:

            # ==============================
            # CNN Accuracy
            # ==============================

            cnn_outputs = cnn(
                images
            )

            _, cnn_predictions = torch.max(
                cnn_outputs,
                1
            )

            cnn_correct += (
                cnn_predictions
                ==
                labels
            ).sum().item()

            cnn_total += (
                labels.size(0)
            )

            # ==============================
            # CNN Probabilities
            # ==============================

            cnn_probs = (
                cnn.predict_proba(
                    images
                )
                .cpu()
                .numpy()
            )

            # ==============================
            # CNN Features
            # ==============================

            features = (
                cnn.extract_features(
                    images
                )
                .cpu()
                .numpy()
            )

            # ==============================
            # SVM Accuracy
            # ==============================

            svm_predictions = (
                svm.predict(
                    features
                )
            )

            svm_correct += (
                svm_predictions
                ==
                labels.numpy()
            ).sum()

            svm_total += (
                labels.size(0)
            )

            # ==============================
            # SVM Probabilities
            # ==============================

            svm_probs = (
                svm.predict_proba(
                    features
                )
            )

            # ==============================
            # Ensemble
            # ==============================

            ensemble_probs = (

                0.6
                *
                cnn_probs

                +

                0.4
                *
                svm_probs

            )

            predictions = np.argmax(
                ensemble_probs,
                axis=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions
            )

            y_prob.extend(
                ensemble_probs
            )

    # =====================================
    # CNN Accuracy
    # =====================================

    cnn_accuracy = (
        cnn_correct
        /
        cnn_total
    )

    # =====================================
    # SVM Accuracy
    # =====================================

    svm_accuracy = (
        svm_correct
        /
        svm_total
    )

    # =====================================
    # Ensemble Metrics
    # =====================================

    results = evaluate_model(

        np.array(
            y_true
        ),

        np.array(
            y_pred
        ),

        np.array(
            y_prob
        )

    )

    # =====================================
    # Save Histories
    # =====================================

    client2_accuracy_history.append(
        results["accuracy"]
    )

    client2_loss_history.append(
        results["loss"]
    )

    client2_precision_history.append(
        results["precision"]
    )

    client2_recall_history.append(
        results["recall"]
    )

    client2_f1_history.append(
        results["f1_score"]
    )

    client2_cnn_accuracy_history.append(
        cnn_accuracy
    )

    client2_svm_accuracy_history.append(
        svm_accuracy
    )

    client2_confusion_matrix_history.append(
        results["confusion_matrix"]
    )

    # =====================================
    # Print Standard Metrics
    # =====================================

    print_metrics(
        round_number,
        results,
        "Client2"
    )

    # =====================================
    # Print Additional Metrics
    # =====================================

    print(
        "\n"
        "=============================="
    )

    print(
        f"CNN Accuracy      : "
        f"{cnn_accuracy:.4f}"
    )

    print(
        f"SVM Accuracy      : "
        f"{svm_accuracy:.4f}"
    )

    print(
        f"Ensemble Accuracy : "
        f"{results['accuracy']:.4f}"
    )

    print(
        "==============================\n"
    )

    results["cnn_accuracy"] = (
        cnn_accuracy
    )

    results["svm_accuracy"] = (
        svm_accuracy
    )

    return results


# =====================================================
# FLOWER CLIENT
# =====================================================

class FlowerClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):

        return [

            val.cpu().numpy()

            for _, val

            in cnn.state_dict().items()

        ]

    def set_parameters(
        self,
        parameters
    ):

        cnn_keys = list(
            cnn.state_dict().keys()
        )

        state_dict = {

            k: torch.tensor(v)

            for k, v in zip(
                cnn_keys,
                parameters
            )

        }

        cnn.load_state_dict(
            state_dict,
            strict=True
        )

    def fit(
        self,
        parameters,
        config
    ):

        self.set_parameters(
            parameters
        )

        train()

        return (

            self.get_parameters(
                config
            ),

            len(
                train_dataset
            ),

            {}

        )

    def evaluate(
        self,
        parameters,
        config
    ):

        self.set_parameters(
            parameters
        )

        current_round = (
            len(
                client2_accuracy_history
            )
            + 1
        )

        results = evaluate_local(
            current_round
        )

        return (

            float(
                results["loss"]
            ),

            len(
                test_dataset
            ),

            {

                "accuracy":
                float(
                    results[
                        "accuracy"
                    ]
                ),

                "precision":
                float(
                    results[
                        "precision"
                    ]
                ),

                "recall":
                float(
                    results[
                        "recall"
                    ]
                ),

                "f1":
                float(
                    results[
                        "f1_score"
                    ]
                )

            }

        )


# =====================================================
# SAVE METRICS
# =====================================================

def save_client_metrics():

    import pandas as pd

    data = {

        "Round":

        list(

            range(

                1,

                len(
                    client2_accuracy_history
                ) + 1

            )

        ),

        "CNN_Accuracy":
        client2_cnn_accuracy_history,

        "SVM_Accuracy":
        client2_svm_accuracy_history,

        "Ensemble_Accuracy":
        client2_accuracy_history,

        "Loss":
        client2_loss_history,

        "Precision":
        client2_precision_history,

        "Recall":
        client2_recall_history,

        "F1":
        client2_f1_history,

        "Confusion_Matrix":
        [cm.tolist() for cm in client2_confusion_matrix_history]

    }

    df = pd.DataFrame(
        data
    )

    df.to_csv(
        "client2_metrics.csv",
        index=False
    )

    print(
        "\nMetrics Saved : "
        "client2_metrics.csv"
    )


# =====================================================
# SAVE PLOTS
# =====================================================

def save_client_plots():

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os

    os.makedirs("plots", exist_ok=True)

    if not client2_accuracy_history:
        print("No data to plot.")
        return

    rounds = list(range(1, len(client2_accuracy_history) + 1))

    # ====== PLOT 1: Accuracy Comparison (CNN, SVM, Ensemble) ======
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, client2_cnn_accuracy_history, marker='o', linewidth=2, label='CNN', color='blue')
    plt.plot(rounds, client2_svm_accuracy_history, marker='s', linewidth=2, label='SVM', color='green')
    plt.plot(rounds, client2_accuracy_history, marker='^', linewidth=2, label='Ensemble', color='red')
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Client2 - CNN vs SVM vs Ensemble Accuracy", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("plots/client2_accuracy_comparison.png", dpi=150)
    plt.close()
    print("Saved: plots/client2_accuracy_comparison.png")

    # ====== PLOT 2: Loss vs Round ======
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, client2_loss_history, marker='o', linewidth=2, color='red', markersize=8)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Client2 - Loss vs Round", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/client2_loss_vs_round.png", dpi=150)
    plt.close()
    print("Saved: plots/client2_loss_vs_round.png")

    # ====== PLOT 3: Precision, Recall, F1-Score ======
    if client2_precision_history and client2_recall_history and client2_f1_history:
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, client2_precision_history, marker='o', linewidth=2, label='Precision', color='blue')
        plt.plot(rounds, client2_recall_history, marker='s', linewidth=2, label='Recall', color='green')
        plt.plot(rounds, client2_f1_history, marker='^', linewidth=2, label='F1-Score', color='red')
        plt.xlabel("Round", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.title("Client2 - Precision, Recall, F1-Score", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig("plots/client2_metrics.png", dpi=150)
        plt.close()
        print("Saved: plots/client2_metrics.png")

    # ====== PLOT 4: Confusion Matrix (Final Round) ======
    if client2_confusion_matrix_history:
        cm = client2_confusion_matrix_history[-1]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Client2 Confusion Matrix (Final Round)', fontsize=14)
        plt.tight_layout()
        plt.savefig("plots/client2_confusion_matrix.png", dpi=150)
        plt.close()
        print("Saved: plots/client2_confusion_matrix.png")

    print("\nClient2 Graphs Saved Successfully!")


# =====================================================
# START FLOWER CLIENT
# =====================================================

try:

    fl.client.start_numpy_client(

        server_address=
        "127.0.0.1:8080",

        client=
        FlowerClient()

    )

finally:

    save_client_metrics()

    save_client_plots()
