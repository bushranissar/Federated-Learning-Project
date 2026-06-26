# =====================================================
# server.py
# CNN + SVM Ensemble Federated Learning
# Flower Server
# =====================================================

from flwr.server import start_server
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, EvaluateIns, Scalar, EvaluateRes
from typing import Optional, Tuple, List, Dict, Union
import matplotlib.pyplot as plt

# =====================================================
# GLOBAL METRICS TRACKING
# =====================================================

server_accuracy_history = []
server_loss_history = []
server_precision_history = []
server_recall_history = []
server_f1_history = []
server_confusion_matrix_history = []

# =====================================================
# GLOBAL METRIC AGGREGATION
# =====================================================

def weighted_average(metrics):

    total_examples = sum(num_examples for num_examples, _ in metrics)

    accuracy = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ) / total_examples

    precision = sum(
        num_examples * m["precision"]
        for num_examples, m in metrics
    ) / total_examples

    recall = sum(
        num_examples * m["recall"]
        for num_examples, m in metrics
    ) / total_examples

    f1 = sum(
        num_examples * m["f1"]
        for num_examples, m in metrics
    ) / total_examples

    print("\n" + "=" * 60)
    print("GLOBAL MODEL METRICS")
    print("=" * 60)
    print(f"Global Accuracy  : {accuracy:.4f}")
    print(f"Global Precision : {precision:.4f}")
    print(f"Global Recall    : {recall:.4f}")
    print(f"Global F1 Score  : {f1:.4f}")
    print("=" * 60)

    server_accuracy_history.append(accuracy)
    server_precision_history.append(precision)
    server_recall_history.append(recall)
    server_f1_history.append(f1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# =====================================================
# CUSTOM STRATEGY
# =====================================================

class FedAvgWithMetrics(FedAvg):

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        result = super().evaluate(server_round, parameters)

        if result is not None:
            loss, metrics_dict = result
            server_loss_history.append(loss)
            print(f"\nRound {server_round} - Global Loss: {loss:.4f}")

        return result

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if loss_aggregated is not None:
            server_loss_history.append(loss_aggregated)
            print(f"\nRound {server_round} - Aggregated Loss: {loss_aggregated:.4f}")

        return loss_aggregated, metrics_aggregated

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ):
        return super().configure_evaluate(
            server_round,
            parameters,
            client_manager
        )

strategy = FedAvgWithMetrics(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average
)

print("\n" + "=" * 60)
print("CNN + SVM ENSEMBLE FEDERATED LEARNING")
print("FLOWER SERVER")
print("FedAvg Aggregation")
print("Clients Required : 2")
print("Rounds           : 20")
print("=" * 60)

start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=ServerConfig(num_rounds=20)
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

CLASS_NAMES = [
    "healthy",
    "bacterial spot",
    "late blight",
    "septoria leaf spot",
    "yellow leaf"
]

def save_server_plots():

    import seaborn as sns

    if not server_accuracy_history:
        return

    rounds = list(range(1, len(server_accuracy_history) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, server_accuracy_history, marker='o')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Global Accuracy vs Round")
    plt.grid(True)
    plt.savefig("server_accuracy_vs_round.png", dpi=150)
    plt.close()

    if server_loss_history:
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, server_loss_history, marker='s')
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Global Loss vs Round")
        plt.grid(True)
        plt.savefig("server_loss_vs_round.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, server_accuracy_history, marker='o', label='Accuracy')
        plt.plot(rounds, server_loss_history, marker='s', label='Loss')
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title("Accuracy and Loss vs Round")
        plt.legend()
        plt.grid(True)
        plt.savefig("server_accuracy_loss.png", dpi=150)
        plt.close()

def save_server_metrics():

    import pandas as pd

    if not server_accuracy_history:
        return

    max_len = len(server_accuracy_history)

    def pad(arr):
        return arr + [None] * (max_len - len(arr))

    df = pd.DataFrame({
        "Round": list(range(1, max_len + 1)),
        "Accuracy": pad(server_accuracy_history),
        "Loss": pad(server_loss_history),
        "Precision": pad(server_precision_history),
        "Recall": pad(server_recall_history),
        "F1_Score": pad(server_f1_history),
    })

    df.to_csv("server_metrics.csv", index=False)
    print("Saved: server_metrics.csv")

save_server_plots()
save_server_metrics()
