# =====================================================
# server.py
# CNN + SVM Ensemble Federated Learning
# Flower Server
# =====================================================

from flwr.server import start_server

from flwr.server import ServerConfig

from flwr.server.strategy import FedAvg

# =====================================================
# GLOBAL METRIC AGGREGATION
# =====================================================

def weighted_average(metrics):

    total_examples = sum(

        num_examples

        for num_examples, _

        in metrics

    )

    accuracy = sum(

        num_examples
        *
        m["accuracy"]

        for num_examples, m

        in metrics

    ) / total_examples

    precision = sum(

        num_examples
        *
        m["precision"]

        for num_examples, m

        in metrics

    ) / total_examples

    recall = sum(

        num_examples
        *
        m["recall"]

        for num_examples, m

        in metrics

    ) / total_examples

    f1 = sum(

        num_examples
        *
        m["f1"]

        for num_examples, m

        in metrics

    ) / total_examples

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1

    }

# =====================================================
# FEDAVG STRATEGY
# =====================================================

strategy = FedAvg(

    fraction_fit=1.0,

    fraction_evaluate=2.0,

    min_fit_clients=2,

    min_evaluate_clients=2,

    min_available_clients=2,

    evaluate_metrics_aggregation_fn=

        weighted_average

)

# =====================================================
# START SERVER
# =====================================================

print(

    "\n"

    "==================================="

)

print(

    "Starting Flower Server"

)

print(

    "CNN + SVM Ensemble"

)

print(

    "FedAvg Aggregation"

)

print(

    "==================================="

)

start_server(

    server_address=

        "0.0.0.0:8080",

    strategy=

        strategy,

    config=

        ServerConfig(

            num_rounds=20

        )

)