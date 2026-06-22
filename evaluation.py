# =====================================================
# evaluation.py
# CNN + SVM Ensemble Evaluation
# =====================================================

import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

    classification_report,

    log_loss

)


# =====================================================
# EVALUATE MODEL
# =====================================================

def evaluate_model(

        y_true,

        y_pred,

        y_prob=None

):

    results = {}

    # ================================================
    # ACCURACY
    # ================================================

    results["accuracy"] = accuracy_score(

        y_true,

        y_pred

    )

    # ================================================
    # PRECISION
    # ================================================

    results["precision"] = precision_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    # ================================================
    # RECALL
    # ================================================

    results["recall"] = recall_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    # ================================================
    # F1 SCORE
    # ================================================

    results["f1_score"] = f1_score(

        y_true,

        y_pred,

        average="weighted",

        zero_division=0

    )

    # ================================================
    # LOSS
    # ================================================

    if y_prob is not None:

        try:

            results["loss"] = log_loss(

                y_true,

                y_prob

            )

        except:

            results["loss"] = 0.0

    else:

        results["loss"] = 0.0

    # ================================================
    # CONFUSION MATRIX
    # ================================================

    results["confusion_matrix"] = confusion_matrix(

        y_true,

        y_pred

    )

    # ================================================
    # CLASSIFICATION REPORT
    # ================================================

    results["classification_report"] = (

        classification_report(

            y_true,

            y_pred,

            digits=4,

            zero_division=0

        )

    )

    return results


# =====================================================
# PRINT RESULTS
# =====================================================

def print_metrics(

        round_number,

        results,

        client_name="Client"

):

    print("\n")

    print("=" * 50)

    print(

        f"{client_name} - ROUND {round_number}"

    )

    print("=" * 50)

    print(

        f"Accuracy  : "

        f"{results['accuracy']:.4f}"

    )

    print(

        f"Loss      : "

        f"{results['loss']:.4f}"

    )

    print(

        f"Precision : "

        f"{results['precision']:.4f}"

    )

    print(

        f"Recall    : "

        f"{results['recall']:.4f}"

    )

    print(

        f"F1 Score  : "

        f"{results['f1_score']:.4f}"

    )

    print("\nConfusion Matrix:\n")

    print(

        results["confusion_matrix"]

    )

    print("\nClassification Report:\n")

    print(

        results["classification_report"]

    )

    print("\n")