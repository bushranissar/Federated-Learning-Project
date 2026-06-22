# =====================================================
# svm_model.py
# Federated Linear SVM
# CNN Features -> SVM Classifier
# =====================================================

import numpy as np

from sklearn.linear_model import SGDClassifier


class FederatedSVM:

    def __init__(self):

        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate="optimal",
            random_state=42
        )

        self.initialized = False

    # =================================================
    # INITIAL TRAINING
    # =================================================

    def initialize(self, X, y):

        if not self.initialized:

            self.model.partial_fit(
                X,
                y,
                classes=np.array([0, 1, 2, 3, 4])
            )

            self.initialized = True

    # =================================================
    # TRAIN
    # =================================================

    def train(self, X, y):

        if not self.initialized:

            self.initialize(X, y)

        else:

            self.model.partial_fit(
                X,
                y
            )

    # =================================================
    # PREDICTION
    # =================================================

    def predict(self, X):

        return self.model.predict(X)
    
    def predict_proba(self, X):
      
        return self.model.predict_proba(X)

    # =================================================
    # DECISION SCORES
    # =================================================

    def decision_function(self, X):

        return self.model.decision_function(X)

    # =================================================
    # GET PARAMETERS
    # =================================================

    def get_parameters(self):
        if not self.initialized:
            return[
                np.zeros((5, 128)),
                np.zeros(5)
            ]

        return [

            self.model.coef_,

            self.model.intercept_

        ]

    # =================================================
    # SET PARAMETERS
    # =================================================

    def set_parameters(self, parameters):

        coef, intercept = parameters

        self.model.coef_ = coef

        self.model.intercept_ = intercept

        self.model.classes_ = np.array(
            [0, 1, 2, 3, 4]
        )

        self.initialized = True