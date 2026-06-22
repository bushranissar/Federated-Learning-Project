# =====================================================
# cnn_model.py
# CNN Feature Extractor + CNN Classifier
# 5 Classes
# =====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):

    def __init__(self, num_classes=5):

        super(CNNModel, self).__init__()

        # =============================================
        # CONVOLUTION BLOCK 1
        # =============================================

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # =============================================
        # CONVOLUTION BLOCK 2
        # =============================================

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # =============================================
        # CONVOLUTION BLOCK 3
        # =============================================

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(64)

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # =============================================
        # FEATURE LAYER
        # =============================================

        self.feature_layer = nn.Linear(
            64 * 8 * 8,
            128
        )

        # =============================================
        # DROPOUT
        # =============================================

        self.dropout = nn.Dropout(
            p=0.5
        )

        # =============================================
        # CNN CLASSIFIER
        # =============================================

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    # =================================================
    # FEATURE EXTRACTION
    # =================================================

    def extract_features(self, x):

        x = self.pool1(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool3(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        features = F.relu(
            self.feature_layer(x)
        )

        return features

    # =================================================
    # CNN CLASSIFICATION
    # =================================================

    def forward(self, x):

        features = self.extract_features(x)

        features = self.dropout(features)

        output = self.classifier(features)

        return output

    # =================================================
    # PROBABILITIES
    # =================================================

    def predict_proba(self, x):

        logits = self.forward(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        return probabilities