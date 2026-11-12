import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from dataset import (
    RetinopathyDataset,
    TEST_TRANSFORM
)


PROJECT = Path(r"D:\DiabeticRetinopathy")

TEST_CSV = PROJECT / "data/splits/test.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]


def create_model(model_name, dropout):

    if model_name == "resnet18":

        model = models.resnet18(weights=None)

        features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, 5)
        )

    else:

        model = models.efficientnet_b0(weights=None)

        features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, 5)
        )

    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "efficientnet_b0"]
    )

    args = parser.parse_args()

    model_name = args.model

    model_path = (
        PROJECT / "models" /
        model_name / "best_model.pth"
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model = create_model(
        model_name,
        checkpoint["dropout"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    dataset = RetinopathyDataset(
        TEST_CSV,
        TEST_TRANSFORM
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    actual = []
    predicted = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    # Metrics

    results = {
        "Accuracy": accuracy_score(actual, predicted),

        "Balanced Accuracy": balanced_accuracy_score(
            actual, predicted
        ),

        "Macro Precision": precision_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ),

        "Macro Recall": recall_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            actual,
            predicted,
            average="macro",
            zero_division=0
        ),

        "Quadratic Weighted Kappa": cohen_kappa_score(
            actual,
            predicted,
            weights="quadratic"
        )
    }

    print("\nMODEL:", model_name)

    for metric, value in results.items():

        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1, 2, 3, 4],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    # Confusion matrix

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1, 2, 3, 4]
    )

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted Grade")
    plt.ylabel("Actual Grade")

    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    output_dir = PROJECT / "reports"
    output_dir.mkdir(exist_ok=True)

    plt.savefig(
        output_dir / f"{model_name}_confusion_matrix.png"
    )

    plt.show()

    # Save metrics

    pd.DataFrame(
        [results]
    ).to_csv(
        output_dir / f"{model_name}_metrics.csv",
        index=False
    )


if __name__ == "__main__":
    main()