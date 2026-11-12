import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
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

from dataset import RetinopathyDataset, TEST_TRANSFORM
from train import (
    create_model,
    predict_raw,
    apply_thresholds,
    DEVICE,
    NUM_WORKERS,
    DEFAULT_THRESHOLDS
)


PROJECT = Path(r"D:\DiabeticRetinopathy")
TEST_CSV = PROJECT / "data/splits/test.csv"

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
LABELS = [0, 1, 2, 3, 4]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable flip test-time augmentation")
    args = parser.parse_args()

    model_name = args.model
    use_tta = not args.no_tta

    model_path = PROJECT / "models" / model_name / "best_model.pth"

    # weights_only=False: checkpoint holds numpy scalars (our own file)
    checkpoint = torch.load(
        model_path, map_location=DEVICE, weights_only=False
    )

    model = create_model(model_name, checkpoint["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    thresholds = checkpoint.get("thresholds", DEFAULT_THRESHOLDS)

    loader = DataLoader(
        RetinopathyDataset(TEST_CSV, TEST_TRANSFORM),
        batch_size=checkpoint["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )

    raw, actual = predict_raw(model, loader, tta=use_tta)
    predicted = apply_thresholds(raw, thresholds)

    results = {
        "Accuracy": accuracy_score(actual, predicted),
        "Balanced Accuracy": balanced_accuracy_score(actual, predicted),
        "Macro Precision": precision_score(
            actual, predicted, average="macro", zero_division=0),
        "Macro Recall": recall_score(
            actual, predicted, average="macro", zero_division=0),
        "Macro F1": f1_score(
            actual, predicted, average="macro", zero_division=0),
        "Quadratic Weighted Kappa": cohen_kappa_score(
            actual, predicted, weights="quadratic"),
        "QWK (default thresholds)": cohen_kappa_score(
            actual, apply_thresholds(raw, DEFAULT_THRESHOLDS),
            weights="quadratic"),
    }

    print("\nMODEL:", model_name, "| TTA:", use_tta)
    print("Thresholds:", [round(float(t), 3) for t in thresholds])

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        actual, predicted, labels=LABELS,
        target_names=CLASS_NAMES, zero_division=0
    ))

    output_dir = PROJECT / "reports"
    output_dir.mkdir(exist_ok=True)

    # Confusion matrices: counts and row-normalised (recall per class)
    cm = confusion_matrix(actual, predicted, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"{model_name} Confusion Matrix (counts)")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f"{model_name} Confusion Matrix (row-normalised)")

    for ax in axes:
        ax.set_xlabel("Predicted Grade")
        ax.set_ylabel("Actual Grade")

    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.show()

    pd.DataFrame([results]).to_csv(
        output_dir / f"{model_name}_metrics.csv", index=False
    )

    # Raw predictions for later analysis / ensembling
    pd.DataFrame({"actual": actual, "raw": raw, "predicted": predicted}).to_csv(
        output_dir / f"{model_name}_test_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()