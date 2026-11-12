import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna

from torch.utils.data import DataLoader
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from dataset import (
    RetinopathyDataset,
    TRAIN_TRANSFORM,
    TEST_TRANSFORM
)


PROJECT = Path(r"D:\DiabeticRetinopathy")

TRAIN_CSV = PROJECT / "data/splits/train.csv"
VAL_CSV = PROJECT / "data/splits/validation.csv"

MODEL_DIR = PROJECT / "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 5

N_TRIALS = 5
EPOCHS_PER_TRIAL = 5
FINAL_EPOCHS = 10


# --------------------------------------------------
# 1. Calculate class weights
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=train_df["level"].values
)

CLASS_WEIGHTS = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


# --------------------------------------------------
# 2. Load model
# --------------------------------------------------

def create_model(model_name, dropout):

    if model_name == "resnet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, NUM_CLASSES)
        )

    elif model_name == "efficientnet_b0":

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(features, NUM_CLASSES)
        )

    else:
        raise ValueError("Unknown model")

    return model.to(DEVICE)


# --------------------------------------------------
# 3. Data loaders
# --------------------------------------------------

def create_loaders(batch_size):

    train_dataset = RetinopathyDataset(
        TRAIN_CSV,
        TRAIN_TRANSFORM
    )

    val_dataset = RetinopathyDataset(
        VAL_CSV,
        TEST_TRANSFORM
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# --------------------------------------------------
# 4. Evaluation on validation data
# --------------------------------------------------

def validate(model, loader):

    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    return cohen_kappa_score(
        actual,
        predicted,
        weights="quadratic"
    )


# --------------------------------------------------
# 5. Train one configuration
# --------------------------------------------------

def train_model(
    model_name,
    learning_rate,
    weight_decay,
    dropout,
    batch_size,
    epochs,
    save_path=None
):

    train_loader, val_loader = create_loaders(
        batch_size
    )

    model = create_model(
        model_name,
        dropout
    )

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    best_qwk = -1.0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for images, labels in progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda"
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            progress.set_postfix(
                loss=loss.item()
            )

        average_loss = running_loss / len(train_loader)

        qwk = validate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss={average_loss:.4f}, "
            f"Validation QWK={qwk:.4f}"
        )

        if qwk > best_qwk:

            best_qwk = qwk

            if save_path:

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "qwk": best_qwk,
                        "model_name": model_name,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "epoch": epoch + 1
                    },
                    save_path
                )

    return best_qwk


# --------------------------------------------------
# 6. Hyperparameter optimisation
# --------------------------------------------------

def objective(trial, model_name):

    learning_rate = trial.suggest_float(
        "learning_rate",
        0.00001,
        0.001,
        log=True
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        0.00001,
        0.001,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.2,
        0.5
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [2, 4]
    )

    qwk = train_model(
        model_name=model_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        batch_size=batch_size,
        epochs=EPOCHS_PER_TRIAL
    )

    return qwk


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "efficientnet_b0"]
    )

    args = parser.parse_args()

    model_name = args.model

    output_dir = MODEL_DIR / model_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Device:", DEVICE)
    print("Training:", model_name)

    # Hyperparameter search

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_name
        ),
        n_trials=N_TRIALS
    )

    print("\nBest hyperparameters:")
    print(study.best_params)

    print("Best validation QWK:", study.best_value)

    study.trials_dataframe().to_csv(
        output_dir / "optimization_results.csv",
        index=False
    )

    # Final training using best hyperparameters

    best = study.best_params

    best_model_path = output_dir / "best_model.pth"

    print("\nTraining final model...")

    train_model(
        model_name=model_name,
        learning_rate=best["learning_rate"],
        weight_decay=best["weight_decay"],
        dropout=best["dropout"],
        batch_size=best["batch_size"],
        epochs=FINAL_EPOCHS,
        save_path=best_model_path
    )

    print("\nTraining completed.")
    print("Model saved:", best_model_path)


if __name__ == "__main__":
    main()