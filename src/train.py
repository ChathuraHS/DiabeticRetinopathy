import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from dataset import RetinopathyDataset, TRAIN_TRANSFORM, TEST_TRANSFORM


PROJECT = Path(r"D:\DiabeticRetinopathy")

TRAIN_CSV = PROJECT / "data/splits/train.csv"
VAL_CSV = PROJECT / "data/splits/validation.csv"
MODEL_DIR = PROJECT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

N_TRIALS = 6
EPOCHS_PER_TRIAL = 4
FINAL_EPOCHS = 25
NUM_WORKERS = 4

DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]

# Batch sizes that fit a 6 GB GPU with AMP at 512px
BATCH_SIZES = {
    "resnet18": [16, 32],
    "efficientnet_b0": [8, 16],
}

torch.backends.cudnn.benchmark = True


# --------------------------------------------------
# Model: single-output regression head
# --------------------------------------------------

def create_model(model_name, dropout, pretrained=True):

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, 1)
        )

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, 1)
        )

    else:
        raise ValueError("Unknown model")

    return model.to(DEVICE)


# --------------------------------------------------
# Data
# --------------------------------------------------

def create_loaders(batch_size):

    train_ds = RetinopathyDataset(TRAIN_CSV, TRAIN_TRANSFORM)
    val_ds = RetinopathyDataset(VAL_CSV, TEST_TRANSFORM)

    # Softened class balancing: weight ~ 1/sqrt(class frequency)
    labels = np.array(train_ds.labels)
    counts = np.bincount(labels, minlength=5).astype(float)
    sample_weights = (1.0 / np.sqrt(counts))[labels]

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True
    )

    common = dict(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        persistent_workers=NUM_WORKERS > 0
    )

    train_loader = DataLoader(train_ds, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    return train_loader, val_loader


# --------------------------------------------------
# Prediction, thresholds, TTA
# --------------------------------------------------

def apply_thresholds(preds, thresholds=DEFAULT_THRESHOLDS):
    return np.digitize(preds, thresholds)


def qwk_score(labels, preds, thresholds=DEFAULT_THRESHOLDS):
    return cohen_kappa_score(
        labels, apply_thresholds(preds, thresholds), weights="quadratic"
    )


@torch.no_grad()
def predict_raw(model, loader, tta=False):
    """Return (raw regression outputs, labels). TTA = mean over 4 flips."""
    model.eval()
    preds, labels_all = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)

        views = [images]
        if tta:
            views += [images.flip(3), images.flip(2), images.flip(2).flip(3)]

        out = 0
        for v in views:
            with torch.autocast(device_type=DEVICE.type,
                                dtype=torch.float16, enabled=USE_CUDA):
                out = out + model(v).float().squeeze(1)
        out = out / len(views)

        preds.append(out.cpu().numpy())
        labels_all.append(labels.numpy())

    return np.concatenate(preds), np.concatenate(labels_all)


def optimise_thresholds(preds, labels, passes=3):
    """Coordinate-descent search of the 4 cut-points to maximise QWK."""
    thr = list(DEFAULT_THRESHOLDS)
    best = qwk_score(labels, preds, thr)

    for _ in range(passes):
        for i in range(4):
            for cand in np.linspace(thr[i] - 0.5, thr[i] + 0.5, 41):
                trial = thr.copy()
                trial[i] = cand
                if trial != sorted(trial):
                    continue
                score = qwk_score(labels, preds, trial)
                if score > best:
                    best, thr = score, trial

    return thr, best


# --------------------------------------------------
# Train one configuration
# --------------------------------------------------

def train_model(model_name, learning_rate, weight_decay, dropout,
                batch_size, epochs, save_path=None):

    train_loader, val_loader = create_loaders(batch_size)
    model = create_model(model_name, dropout)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

    best_qwk = -1.0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=DEVICE.type,
                                dtype=torch.float16, enabled=USE_CUDA):
                outputs = model(images).squeeze(1)

            loss = criterion(outputs.float(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        scheduler.step()

        preds, labels_np = predict_raw(model, val_loader)
        qwk = qwk_score(labels_np, preds)

        print(f"Epoch {epoch + 1}: Loss={running_loss / len(train_loader):.4f}, "
              f"Validation QWK={qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            if save_path:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "qwk": best_qwk,
                    "thresholds": DEFAULT_THRESHOLDS,
                    "task": "regression",
                    "model_name": model_name,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "epoch": epoch + 1
                }, save_path)

    return best_qwk


# --------------------------------------------------
# Optuna
# --------------------------------------------------

def objective(trial, model_name):

    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES[model_name])

    try:
        return train_model(model_name, lr, wd, dropout,
                           batch_size, EPOCHS_PER_TRIAL)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["resnet18", "efficientnet_b0"])
    args = parser.parse_args()

    model_name = args.model
    output_dir = MODEL_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Training:", model_name)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, model_name), n_trials=N_TRIALS)

    print("\nBest hyperparameters:", study.best_params)
    print("Best validation QWK:", study.best_value)

    study.trials_dataframe().to_csv(
        output_dir / "optimization_results.csv", index=False
    )

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

    # Tune decision thresholds on validation with TTA, store in checkpoint
    print("\nOptimising thresholds with TTA on validation set...")
    ckpt = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model = create_model(model_name, best["dropout"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, val_loader = create_loaders(best["batch_size"])
    preds, labels = predict_raw(model, val_loader, tta=True)

    before = qwk_score(labels, preds)
    thresholds, after = optimise_thresholds(preds, labels)
    print(f"Val QWK (TTA, default thresholds): {before:.4f}")
    print(f"Val QWK (TTA, tuned thresholds):   {after:.4f}")
    print("Thresholds:", [round(t, 3) for t in thresholds])

    ckpt["thresholds"] = thresholds
    ckpt["qwk_tta_tuned"] = after
    torch.save(ckpt, best_model_path)

    print("\nTraining completed.")
    print("Model saved:", best_model_path)


if __name__ == "__main__":
    main()