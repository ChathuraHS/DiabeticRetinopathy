from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT = Path(r"D:\DiabeticRetinopathy")

INPUT_FILE = PROJECT / "reports" / "audited_images.csv"
OUTPUT_DIR = PROJECT / "data" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_patient_id(image_id):
    # Example: 10_left -> 10
    parts = image_id.rsplit("_", 1)

    if len(parts) != 2:
        raise ValueError(f"Unexpected image ID: {image_id}")

    patient_id, eye = parts

    if eye not in ["left", "right"]:
        raise ValueError(f"Unexpected eye identifier: {image_id}")

    return patient_id


def main():
    df = pd.read_csv(INPUT_FILE)

    df["patient_id"] = df["image"].apply(extract_patient_id)

    # First split: 80% development data, 20% test data.
    outer_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    development_indices, test_indices = next(
        outer_splitter.split(
            df,
            y=df["level"],
            groups=df["patient_id"]
        )
    )

    development = df.iloc[development_indices].copy()
    test = df.iloc[test_indices].copy()

    # Second split: 80% of development for training,
    # 20% of development for validation.
    inner_splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_indices, validation_indices = next(
        inner_splitter.split(
            development,
            y=development["level"],
            groups=development["patient_id"]
        )
    )

    train = development.iloc[train_indices].copy()
    validation = development.iloc[validation_indices].copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    # Verify patient isolation.
    train_patients = set(train["patient_id"])
    validation_patients = set(validation["patient_id"])
    test_patients = set(test["patient_id"])

    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSPLIT COMPLETE")

    for name, subset in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        print(f"\n{name}: {len(subset)} images")
        print(subset["level"].value_counts().sort_index())

    print("\nPatient overlap check passed.")


if __name__ == "__main__":
    main()