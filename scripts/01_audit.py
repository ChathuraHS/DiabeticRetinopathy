from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


PROJECT = Path(r"D:\DiabeticRetinopathy")

IMAGE_DIR = PROJECT / "data" / "extracted" / "train"
LABEL_FILE = PROJECT / "data" / "raw" / "trainLabels.csv"
REPORT_DIR = PROJECT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = [".jpeg", ".jpg", ".png"]


def main():
    labels = pd.read_csv(LABEL_FILE)

    required_columns = {"image", "level"}

    if not required_columns.issubset(labels.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    labels["image"] = labels["image"].astype(str).str.strip()

    if labels["image"].duplicated().any():
        duplicates = labels[labels["image"].duplicated(keep=False)]
        duplicates.to_csv(
            REPORT_DIR / "duplicate_label_ids.csv",
            index=False
        )
        raise ValueError("Duplicate image IDs found in CSV.")

    valid_images = []
    failures = []

    for _, row in tqdm(labels.iterrows(), total=len(labels)):
        image_id = row["image"]
        label = row["level"]

        if pd.isna(label) or label not in [0, 1, 2, 3, 4]:
            failures.append({
                "image": image_id,
                "reason": "Invalid label"
            })
            continue

        image_path = None

        for extension in SUPPORTED_EXTENSIONS:
            candidate = IMAGE_DIR / f"{image_id}{extension}"

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            failures.append({
                "image": image_id,
                "reason": "Image file not found"
            })
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            valid_images.append({
                "image": image_id,
                "level": int(label),
                "source_path": str(image_path),
                "original_width": width,
                "original_height": height,
                "original_mode": mode,
                "source_size_bytes": image_path.stat().st_size
            })

        except (OSError, ValueError, UnidentifiedImageError) as error:
            failures.append({
                "image": image_id,
                "reason": str(error)
            })

    valid_df = pd.DataFrame(valid_images)
    failures_df = pd.DataFrame(failures)

    valid_df.to_csv(
        REPORT_DIR / "audited_images.csv",
        index=False
    )

    failures_df.to_csv(
        REPORT_DIR / "audit_failures.csv",
        index=False
    )

    print("\nAUDIT COMPLETE")
    print("Labelled images:", len(labels))
    print("Valid images:", len(valid_df))
    print("Failed images:", len(failures_df))

    if not valid_df.empty:
        print("\nClass distribution:")
        print(valid_df["level"].value_counts().sort_index())


if __name__ == "__main__":
    main()