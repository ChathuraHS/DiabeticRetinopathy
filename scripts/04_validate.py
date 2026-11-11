from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT = Path(r"D:\DiabeticRetinopathy")


REPORT_DIR = PROJECT / "reports"
SPLIT_DIR = PROJECT / "data" / "splits"

metadata = pd.read_csv(
    REPORT_DIR / "processed_metadata.csv"
)

failures = pd.read_csv(
    REPORT_DIR / "preprocessing_failures.csv"
)

# 1. Check duplicate image IDs.
assert metadata["image"].is_unique, "Duplicate image IDs found"

# 2. Check valid labels.
assert metadata["level"].isin([0, 1, 2, 3, 4]).all()

# 3. Check valid split names.
assert metadata["split"].isin(
    ["train", "validation", "test"]
).all()

# 4. Check that every processed image exists and is valid.
for path in metadata["processed_path"]:

    path = Path(path)

    assert path.exists(), f"Missing image: {path}"

    with Image.open(path) as image:
        assert image.size == (512, 512)
        assert image.mode == "RGB"
        image.verify()

# 5. Check patient isolation.
train = set(
    metadata.loc[metadata["split"] == "train", "patient_id"]
)

validation = set(
    metadata.loc[
        metadata["split"] == "validation",
        "patient_id"
    ]
)

test = set(
    metadata.loc[metadata["split"] == "test", "patient_id"]
)

assert train.isdisjoint(validation)
assert train.isdisjoint(test)
assert validation.isdisjoint(test)

# 6. Check that every split CSV row is accounted for.
for split_name in ["train", "validation", "test"]:

    original_split = pd.read_csv(
        SPLIT_DIR / f"{split_name}.csv"
    )

    processed_split = metadata[
        metadata["split"] == split_name
    ]

    expected_ids = set(original_split["image"])
    successful_ids = set(processed_split["image"])

    failed_ids = set(
        failures.loc[
            failures["split"] == split_name,
            "image"
        ]
    )

    assert expected_ids == successful_ids.union(failed_ids), (
        f"Unaccounted images in {split_name}"
    )

    assert successful_ids.isdisjoint(failed_ids)

# 7. Display summary.
print("\nDATASET VALIDATION COMPLETE")

print("\nImages by split:")
print(metadata["split"].value_counts())

print("\nClass distribution:")
print(
    metadata.groupby(["split", "level"])
    .size()
    .unstack(fill_value=0)
)

original_size = metadata["source_size_bytes"].sum()
processed_size = metadata["processed_size_bytes"].sum()

print("\nStorage comparison:")
print(f"Original: {original_size / (1024**3):.2f} GB")
print(f"Processed: {processed_size / (1024**3):.2f} GB")

if original_size > 0:
    reduction = (1 - processed_size / original_size) * 100
    print(f"Size reduction: {reduction:.2f}%")

print("\nSuccessful images:", len(metadata))
print("Preprocessing failures:", len(failures))