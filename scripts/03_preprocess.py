from pathlib import Path
import json

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


PROJECT = Path(r"D:\DiabeticRetinopathy")

SPLIT_DIR = PROJECT / "data" / "splits"
OUTPUT_DIR = PROJECT / "data" / "processed" / "images"
REPORT_DIR = PROJECT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (1024, 1024)
JPEG_QUALITY = 95

SPLITS = ["train", "validation", "test"]

from PIL import Image, ImageOps

def preprocess_image(image):
    # Correct EXIF orientation.
    image = ImageOps.exif_transpose(image)

    # Convert to RGB.
    image = image.convert("RGB")

    # Resize while preserving the entire image.
    image.thumbnail(
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )

    # Create a black 1024x1024 background.
    canvas = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))

    # Centre the resized image without cropping.
    x = (IMAGE_SIZE[0] - image.width) // 2
    y = (IMAGE_SIZE[1] - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def process_image(source_path, output_path):
    with Image.open(source_path) as original:

        # Apply orientation correction, RGB conversion,
        # Resize while preserving aspect ratio.
        processed = preprocess_image(original)

        # Save standardized JPEG.
        processed.save(
            output_path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True
        )

    # Verify the saved image.
    with Image.open(output_path) as saved:
        saved.verify()

    with Image.open(output_path) as saved:
        assert saved.size == IMAGE_SIZE
        assert saved.mode == "RGB"


def main():
    metadata = []
    failures = []

    for split_name in SPLITS:

        split_file = SPLIT_DIR / f"{split_name}.csv"
        df = pd.read_csv(split_file)

        print(f"\nProcessing {split_name}: {len(df)} images")

        for _, row in tqdm(df.iterrows(), total=len(df)):

            image_id = row["image"]
            source_path = Path(row["source_path"])

            output_path = OUTPUT_DIR / f"{image_id}.jpg"

            try:
                # Skip already completed files after checking them.
                if output_path.exists():
                    try:
                        with Image.open(output_path) as img:
                            img.verify()

                        with Image.open(output_path) as img:
                            if img.size != IMAGE_SIZE or img.mode != "RGB":
                                raise ValueError("Existing output has wrong format")

                    except Exception:
                        output_path.unlink()
                        process_image(source_path, output_path)
                else:
                    process_image(source_path, output_path)

                metadata.append({
                    "image": image_id,
                    "level": int(row["level"]),
                    "patient_id": str(row["patient_id"]),
                    "split": split_name,
                    "source_path": str(source_path),
                    "processed_path": str(output_path),
                    "original_width": int(row["original_width"]),
                    "original_height": int(row["original_height"]),
                    "processed_width": IMAGE_SIZE[0],
                    "processed_height": IMAGE_SIZE[1],
                    "source_size_bytes": int(row["source_size_bytes"]),
                    "processed_size_bytes": output_path.stat().st_size,
                    "jpeg_quality": JPEG_QUALITY
                })

            except Exception as error:
                failures.append({
                    "image": image_id,
                    "split": split_name,
                    "source_path": str(source_path),
                    "error": str(error)
                })

    metadata_columns = [
        "image",
        "level",
        "patient_id",
        "split",
        "source_path",
        "processed_path",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
        "source_size_bytes",
        "processed_size_bytes",
        "jpeg_quality"
    ]

    failure_columns = [
        "image",
        "split",
        "source_path",
        "error"
    ]

    metadata_df = pd.DataFrame(
        metadata,
        columns=metadata_columns
    )

    failures_df = pd.DataFrame(
        failures,
        columns=failure_columns
    )

    metadata_df.to_csv(
        REPORT_DIR / "processed_metadata.csv",
        index=False
    )

    failures_df.to_csv(
        REPORT_DIR / "preprocessing_failures.csv",
        index=False
    )

    config = {
        "image_size": IMAGE_SIZE,
        "color_mode": "RGB",
        "format": "JPEG",
        "jpeg_quality": JPEG_QUALITY,
        "resizing": "Aspect-ratio-preserving square padding",
        "padding_color": [0, 0, 0],
        "resampling": "Lanczos",
        "exif_orientation": True
    }

    with open(
        PROJECT / "preprocessing_config.json",
        "w"
    ) as file:
        json.dump(config, file, indent=4)

    print("\nPREPROCESSING COMPLETE")
    print("Successful:", len(metadata_df))
    print("Failed:", len(failures_df))

    if not metadata_df.empty:
        print(
            "Total processed size:",
            round(metadata_df["processed_size_bytes"].sum() / (1024 ** 3), 2),
            "GB"
        )


if __name__ == "__main__":
    main()