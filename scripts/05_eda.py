from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT = Path(r"D:\DiabeticRetinopathy")

REPORT_DIR = PROJECT / "reports"

df = pd.read_csv(
    REPORT_DIR / "processed_metadata.csv"
)

# Class distribution.
class_counts = df["level"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")

plt.title("Diabetic Retinopathy Class Distribution")
plt.xlabel("Severity Grade")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "class_distribution.png",
    dpi=300
)
plt.close()

# Image size comparison.
df["original_size_kb"] = (
    df["source_size_bytes"] / 1024
)

df["processed_size_kb"] = (
    df["processed_size_bytes"] / 1024
)

plt.figure(figsize=(8, 5))

plt.hist(
    df["original_size_kb"],
    bins=40,
    alpha=0.6,
    label="Original"
)

plt.hist(
    df["processed_size_kb"],
    bins=40,
    alpha=0.6,
    label="Processed"
)

plt.title("Original vs Processed Image File Sizes")
plt.xlabel("File size (KB)")
plt.ylabel("Number of Images")
plt.legend()

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "image_size_comparison.png",
    dpi=300
)
plt.close()

print("EDA reports generated.")