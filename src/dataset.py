import cv2
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 640
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# --------------------------------------------------
# Retina-specific preprocessing
# --------------------------------------------------

def crop_retina(img, threshold=15):
    """Crop black borders/padding, pad back to a square, resize."""
    mask = img.mean(axis=2) > threshold

    if mask.sum() > 100:
        ys, xs = np.where(mask)
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    img = cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                      interpolation=cv2.INTER_AREA)


def ben_graham(img, sigma=None):
    sigma = sigma or IMAGE_SIZE / 50
    """Subtract local average colour to highlight lesions."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)

    # Grey out everything outside the circular retina.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    c = IMAGE_SIZE // 2
    cv2.circle(mask, (c, c), int(c * 0.95), 1, -1)
    img[mask == 0] = 128
    return img


def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_retina(img)
    img = ben_graham(img)
    return Image.fromarray(img)


# --------------------------------------------------
# Transforms (applied after Ben Graham, so no ColorJitter)
# --------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(
        degrees=180,
        scale=(0.9, 1.1),
        fill=(128, 128, 128)
    ),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class RetinopathyDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.paths = self.df["processed_path"].tolist()
        self.labels = self.df["level"].astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = load_image(self.paths[index])

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]