import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Pretrained ImageNet models expect tensors with
# ImageNet channel-wise normalisation.

TRAIN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class RetinopathyDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["processed_path"]
        ).convert("RGB")

        label = int(row["level"])

        if self.transform:
            image = self.transform(image)

        return image, label
