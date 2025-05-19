import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MedDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=224):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label
