from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image


class CollageMNIST(Dataset):
    def __init__(
        self,
        img_dir="/home/metod/datasets/collage_mnist/",
        transform=None,
        target_transform=None,
        train=True,
        download=False,
        root=False,
    ):
        """
        root and download args are there to ensure compatibility with get_image_dataset function in image.py
        """
        self.train = train
        if self.train:
            img_dir = os.path.join(img_dir, "train")
        else:
            img_dir = os.path.join(img_dir, "test")
        self.img_labels = pd.read_csv(os.path.join(img_dir, "labels.csv"))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('L')  # images are converted to grayscale
        image = np.array(image) / 255.
        label = self.img_labels.iloc[idx, 1]
        label_type = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.train:
            return image, label
        else:
            return image, (label, label_type)
