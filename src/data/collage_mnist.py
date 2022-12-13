from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np


class CollageMNIST(Dataset):
    def __init__(
        self,
        img_dir="/home/metod/datasets/mnist_collage/",
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
            img_dir = os.path.join(img_dir, "train-all")
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
        image = np.load(img_path)
        digit_label = self.img_labels.iloc[idx, 1]
        color_label = self.img_labels.iloc[idx, 2]
        color_label_random = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.train:
            return image, digit_label
        else:
            return image, (digit_label, color_label, color_label_random)
