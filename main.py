import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MNISTDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, transform=None, is_test=False):
        super(MNISTDataset, self).__init__()
        dataset = []
        for i, row in data_df.iterrows():
            data = row.to_numpy()
            if is_test:
                label = -1
                image = data.reshape(28, 28).astype(np.uint8)
            else:
                label = data[0]
                image = data[1:].reshape(28, 28).astype(np.uint8)

            if transform is not None:
                image = transform(image)

            dataset.append((image, label))

        self.dataset = dataset
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


if __name__ == "__main__":
    data_train = pd.read_csv('./data/train.csv')
    train_data = data_train.drop('label', axis=1).values
    train_mean = train_data.mean() / 255.
    train_std = train_data.std() / 255.
    eval_count = 1000
    train_trainsform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[train_mean], std=[train_std])
    ])

    test_transform = val_transform

    train_dataset = MNISTDataset(data_train, train_trainsform)
    eval_dataset = MNISTDataset(data_train.iloc[-eval_count:], val_transform)

    data_test = pd.read_csv('./data/test.csv')
    test_dataset = MNISTDataset(data_test, test_transform, is_test=True)

    # row = data_train.iloc[1].to_numpy()
    # label = row[0]
    # digit_img = row[1:].reshape(28,28)
    # plt.imshow(digit_img, interpolation='nearest', cmap='gray')
    # plt.show()