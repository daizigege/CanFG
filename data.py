

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random


class CelebA(data.Dataset):
    def __init__(self, data_path, mode):
        super(CelebA, self).__init__()
        self.image_path = data_path
        images = sorted(os.listdir(self.image_path))

        if mode == 'train':
            self.images = images[:180000]
        if mode == 'valid':
            self.images = images[182000:185000]
        if mode == 'test':
            self.images = images[182637:]

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.image_path, self.images[index])))
        return img

    def __len__(self):
        return self.length




