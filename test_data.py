

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random


class CelebA_test(data.Dataset):
    def __init__(self, data_path1,data_path2):
        super(CelebA_test, self).__init__()
        self.data_path1 = data_path1
        self.data_path2 = data_path2
        self.images = list(set(sorted(os.listdir(self.data_path1))) & set(sorted(os.listdir(self.data_path2))))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img1 = self.tf(Image.open(os.path.join(self.data_path1 , self.images[index])))
        img2 = self.tf(Image.open(os.path.join(self.data_path2 , self.images[index])))
        return img1,img2,self.images[index]

    def __len__(self):
        return self.length








# if __name__ == '__main__':
#     import argparse
#     import torchvision.utils as vutils
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', default='/data_disk/wangtao/celeba_256/', type=str)
#     args = parser.parse_args()
#     dataset = CelebA_HQ(args.data_path, 128, 'train')
#     dataloader = data.DataLoader(
#         dataset, batch_size=64, shuffle=True, drop_last=False
#     )
#
#     for x, y in dataloader:
#         vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
#         print(y)
#         break
#     del x, y