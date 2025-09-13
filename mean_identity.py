
# encoding: utf-8
import argparse
import datetime

from data5 import CelebA_HQ
import torch.utils.data as data

import torch
import torchvision.utils as vutils
from model5 import AttGAN

from helpers import Progressbar, add_scalar_dict
import os

os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import random
import numpy as np


import torch.nn.functional as F

train_dataset = CelebA_HQ('/data_disk/wangtao/mymodel/data/img_align_celeba/', 128, 'train')

train_dataloader = data.DataLoader(
    train_dataset, batch_size=128,
    shuffle=False, drop_last=True
)
from premodels.irse import  Backbone
FR = Backbone(50, 0.6, 'ir_se')
FR.eval()
FR.load_state_dict(torch.load('premodels/model_ir_se50.pth'), strict=False)
FR=FR.cuda()


progressbar = Progressbar()


it = 0
it_per_epoch = len(train_dataset) // 128

with torch.no_grad():
    sum=torch.zeros(512).cuda()
    number=0
    for img_a in progressbar(train_dataloader):
        img_a = img_a.cuda()
        emb_img = FR(F.interpolate(img_a, (112, 112), mode='bilinear', align_corners=True)).mean(0)
        sum=sum+emb_img
        number=number+1
    mean_identity=sum/number
    torch.save(mean_identity, 'mean_identity.pt')
    # todo
    # modelpath='pts/id_' + str(args.lambda_id) + '_L_' + str(args.bothL) + '_ad_' + str(
    #     args.add2) + '_faces_' + args.facemodels + '_thd_' + str(args.thread) + '_'
    # if epoch == 92:
    #     attgan.save(modelpath + str(epoch) + '.pt')
    # if epoch == 94:
    #     attgan.save(modelpath + str(epoch) + '.pt')
    # if epoch==args.epochs-1 :
    #     attgan.save(modelpath + str(epoch) + '.pt')



