
# encoding: utf-8
import argparse
from data import CelebA
import torch.utils.data as data

import torch
import torchvision.utils as vutils
# todo
from CanFG import CanFG

import os

os.environ['CUDA_VISIBLE_DEVICES'] ='0'


class Progressbar():
    def __init__(self):
        self.p = None
    def __call__(self, iterable):
        from tqdm import tqdm
        self.p = tqdm(iterable)
        return self.p
    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)



def parse(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', default='/data_disk/wangtao/mymodel/data/img_align_celeba/', type=str)
    parser.add_argument('--data_path', default='/media/HDD1/wangtao/datatset/img_align_celeba/img_128_align/', type=str)

    parser.add_argument('--lambda_rec', type=float, default=100)
    parser.add_argument('--lambda_gp',  type=float, default=10.0)
    parser.add_argument('--lambda_id', type=float, default=0)
    parser.add_argument('--lambda_em', type=float, default=500)
    parser.add_argument('--lambda_lp', type=float, default=10)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=125, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)# todo
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=4, help='# of d updates per g update')

    parser.add_argument('--b_distribution', dest='b_distribution', default='none',
                        choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=32 , help='# of sample images')

    parser.add_argument('--gpu', dest='gpu', action='store_true',default=True)

    return parser.parse_args(args)

# seed_torch()
args = parse()
print(args)

args.lr_base = args.lr
args.betas = (args.beta1, args.beta2)

train_dataset = CelebA(args.data_path, 'train')
valid_dataset = CelebA(args.data_path, 'valid')
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=False, drop_last=True

)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

CanFG = CanFG(args)
# todo 预训练模型
CanFG.load('premodels/anonymized_rec100_id_10_em_0_lp_0.pt')


progressbar = Progressbar()


it = 0
it_per_epoch = len(train_dataset) // args.batch_size

for epoch in range(75,args.epochs):
    lr = args.lr_base / (10 ** (epoch // 100))
    CanFG.set_lr(lr)

    for img_a in progressbar(train_dataloader):
        CanFG.train()
        img_a = img_a.cuda() if args.gpu else img_a
        #
        if (it + 1) % (args.n_d + 1) != 0:
            errD = CanFG.trainD(img_a)
        else:
            errG = CanFG.trainG(img_a)
            progressbar.say(epoch=epoch, iter=it + 1, d_loss=errD['d_loss'], g_loss=errG['sum_loss'],gr_loss=errG['gr_loss'],em_loss=errG['em_loss'],id_loss=errG['id_loss'])
        it += 1

    if epoch %1==0 :
        CanFG.save('premodels/test_irse50_seed85_anonymized_'+str(args.lambda_rec)+'_id_'+str(args.lambda_id)+'_em_'+str(args.lambda_em)+'_lp_'+str(args.lambda_lp)+'.pt')

    if epoch%1==0:

        CanFG.eval()
        with torch.no_grad():
            iter=0
            for img_a in valid_dataloader:
                img_a = img_a.cuda() if args.gpu else img_a
                samples = [img_a]
                img_fake=CanFG.G(img_a)
                img_ano = CanFG.Ano(img_a)
                samples.append(img_fake)
                samples.append(img_ano)
                samples = torch.cat(samples, dim=3)
                filename='test_images/test_irse50_seed85_'+'rec'+str(args.lambda_rec)+'_id_'+str(args.lambda_id)+'_em_'+str(args.lambda_em)+'_lp_'+str(args.lambda_lp)+'_'+ str(epoch) + '.jpg'

                vutils.save_image(samples, filename, nrow=1, normalize=True, range=(-1., 1.))
                break



