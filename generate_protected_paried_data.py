
import argparse
import datetime
from test_data import CelebA_test
import torch.utils.data as data

import torch
import torchvision.utils as vutils
#todo
from CanFG import CanFG

import os

os.environ['CUDA_VISIBLE_DEVICES'] ='0'


def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp',  type=float, default=10.0)
    parser.add_argument('--lambda_id', type=float, default=1)
    parser.add_argument('--lambda_em', type=float, default=10)
    parser.add_argument('--lambda_lp', type=float, default=1)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=140, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)# todo
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=4, help='# of d updates per g update')

    parser.add_argument('--b_distribution', dest='b_distribution', default='none',
                        choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=1 , help='# of sample images')

    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true',default=True)
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    return parser.parse_args(args)

args = parse()
print(args)

args.lr_base = args.lr
args.betas = (args.beta1, args.beta2)
dataset='CelebA'
#配对的数据（包括相同身份和不同身份）#

valid_dataset = CelebA_test('/media/HDD1/wangtao/lunwen5_new/data/'+dataset+'/A_/','/media/HDD1/wangtao/lunwen5_new/data/'+dataset+'/AA_/')
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)

CanFG = CanFG(args)
CanFG.load('premodels/seed85_anonymized_100_id_0_em_500_lp_10.pt')




if not os.path.exists('data/'+dataset+'/protected_A/'):
    os.makedirs('data/'+dataset+'/protected_A/')
if not os.path.exists('data/'+dataset+'/protected_AA/'):
    os.makedirs('data/'+dataset+'/protected_AA/')

torch.save(CanFG.EM.state_dict(), 'premodels/irse50_seed85_anonymized_100_id_0_em_500_lp_10_EM.pt')
CanFG.eval()
with torch.no_grad():
    iter=0
    for img_1,img_2,name in valid_dataloader:
        img_1 = img_1.cuda() if args.gpu else img_1
        img_2 = img_2.cuda() if args.gpu else img_2
        img_1=CanFG.G(img_1)
        img_2 = CanFG.G(img_2)
        names=name[0][0:-3]+'png'
        # names=name[0]
        filename1='data/'+dataset+'/protected_A/'+names
        filename2 = 'data/'+dataset+'/protected_AA/'+ names
        print(filename1)

        vutils.save_image(img_1, filename1, nrow=1, normalize=True, range=(-1., 1.))
        vutils.save_image(img_2, filename2, nrow=1, normalize=True, range=(-1., 1.))



