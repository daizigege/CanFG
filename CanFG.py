import functools
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from premodels.irse import Backbone
import lpips
MAX_DIM = 64 * 16  # 1024
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(85)
torch.manual_seed(85)

# todo 种子设置

def random_orthogonal_matrix(n):
    # 生成随机矩阵
    random_matrix = torch.randn(n, n)
    # 计算 QR 分解
    q, _ = torch.qr(random_matrix)
    return q
# 生成一个512维度的随机正交矩阵
matrix_512d = random_orthogonal_matrix(512).cuda()
print(matrix_512d)







class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                  shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2 ** enc_layers  # f_size = 4 for 128x128

        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

        layers = []
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2 ** (dec_layers - i - 1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                n_in = n_in + n_in // 2 if self.shortcut_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs):
        # a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        # z = torch.cat([zs[-1], a_tile], dim=1)
        z=zs[-1]
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            # if self.inject_layers > i:
            #     a_tile = a.view(a.size(0), -1, 1, 1) \
            #         .repeat(1, 1, self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1))
            #     z = torch.cat([z, a_tile], dim=1)
        return z

    def forward(self, x, mode='enc-dec'):
        if mode == 'enc-dec':
            return self.decode(self.encode(x))

class Discriminators(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminators, self).__init__()
        self.f_size = img_size // 2 ** n_layers

        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h)



class CanFG(nn.Module):
    def __init__(self, args):
        super(CanFG, self).__init__()
        self.id=True
        self.rec=True
        self.mode = args.mode
        self.gpu = args.gpu
        self.device=  torch.device('cuda')if self.gpu else  torch.device('cpu')
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp
        self.lambda_id=args.lambda_id
        self.lambda_em=args.lambda_em
        self.lambda_lp=args.lambda_lp

        if self.lambda_lp>0:
            self.LPIPS = lpips.LPIPS(net='vgg').to(self.device)

        #生成器
        self.G = Generator()
        self.G.train()
        if self.gpu: self.G.cuda()


        #提取虚拟身份的网络
        #todo 也要修改模型结构
        #
        self.EM =Backbone(50, 0.6, 'ir_se')

        self.EM.train()
        if self.gpu: self.EM.cuda()

        #鉴别器
        self.D = Discriminators()
        self.D.train()
        if self.gpu: self.D.cuda()

        #预训练的人脸匿名器
        self.Ano = Generator()
        self.Ano.eval()
        self.Ano.load_state_dict(torch.load('premodels/anonymized_rec100_id_10_em_0_lp_0_G.pt'), strict=False)
        if self.gpu: self.Ano.cuda()

        #预训练的身份提取器
        self.FR = Backbone(50, 0.6, 'ir_se')
        self.FR.eval()
        self.FR.load_state_dict(torch.load('premodels/model_ir_se50.pth'),strict=False)
        if self.gpu: self.FR.cuda()

        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_EM = optim.Adam(self.EM.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=args.betas)



    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_EM.param_groups:
            g['lr'] = lr
        for g in self.optim_D.param_groups:
            g['lr'] = lr


    def trainG(self, img_a):
        for p in self.D.parameters():
            p.requires_grad = False
        img_fake = self.G(img_a)#保护人脸

        img_ano = self.Ano (img_a).detach()  # 匿名人脸

        #1、GAN 损失
        d_fake = self.D(img_fake)
        if self.mode == 'wgan':
            gf_loss = -d_fake.mean()
        if self.mode == 'lsgan':  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        # 2、身份提取
        with torch.no_grad():
            emb_img = self.FR(F.interpolate(img_a, (112, 112), mode='bilinear', align_corners=True))

        #3、重构损失
        if self.lambda_rec > 0:
            gr_loss = F.l1_loss(img_fake, img_ano)
        else:
            gr_loss =F.l1_loss(img_a, img_a)
        #4、虚拟身份提取损失
        if self.lambda_em > 0:
            EM_fake = self.EM(F.interpolate(img_fake, (112, 112), mode='bilinear', align_corners=True))
            EM_loss = F.mse_loss(torch.matmul(emb_img, matrix_512d),EM_fake)
        else:
            EM_loss =F.l1_loss(img_a, img_a)
        # 计算LPIPS损失
        if self.lambda_lp>0:
            lpips_loss = self.LPIPS(img_fake, img_ano).mean()
        else:
            lpips_loss = F.l1_loss(img_a, img_a)

        sum_loss = gf_loss + self.lambda_rec * gr_loss+self.lambda_em*EM_loss+self.lambda_lp*lpips_loss

        self.optim_G.zero_grad()

        self.optim_EM.zero_grad()

        sum_loss.backward()
        self.optim_G.step()
        self.optim_EM.step()
        errG = {
            'sum_loss': sum_loss.item(), 'gr_loss': gr_loss.item(),
            'id_loss': 0,'em_loss': EM_loss.item()
        }
        return errG


    def trainD(self, img_a):
        for p in self.D.parameters():
            p.requires_grad = True
        img_fake = self.G(img_a).detach()
        d_real = self.D(img_a)
        d_fake = self.D(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        if self.mode == 'wgan':
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == 'lsgan':  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + \
                      F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == 'dcgan':  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)

        d_loss = df_loss + self.lambda_gp * df_gp

        self.optim_D.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        errD = {
            'd_loss': d_loss.item(), 'df_loss': 0,
            'df_gp': 0, 'dc_loss': 0
        }
        return errD



    def train(self):
        self.G.train()
        self.D.train()
        self.EM.train()

    def eval(self):
        self.G.eval()
        self.EM.eval()
        self.D.eval()

    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'EM': self.EM.state_dict(),
            'D': self.D.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_EM': self.optim_EM.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        # if 'G' in states:
        self.G.load_state_dict(states['G'])
        # if 'EM' in states:
        try:
            # 可能会引发错误的代码
            self.EM.load_state_dict(states['EM'])
        except Exception as e:
            # 如果发生 ZeroDivisionError 异常，这里的代码会执行
            print("虚拟身份提取器的模型参数不对")


        # if 'D' in states:
        self.D.load_state_dict(states['D'])
        # if 'optim_G' in states:
        self.optim_G.load_state_dict(states['optim_G'])
        # if 'optim_D' in states:
        self.optim_D.load_state_dict(states['optim_D'])
        # if 'optim_EM' in states:
        try:
            # 可能会引发错误的代码
            self.optim_EM.load_state_dict(states['optim_EM'])
        except Exception as e:
            print("虚拟身份提取器的模型参数不对")


    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


