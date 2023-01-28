import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)

def gamma(x, clip=True, gamma=2.2):
    if clip: # prevent numerical instability
        x = x.clamp_min(1e-6)
    return x ** (1/gamma)

def norm(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    return gradient_orig


def Pyramid_Sample(img, max_scale=8):
    imgs = []
    sample = img
    power = 1
    while 2**power <= max_scale:
        sample = nn.AvgPool2d(2,2)(sample)
        imgs.append(sample)
        power += 1
    return imgs


def Pyramid_Loss(lows, highs, loss_fn=F.l1_loss, rate=1., norm=True):
    losses = []
    for low, high in zip(lows, highs):
        losses.append( loss_fn(low, high) )
    pyramid_loss = 0
    scale = 0
    lam = 1
    for i, loss in enumerate(losses):
        pyramid_loss += loss * lam
        scale += lam
        lam = lam * rate
    if norm:
        pyramid_loss = pyramid_loss / scale
    return pyramid_loss

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, low, high):
        diff = low - high
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class Unet_Loss(nn.Module):
    def __init__(self, charbonnier=False):
        super().__init__()
        self.l1_loss = L1_Charbonnier_loss() if charbonnier else F.l1_loss

    def grad_loss(self, low, high):
        grad_x = torch.abs(gradient(low, 'x') - gradient(high, 'x'))
        grad_y = torch.abs(gradient(low, 'y') - gradient(high, 'y'))
        grad_norm = torch.mean(grad_x + grad_y)
        return grad_norm
    
    def pyramid_loss(self, low, high):
        h2, h4, h8 = Pyramid_Sample(high, max_scale=8)
        l2, l4, l8 = Pyramid_Sample(low, max_scale=8)
        loss = Pyramid_Loss([low, l2, l4, l8], [high, h2, h4, h8], loss_fn=self.loss, rate=0.5, norm=True)
        return loss

    def loss(self, low, high):
        # loss_grad = self.grad_loss(low, high)
        loss_recon = self.l1_loss(low, high)
        # loss_recon += self.l1_loss(gamma(low), gamma(high))
        # loss_recon /= 2
        return loss_recon# + loss_grad

    def forward(self, low, high, pyramid=False):
        if pyramid:
            loss = self.pyramid_loss(low, high)
        else:
            loss = self.loss(low, high)
        # loss_recon = F.l1_loss(low, high)
        # loss_grad = self.grad_loss(low, high)
        # loss_enhance = self.mutual_consistency(low, high, hook)
        return loss

class Unet_dpsv_Loss(Unet_Loss):
    def __init__(self, charbonnier=False):
        super().__init__()
        self.l1_loss = L1_Charbonnier_loss() if charbonnier else F.l1_loss

    def forward(self, output, target):
        scale = 2 ** (len(output) - 1)
        target = [target,] + Pyramid_Sample(target, max_scale=scale)
        # loss_restore = self.loss(output, target)
        loss_restore = Pyramid_Loss(output, target,
                                    loss_fn=self.loss, rate=1, norm=False)
        return loss_restore

class Unet_dpsv_Loss_up(Unet_Loss):
    def __init__(self, charbonnier=False):
        super().__init__()
        self.l1_loss = L1_Charbonnier_loss() if charbonnier else F.l1_loss

    def forward(self, output, target):
        scale = 2 ** (len(output) - 2)
        target = [target, target,] + Pyramid_Sample(target, max_scale=scale)
        # loss_restore = self.loss(output, target)
        loss_restore = Pyramid_Loss(output, target,
                                    loss_fn=self.loss, rate=1, norm=False)
        return loss_restore

def PSNR_Loss(low, high):
    shape = low.shape
    if len(shape) <= 3:
        psnr = -10.0 * torch.log(torch.mean(torch.pow(high-low, 2))) / torch.log(torch.as_tensor(10.0))
    else:
        psnr = torch.zeros(shape[0])
        for i in range(shape[0]):
            psnr[i]=-10.0 * torch.log(torch.mean(torch.pow(high[i]-low[i], 2))) / torch.log(torch.as_tensor(10.0))
        # print(psnr)
        psnr = torch.mean(psnr)# / shape[0]
    return psnr 

class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)


class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(DEVICE)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(DEVICE)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(DEVICE)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class GAN_Loss(nn.Module):
    def __init__(self, mode='RaSGAN'):
        super().__init__()
        self.gan_mode = mode
    
    def forward(self, D_real, D_fake, D_fake_for_G):
        y_ones = torch.ones_like(D_real)
        y_zeros = torch.zeros_like(D_fake)

        if self.gan_mode == 'RSGAN':
            ### Relativistic Standard GAN
            BCE_stable = torch.nn.BCEWithLogitsLoss()
            # Discriminator loss
            errD = BCE_stable(D_real - D_fake, y_ones)
            loss_D = torch.mean(errD)
            # Generator loss
            errG = BCE_stable(D_fake_for_G - D_real, y_ones)
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'SGAN':
            criterion = torch.nn.BCEWithLogitsLoss()
            # Real data Discriminator loss
            errD_real = criterion(D_real, y_ones)
            # Fake data Discriminator loss
            errD_fake = criterion(D_fake, y_zeros)
            loss_D = torch.mean(errD_real + errD_fake) / 2
            # Generator loss
            errG = criterion(D_fake_for_G, y_ones)
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'RaSGAN':
            BCE_stable = torch.nn.BCEWithLogitsLoss()
            # Discriminator loss
            errD = (BCE_stable(D_real - torch.mean(D_fake), y_ones) + 
                    BCE_stable(D_fake - torch.mean(D_real), y_zeros))/2
            loss_D = torch.mean(errD)
            # Generator loss
            errG = (BCE_stable(D_real - torch.mean(D_fake_for_G), y_zeros) + 
                    BCE_stable(D_fake_for_G - torch.mean(D_real), y_ones))/2
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'RaLSGAN':
            # Discriminator loss
            errD = (torch.mean((D_real - torch.mean(D_fake) - y_ones) ** 2) + 
                    torch.mean((D_fake - torch.mean(D_real) + y_ones) ** 2))/2
            loss_D = errD
            # Generator loss (You may want to resample again from real and fake data)
            errG = (torch.mean((D_real - torch.mean(D_fake_for_G) + y_ones) ** 2) + 
                    torch.mean((D_fake_for_G - torch.mean(D_real) - y_ones) ** 2))/2
            loss_G = errG
        
        return loss_D, loss_G