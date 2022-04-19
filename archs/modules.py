
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block)
    return nn.Sequential(*layers)

class Module_with_Init(nn.Module):
    def __init__(self,):
        super().__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

class ResConvBlock_CBAM(nn.Module):
    def __init__(self, in_nc, nf=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cbam = CBAM(nf)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = self.res_scale * self.cbam(self.relu(self.conv2(x))) + x
        return x + out * self.res_scale

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, nf=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class conv1x1(nn.Module):
    def __init__(self, in_nc, out_nc, is_activate=True):
        super().__init__()
        self.conv =nn.Conv2d(in_nc, out_nc, kernel_size=1, padding=0, stride=1)
        if is_activate:
            self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_c, out_c, is_activate=True):
        super().__init__()
        self.activation = nn.ReLU(inplace=True) if is_activate else nn.Sequential()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=1),
            self.activation,
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        )

        if in_c != out_c:
            self.short_cut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, stride=1)
            )
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))

    def forward(self, x):
        output = self.block(x)
        output += self.short_cut(x)
        output = self.activation(output)
        return output
class conv3x3(nn.Module):
    def __init__(self, in_nc, out_nc, stride=2, is_activate=True):
        super().__init__()
        self.conv =nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1, stride=stride)
        if is_activate:
            self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class convWithBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, is_activate=True, is_bn=True):
        super(convWithBN, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                               stride=stride, bias=False)),
        ]))
        if is_bn:
            self.conv.add_module("BN", nn.BatchNorm2d(out_c))
        if is_activate:
            self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleCvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleCvBlock, self).__init__()
        self.block = nn.Sequential(
            convWithBN(in_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False),
            convWithBN(out_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False)
        )

    def forward(self, x):
        output = self.block(x)
        return output

class nResBlocks(nn.Module):
    def __init__(self, nf, nlayers=2):
        super().__init__()
        self.blocks = make_layer(ResidualBlock(nf, nf), n_layers=nlayers)
    
    def forward(self, x):
        return self.blocks(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, is_activate=True):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            convWithBN(in_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False),
            convWithBN(out_c, out_c, kernel_size=3, padding=1, stride=1, is_activate=False, is_bn=False)
        )

        if in_c != out_c:
            self.short_cut = nn.Sequential(
                convWithBN(in_c, out_c, kernel_size=1, padding=0, stride=1, is_activate=False, is_bn=False)
            )
        else:
            self.short_cut = nn.Sequential(OrderedDict([]))
        
        self.activation = nn.LeakyReLU(0.2, inplace=False) if is_activate else nn.Sequential()

    def forward(self, x):
        output = self.block(x)
        output = self.activation(output)
        output += self.short_cut(x)
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.in_nc = in_planes
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.concat = Concat()
        self.mean = torch.mean
        self.max = torch.max

    def forward(self, x):
        avgout = self.mean(x, 1, True)
        maxout, _ = self.max(x, 1, True)
        x = self.concat([avgout, maxout], 1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out

class MaskMul(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x, mask):
        if mask.shape[1] != x.shape[1]:
            mask = torch.mean(mask, dim=1, keepdim=True)
        pooled_mask = F.avg_pool2d(mask, self.scale_factor)
        out = torch.mul(x, pooled_mask)
        return out

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels=None, up_scale=2, mode='bilinear'):
        super(UpsampleBLock, self).__init__()
        if mode == 'pixel_shuffle':
            self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
            self.up = nn.PixelShuffle(up_scale)
        elif mode=='bilinear':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.up = nn.UpsamplingBilinear2d(scale_factor=up_scale)
        else:
            print(f"Please tell me what is '{mode}' mode ????")
            raise NotImplementedError
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        x = self.relu(x)
        return x

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = 1
        self.concat = torch.cat
    
    def padding(self, tensors):
        if len(tensors) > 2: 
            return tensors
        x , y = tensors
        xb, xc, xh, xw = x.size()
        yb, yc, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, (diffX // 2, diffX - diffX//2, 
                    diffY // 2, diffY - diffY//2))
        return (x, y)

    def forward(self, x, dim=None):
        x = self.padding(x)
        return self.concat(x, dim if dim is not None else self.dim)

if __name__ == '__main__':
    from torchsummary import summary
    x = torch.randn((1,32,16,16))
    for k in range(1,3):
        # up = upsample(32, 2**k)
        # down = downsample(32//(2**k), 2**k)
        # x_up = up(x)
        # x_down = down(x_up)
        # s_up = (32,16,16)
        # summary(up,s,device='cpu')
        # summary(down,s,device='cpu')
        print(k)
