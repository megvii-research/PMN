from .modules import *
from typing import TypeVar
T = TypeVar('T', bound='Module')

class UNetSeeInDark(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = args['nframes'] // 2
        self.res = args['res']
        nframes = self.args['nframes']
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv1_1 = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*16, nf*16, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*16, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))
        
        conv10 = self.conv10_1(conv9)
        if self.res:
            out = conv10 + x
        else:
            out = conv10
        return out

class DeepUnet(nn.Module):
    def __init__(self, in_nc=4, out_nc=4, nf=32, res=False):
        super().__init__()
        self.res = res
        
        self.conv1_1 = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(nf, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(nf*4, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5_1 = nn.Conv2d(nf*8, nf*16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf*16, nf*16, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf*16, nf*8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf*4, nf*2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        
        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)

        # Deep Supervision
        self.out8 = nn.Conv2d(nf*8, out_nc, kernel_size=1)
        self.out4 = nn.Conv2d(nf*4, out_nc, kernel_size=1)
        self.out2 = nn.Conv2d(nf*2, out_nc, kernel_size=1)
    
    def forward(self, x, noise_map=None):
        shape= x.size()
        x = x.view(-1,shape[-3],shape[-2],shape[-1])

        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)
        
        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))
        
        out= self.conv10_1(conv9)

        if self.training:
            # Deep Supervision
            out8 = self.out8(conv6)
            out4 = self.out4(conv7)
            out2 = self.out2(conv8)
            if self.res:
                x2 = F.avg_pool2d(x, 2)
                x4 = F.avg_pool2d(x2, 2)
                x8 = F.avg_pool2d(x4, 2)
                out += x
                out2 += x2
                out4 += x4
                out8 += x8
            return [out, out2, out4, out8]
        else:
            if self.res:
                out = out + x
            return out

class ResUnet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = args['nframes'] // 2
        self.res = args['res']
        nframes = self.args['nframes']
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = ResidualBlock(nf, nf, is_activate=False)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = ResidualBlock(nf*2, nf*2, is_activate=False)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = ResidualBlock(nf*4, nf*4, is_activate=False)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = ResidualBlock(nf*8, nf*8, is_activate=False)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = ResidualBlock(nf*16, nf*16, is_activate=False)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = ResidualBlock(nf*16, nf*8, is_activate=False)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = ResidualBlock(nf*8, nf*4, is_activate=False)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = ResidualBlock(nf*4, nf*2, is_activate=False)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = ResidualBlock(nf*2, nf, is_activate=False)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, noise_map=None):
        # shape= x.size()
        # x = x.view(-1,shape[-3],shape[-2],shape[-1])

        conv_in = self.relu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)
        
        conv10 = self.conv10(conv9)
        if self.res:
            out = conv10 + x#[:, self.cf*4:self.cf*4+4]
        else:
            out = conv10

        return out

class DeepResUnet(Module_with_Init):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.nframes = args['nframes']
        self.cf = args['nframes'] // 2
        self.res = args['res']
        nframes = self.args['nframes']
        nf = args['nf']
        in_nc = args['in_nc']
        out_nc = args['out_nc']

        self.conv_in = nn.Conv2d(in_nc*nframes, nf, kernel_size=3, stride=1, padding=1)

        self.conv1 = ResidualBlock(nf, nf)
        self.pool1 = conv3x3(nf, nf*2)
        
        self.conv2 = ResidualBlock(nf*2, nf*2)
        self.pool2 = conv3x3(nf*2, nf*4)
        
        self.conv3 = ResidualBlock(nf*4, nf*4)
        self.pool3 = conv3x3(nf*4, nf*8)
        
        self.conv4 = ResidualBlock(nf*8, nf*8)
        self.pool4 = conv3x3(nf*8, nf*16)
        
        self.conv5 = ResidualBlock(nf*16, nf*16)
        
        self.upv6 = nn.ConvTranspose2d(nf*16, nf*8, 2, stride=2)
        self.conv6 = ResidualBlock(nf*16, nf*8)
        
        self.upv7 = nn.ConvTranspose2d(nf*8, nf*4, 2, stride=2)
        self.conv7 = ResidualBlock(nf*8, nf*4)
        
        self.upv8 = nn.ConvTranspose2d(nf*4, nf*2, 2, stride=2)
        self.conv8 = ResidualBlock(nf*4, nf*2)
        
        self.upv9 = nn.ConvTranspose2d(nf*2, nf, 2, stride=2)
        self.conv9 = ResidualBlock(nf*2, nf)
        
        self.conv10 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)

        # Deep Supervision
        self.out8 = nn.Conv2d(nf*8, out_nc, kernel_size=1)
        self.out4 = nn.Conv2d(nf*4, out_nc, kernel_size=1)
        self.out2 = nn.Conv2d(nf*2, out_nc, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, noise_map=None):
        shape= x.size()
        x = x.view(-1,shape[-3],shape[-2],shape[-1])

        conv_in = self.relu(self.conv_in(x))
        
        conv1 = self.conv1(conv_in)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up6 = self.upv6(conv5)
        up6 = torch.cat((up6, conv4), 1)
        conv6 = self.conv6(up6)
        
        up7 = self.upv7(conv6)+conv3
        up7 = torch.cat((up7, conv3), 1)
        conv7 = self.conv7(up7)
        
        up8 = self.upv8(conv7)+conv2
        up8 = torch.cat((up8, conv2), 1)
        conv8 = self.conv8(up8)
        
        up9 = self.upv9(conv8)+conv1
        up9 = torch.cat((up9, conv1), 1)
        conv9 = self.conv9(up9)
        
        out = self.conv10(conv9)
        if self.training:
            # Deep Supervision
            out8 = self.out8(conv6)
            out4 = self.out4(conv7)
            out2 = self.out2(conv8)
            if self.res:
                x2 = F.avg_pool2d(x, 2)
                x4 = F.avg_pool2d(x2, 2)
                x8 = F.avg_pool2d(x4, 2)
                out += x
                out2 += x2
                out4 += x4
                out8 += x8
            return [out, out2, out4, out8]
        else:
            if self.res:
                out = out + x
            return out