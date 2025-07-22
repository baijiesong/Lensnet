import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.avgpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


def WieNer(blur, psf, delta):
    blur_fft = torch.fft.rfft2(blur)
    psf_fft = torch.fft.rfft2(psf)
    psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + delta)
    img = torch.fft.ifftshift(torch.fft.irfft2(psf_fft * blur_fft), (2, 3))
    return img.real


class MWDNet(nn.Module):
    def __init__(self, n_channels, n_classes, psf):
        super(MWDNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.psf = psf    # 1,3,H,W

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self.w = nn.Parameter(torch.tensor(np.ones(4) * 0.001, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(np.ones(4)*0.01, dtype=torch.float32))

    def forward(self, x):            
        x1 = self.inc(x)          
        x2 = self.down1(x1)        
        x3 = self.down2(x2)        
        x4 = self.down3(x3)    
        x5 = self.down4(x4)     

        psf = torch.sum(self.psf, dim=1, keepdim=True)
        psf1 = self.w[0] * psf
        psf2 = self.w[1] * F.avg_pool2d(psf, 2)
        psf3 = self.w[2] * F.avg_pool2d(psf, 4)
        psf4 = self.w[3] * F.avg_pool2d(psf, 8)
        x4 = WieNer(x4, psf4, self.delta[3])
        x3 = WieNer(x3, psf3, self.delta[2])
        x2 = WieNer(x2, psf2, self.delta[1])
        x1 = WieNer(x1, psf1, self.delta[0])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class MWDNet_CPSF(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(MWDNet_CPSF, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.psf = nn.Parameter(torch.randn(1, 3, 256, 256))
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

        self.delta = nn.Parameter(torch.tensor(np.ones(5)*0.01, dtype=torch.float32))
        self.w = nn.Parameter(torch.tensor([0.001], dtype=torch.float32))

        self.inc0 = DoubleConv(3, 64)
        self.down11 = Down(64, 128)
        self.down22 = Down(128, 256)
        self.down33 = Down(256, 512)

    def forward(self, x):        
        x1 = self.inc(x)         
        x2 = self.down1(x1)  
        x3 = self.down2(x2)     
        x4 = self.down3(x3)       
        x5 = self.down4(x4) 

        psf1 = self.inc0(self.w * self.psf)
        psf2 = self.down11(psf1)
        psf3 = self.down22(psf2)
        psf4 = self.down33(psf3)

        x4 = WieNer(x4, psf4, self.delta[3])
        x3 = WieNer(x3, psf3, self.delta[2])
        x2 = WieNer(x2, psf2, self.delta[1])
        x1 = WieNer(x1, psf1, self.delta[0])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.clamp(0, 1)


if __name__ == '__main__':
    model = MWDNet_CPSF().cuda()
    t = torch.randn(1, 3, 256, 256).cuda()
    res = model(t)
    print(res.shape)