import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from skimage.transform import resize


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
    
class double_conv2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch,momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x    

    
    

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv2(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX //
                   2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class FlatNet(nn.Module):
    def __init__(self, n_channels=3):
        super(FlatNet, self).__init__()
        MWDNS = False
        if MWDNS:
            psf = Image.open('MWDNs_psf.png').convert('L')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).unsqueeze(2)
        else:
            psf = Image.open('Mirflickr_psf.tiff').convert('L')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            ds = 4
            psf = resize(
                psf, (psf.shape[0]//ds, psf.shape[1]//ds), mode='constant', anti_aliasing=True)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).unsqueeze(2)
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 3)
        if MWDNS:
            self.PhiL = nn.Parameter(psf) 
            self.PhiR = nn.Parameter(psf)
        else:
            psf_t1 = psf.permute(2, 0, 1)
            psf_t2 = psf.permute(2, 1, 0)
            # self.PhiL = nn.Parameter(torch.randn(270, 270, 3)) 
            # self.PhiR = nn.Parameter(torch.randn(480, 480, 3))
            self.PhiL = nn.Parameter(
                torch.bmm(psf_t1, psf_t2).permute(1, 2, 0))
            self.PhiR = nn.Parameter(
                torch.bmm(psf_t2, psf_t1).permute(1, 2, 0))
        self.bn=nn.BatchNorm2d(n_channels,momentum=0.99)

    def forward(self, Xinp):
        
        X0 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,0,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X11 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,1,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X12 = F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,2,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
        Xout = torch.cat((X12,X11,X0),3)
        x = Xout.permute(0,3,1,2)
        x = self.bn(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
    

if __name__ == '__main__':
    model = FlatNet().cuda()
    t = torch.randn(1, 3, 270, 480).cuda()
    res = model(t)
    print(res.shape)