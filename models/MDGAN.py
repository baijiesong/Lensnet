import torch
import torch.nn as nn


class CBR(nn.Module):
    '''
    Convolution-norm-leaky_relu block.
    batch_mode:
    'I': Instance normalization
    'B': Batch normalization
    'G': Group normalization
    lrelu_use: defalut is True. If False, ReLU is used.
    Other parameters: used for 2D-convolution layer.
    '''

    def __init__(self, in_channel, out_channel, padding=1, use_norm=True, kernel=3, stride=1, lrelu_use=False, slope=0.1, batch_mode='G', rate=1):
        super(CBR, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_norm = use_norm
        self.lrelu = lrelu_use

        self.Conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(kernel, kernel), stride=(stride, stride),
                              padding=padding, dilation=(rate, rate))

        if self.use_norm:
            if batch_mode == 'I':
                self.Batch = nn.InstanceNorm2d(self.out_channel)
            elif batch_mode == 'G':
                self.Batch = nn.GroupNorm(
                    self.out_channel // 16, self.out_channel)
            else:
                self.Batch = nn.BatchNorm2d(self.out_channel)

        self.lrelu = nn.LeakyReLU(negative_slope=slope)
        self.relu = nn.ReLU()

    def forward(self, x):

        if not self.lrelu:
            out = self.relu(self.Batch(self.Conv(x)))

        else:
            if self.use_norm:
                out = self.lrelu(self.Batch(self.Conv(x)))
            else:
                out = self.lrelu(self.Conv(x))

        return out


class SELayer(nn.Module):
    '''
    Squeeze-and-excitation network used in G_theta and D_eta.
    '''

    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MDGAN(nn.Module):
    '''
    Complex amplitude map generator, G_theta.
    input_channel=1 for a single diffraction pattern intensity which used as an input.
    To build diffraction pattern intensity generator, input_channel=2 for input complex amplitude map and output_channel=1 for output diffraction intensity.
    '''

    def __init__(self):
        super(MDGAN, self).__init__()

        self.use_norm = True
        self.input_channel = 3
        self.output_channel = 3
        self.lrelu_use = True
        self.batch_mode = 'B'

        c1 = 3
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2

        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1,
                       use_norm=False, lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1,
                       use_norm=False, lrelu_use=self.lrelu_use)
        self.SE1 = SELayer(channel=c1)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE2 = SELayer(channel=c2)

        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE3 = SELayer(channel=c3)

        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE4 = SELayer(channel=c4)

        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l51 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T5 = nn.ConvTranspose2d(in_channels=c4, out_channels=c4, kernel_size=(
            2, 2), stride=(2, 2), padding=(0, 0))

        self.l61 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l60 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(
            2, 2), stride=(2, 2), padding=(0, 0))

        self.l71 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l70 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(
            2, 2), stride=(2, 2), padding=(0, 0))

        self.l81 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l80 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(
            2, 2), stride=(2, 2), padding=(0, 0))

        self.l91 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l90 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm,
                       lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.conv_out_holo = nn.Conv2d(
                in_channels=c1, out_channels=3, kernel_size=(1, 1), padding=0)
        self.SE_out_holo = SELayer(channel=c1)

        # self.apply(weights_initialize)
        self.mpool0 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        l5 = self.conv_T5(self.l51(self.l50(l4_pool)))

        l4_se = self.SE4(l4)

        # Ensure l4_se has the same spatial dimensions as l5 using interpolation
        l4_se = nn.functional.interpolate(l4_se, size=l5.shape[2:], mode='nearest')

        l6 = torch.cat([l5, l4_se], dim=1)
        l6 = self.conv_T6(self.l60(self.l61(l6)))

        # Apply the same approach to subsequent concatenations
        l3_se = self.SE3(l3)
        l3_se = nn.functional.interpolate(l3_se, size=l6.shape[2:], mode='nearest')

        l7 = torch.cat([l6, l3_se], dim=1)
        l7 = self.conv_T7(self.l70(self.l71(l7)))

        l2_se = self.SE2(l2)
        l2_se = nn.functional.interpolate(l2_se, size=l7.shape[2:], mode='nearest')

        l8 = torch.cat([l7, l2_se], dim=1)
        l8 = self.conv_T8(self.l80(self.l81(l8)))

        l1_se = self.SE1(l1)
        l1_se = nn.functional.interpolate(l1_se, size=l8.shape[2:], mode='nearest')

        l9 = torch.cat([l8, l1_se], dim=1)
        out = self.l90(self.l91(l9))

        # l8 = torch.cat([l7, self.SE2(l2)], dim=1)
        # out = self.l80(self.l81(l8))
        out = self.conv_out_holo(self.SE_out_holo(out))
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')

        return out


if __name__ == '__main__':
    model = MDGAN().cuda()
    t = torch.rand(1, 3, 320, 320).cuda()
    print(model(t).shape)
