import torch
from typing import TYPE_CHECKING, Sequence, Tuple, Union, List
from sacred import Experiment
import sys,os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import torch.nn as nn
import torch.nn.functional as F
import pywt
from einops import rearrange
from config import initialise
from utils.tupperware import tupperware

if TYPE_CHECKING:
    from utils.typing_alias import *

ex = Experiment("WaveNet")
ex = initialise(ex)

def _as_wavelet(wavelet: Union[str, pywt.Wavelet]) -> pywt.Wavelet:
    """将字符串转换为小波对象，或直接返回小波对象"""
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    return wavelet

def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """计算两个1D向量的外积"""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    # 结果是一个矩阵，其中(i,j)位置的值是a[i]*b[j]
    return a_mul * b_mul

def construct_2d_filt(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """构建2D滤波器组
    
    Args:
        lo: 低通输入滤波器
        hi: 高通输入滤波器
        
    Returns:
        堆叠的2D滤波器,维度为[filt_no, 1, height, width]
        四个滤波器按ll, lh, hl, hh顺序排列
    """
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    return torch.stack([ll, lh, hl, hh], 0)

# 将输入的小波对象转换为滤波器张量
def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    wavelet = _as_wavelet(wavelet)
    # 转换为PyTorch张量并根据需要翻转
    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)
    # 从小波对象中获取分解和重构滤波器
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor

# 计算卷积所需的填充量，确保输出大小正确
def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:

    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate"
) -> torch.Tensor:
    wavelet = _as_wavelet(wavelet)
    # 计算需要的填充量
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo))
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo))
    # 对数据进行填充
    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

# 在使用分组卷积的情况下，帮助不同组之间的信息交换
class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups)
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x

class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        # dec_lo: 低通分解滤波器
        # dec_hi: 高通分解滤波器
        # level: 小波分解层数
        # mode: 边界填充模式
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        # 自适应层数
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        l_component = x
        # 1. 构建2D小波核
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)
        # 2. 多层小波分解
        for _ in range(self.level):
            # 填充
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            # 卷积得到高频分量
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            # 分离不同方向的高频分量
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            # 保存分解结果
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        # 保存低频分量
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]

class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component

# Learnable Wavelet Network 可学习的小波网络模块
class LWN(nn.Module):
    """可学习的小波网络模块"""
    
    def __init__(self, 
                 dim: int,
                 wavelet: str = 'haar',
                 initialize: bool = True,
                 head: int = 4,
                 drop_rate: float = 0.,
                 use_ca: bool = False,
                 use_sa: bool = False):
        super().__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        
        # 获取小波滤波器
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )
        
        # 初始化滤波器参数
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*4, dim*6, 1)
        self.conv2 = nn.Conv2d(dim*6, dim*6, 7, padding=3, groups=dim*6)  # dw
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*6, dim*4, 1)
        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2),  # 上采样
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
            # self.sa_norm = LayerNorm2d(dim)
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # 全局池化
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True),  # conv2d
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

    def forward(self, x):
        _, _, H, W = x.shape
        # 1. 小波分解
        ya, (yh, yv, yd) = self.wavedec(x)
        # 2. 特征处理
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        # 3. 小波重构
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None)
        # 4. 空间注意力
        if self.use_sa:
            sa_yh = self.sa_h(yh)   # 水平方向
            sa_yv = self.sa_v(yv)   # 垂直方向
            y = y * (sa_yv + sa_yh)
        # 5. 通道注意力
        if self.use_ca:
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)   # 水平方向
            ca_yv = self.ca_v(yv)   # 垂直方向
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # channel shuffle
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2   # gated channel attention
            y = y * ca
        return y

    def get_wavelet_loss(self):
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        """ Strang 107: Assuming alias cancellation holds:
        P(z) = F(z)H(z)
        Product filter P(z) + P(-z) = 2.
        However since alias cancellation is implemented as soft constraint:
        P_0 + P_1 = 2
        Somehow numpy and torch implement convolution differently.
        For some reason the machine learning people call cross-correlation
        convolution.
        https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172
        Therefore for true convolution one element needs to be flipped.
        """
        # polynomial multiplication is convolution, compute p(z):
        # print(dec_lo.shape, rec_lo.shape)
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0),
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0),
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero

    def alias_cancellation_loss(self):
        """ Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
            F0(z)H0(-z) + F1(z)H1(-z) = 0 """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FFTBlock(nn.Module):
    def __init__(self, n_feat, norm='backward'):  # 'ortho'
        super(FFTBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False)
        )
        self.dim = n_feat
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class FFT3(nn.Module):
    def __init__(self, dim):
        super(FFT3, self).__init__()
        self.dim = dim

        self.conv_in = nn.Conv2d(dim, dim*4, kernel_size=1)

        self.conv2 = nn.Conv2d(dim*4, dim*4, 7, padding=3, groups=dim*4)  # dw
        self.act = nn.GELU()
        self.fft_act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*4, dim, 1)
        self.fft = nn.Parameter(torch.ones((self.dim * 4, 1, 1, 8, 8 // 2 + 1)))
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=8, patch2=8)
        x_patch_fft = torch.fft.rfft2(x_patch.float(), norm='backward')
        y = torch.cat([x_patch_fft.real, x_patch_fft.imag], dim=1)
        y = self.fft_act(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(8, 8), norm='backward')
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=8, patch2=8)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x

class FFT2(nn.Module):
    def __init__(self, dim):
        super(FFT2, self).__init__()
        self.dim = dim
        self.conv_in = nn.Conv2d(dim, dim*2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim*2, dim*2, 7, padding=3, groups=dim*2)  # dw
        self.act = nn.GELU()
        self.fft_act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*2, dim, 1)
        self.fft1 = nn.Parameter(torch.ones((self.dim * 2, 1, 1, 8, 8 // 2 + 1)))
        self.fft2 = nn.Parameter(torch.ones((self.dim * 2, 1, 1, 8, 8 // 2 + 1)))
    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=8, patch2=8)
        x_patch_fft = torch.fft.rfft2(x_patch.float(), norm='backward')
        x_patch_fft = x_patch_fft * self.fft1
        y = torch.cat([x_patch_fft.real, x_patch_fft.imag], dim=1)
        y = self.fft_act(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        x_patch_fft = y * self.fft2
        x_patch = torch.fft.irfft2(x_patch_fft, s=(8, 8), norm='backward')
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=8, patch2=8)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x
# from models.ours.wavelet_block import ResBlock_dwt
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class WaveletBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = LWN(c, wavelet='haar', initialize=True)
        # self.wavelet_block1 = FFT2(c)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma

    def get_wavelet_loss(self):
        return self.wavelet_block1.get_wavelet_loss()


# SEB
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

    def get_wavelet_loss(self):
        return 0.


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super(Encoder, self).__init__()
        self.num_blocks = num_blocks
        self.feature_embed = nn.Conv2d(in_channels=inp_channels, out_channels=dim, kernel_size=3, padding=1, stride=1,
                                       groups=1, bias=True)
        self.b1 = nn.Sequential(*[NAFBlock(dim) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, 2 * dim, 2, 2)
        self.b2 = nn.Sequential(*[NAFBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim * 2, dim * 2 ** 2, 2, 2)
        self.b3 = nn.Sequential(*[NAFBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 2, 2)
        self.b4 = nn.Sequential(*[NAFBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])

    def forward(self, x):
        x = self.feature_embed(x)  # (1, 32, 256, 256)
        x1 = self.b1(x)  # (1, 32, 256, 256)

        x = self.down1(x1)  # (1, 64, 128, 128)
        x2 = self.b2(x)  # (1, 64, 128, 128)

        x = self.down2(x2)  # (1, 128, 64, 64)
        x3 = self.b3(x)  # (1, 128, 64, 64)

        x = self.down3(x3)
        x4 = self.b4(x)

        return x4, x3, x2, x1



class Fusion(nn.Module):
    def __init__(self,
                 dim=32,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super(Fusion, self).__init__()
        self.num_blocks = num_blocks
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])

    def forward(self, x4, x3, x2, x1):
        x3_b = x3.contiguous()
        x = self.up43(x4) + x3
        x3 = self.d3(x)
        # deblur head x3(min) 128
        x2_b = x2.contiguous()
        x = self.up32(x3) + x2
        x2 = self.d2(x)

        return x4, x3, x3_b, x2, x2_b, x1

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss

class Deblur_head(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block = nn.Sequential(
            # nn.Conv2d(num_in, num_mid, kernel_size=1),
            # nn.BatchNorm2d(num_mid),
            # nn.GELU(),
            nn.Conv2d(num_in, num_out, kernel_size=3, stride=1, padding=1),


        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 dim=64,
                 out_channels=3,
                 num_blocks=[2, 4, 4, 6],
                 ):
        super().__init__()
        self.num_blocks = num_blocks
        self.head4 = Deblur_head(int(dim * 2 ** 3), int(dim * 3), 3)
        self.up43 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 4, 1, bias=False),
            nn.PixelShuffle(2)
        )
        self.head3 = Deblur_head(int(dim * 2 ** 2), int(dim * 2 ** 1), out_channels)
        self.up32 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 3, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.head2 = Deblur_head(int(dim * 2 ** 1), int(dim), out_channels)
        self.up21 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, dim * 2 ** 2, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.head1 = Deblur_head(dim, dim, out_channels)

        self.d4 = nn.Sequential(*[WaveletBlock(dim * 2 ** 3) for _ in range(num_blocks[3])])
        self.d3 = nn.Sequential(*[WaveletBlock(dim * 2 ** 2) for _ in range(num_blocks[2])])
        self.d2 = nn.Sequential(*[WaveletBlock(dim * 2) for _ in range(num_blocks[1])])
        self.d1 = nn.Sequential(*[WaveletBlock(dim) for _ in range(num_blocks[0])])


        self.alpha = nn.Parameter(torch.zeros((1, dim * 2, 1, 1)), requires_grad=True)

    def forward(self, x4, x3, x3_b, x2, x2_b, x1):
        # x = x4.contiguous()
        x = self.d4(x4)
        x4 = self.head4(x) if self.training else None

        x = self.up43(x) + x3
        x = self.d3(x)
        x3 = self.head3(x) if self.training else None

        x2_n = x2.contiguous()
        x = self.up32(x) + x2
        x = self.d2(x)
        x2 = self.head2(x) if self.training else None

        x = self.up21(x + x2_n * self.alpha) + x1
        x = self.d1(x)
        x1 = self.head1(x)

        return x1, x2, x3, x4

    def get_wavelet_loss(self):
        wavelet_loss = 0.
        for index, _ in enumerate(self.num_blocks):
            for block in getattr(self, f'd{index+1}'):
                wavelet_loss += block.get_wavelet_loss()
        return wavelet_loss


class LensNet(nn.Module):
    def __init__(self, n_channels=4,
                 dim=64,
                 ):

        super(LensNet, self).__init__()
        # [False, True, True, False]
        # [False, False, False, False]
        self.encoder = Encoder(inp_channels=n_channels,
                                 dim=dim,
                                 num_blocks=[1, 2, 4, 24],
                                 )
        self.fusion = Fusion(dim=dim,
                         num_blocks=[None, 2, 2, None],
                         )
        self.decoder = Decoder(dim=dim,
                         out_channels=3,
                         num_blocks=[2, 2, 2, 2],
                         )

    def __repr__(self):
        return 'MLWNet'

    def forward(self, inp):

        # MLWNet处理
        # x = self.encoder(x)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        # x = self.fusion(*x)  # (1, 128, 64, 64), (1, 64, 128, 128), (1, 32, 256, 256)
        # x1, x2, x3, x4 = self.decoder(*x)
        # 编码
        x4, x3, x2, x1 = self.encoder(inp)
        import ipdb
        ipdb.set_trace()
        fused = self.fusion(x4, x3, x2, x1)
        x1, x2, x3, x4 = self.decoder(*fused)
        return x1 + inp, x2, x3, x4

    def get_wavelet_loss(self):
        return self.fusion.get_wavelet_loss() + self.decoder.get_wavelet_loss()

@ex.automain
def main(_run):
    """主函数用于测试"""
    from torchsummary import summary
    
    args = tupperware(_run.config)
    test_input = torch.randn(1, 4, 64, 64).to(args.device)
    model = LensNet(n_channels=4, dim=64)
    model = model.to(args.device)
    x =model(test_input)
    # 打印模型结构
    summary(model, (4, args.image_height, args.image_width))