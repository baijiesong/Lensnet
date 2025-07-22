import torch
from torch import nn
from collections import OrderedDict
from torch import fft
import torch.nn.functional as F
from PIL import Image
import numpy as np
from skimage.transform import resize


def LenslessCamera(psfs):
    """
    Lensless Camera with optionally separable forward and adjoint models
    """
    sensor = Crop(psfs.shape)
    diffuser = Convolution(sensor.pad(psfs))
    return LenslessOperator(sensor, diffuser)


class LenslessOperator:
    def __init__(self, sensor, diffuser):
        """
        Lensless imaging forward and adjoint models.
        Expects images in (B, C, H, W) format.
        """        
        self.crop  = sensor.forward
        self.pad   = sensor.adjoint
        self.sensor = sensor

        self.convolve = diffuser.forward
        self.cross_correlate = diffuser.adjoint
        self.autocorrelation = diffuser.autocorrelation

    def forward(self, image):
        output = image
        output = self.convolve(output)
        output = self.crop(output)
        return output

    def adjoint(self, image):
        output = image
        output = self.pad(output)
        output = self.cross_correlate(output)
        return output


class Crop:
    def __init__(self, shape):
        """
        Just pads and crops.
        """
        size = shape[-2:]
        pad_size = [s * 2 for s in size]

        ys, yr = divmod(pad_size[0] - size[0], 2)
        xs, xr = divmod(pad_size[1] - size[1], 2)
        ye = ys + size[0]
        xe = xs + size[1]

        self.size     = tuple(size)
        self.pad_size = tuple(pad_size)
        self.y_center = slice(ys, ye)
        self.x_center = slice(xs, xe)
        self.pad_args = xs, xs + xr, ys, ys + yr

        self.crop = self.forward
        self.pad  = self.adjoint

    def forward(self, image):
        if image.shape[-2:] != self.pad_size:
            raise ValueError(
                f"image shape {image.shape} "
                f"does not match padded sensor size {self.pad_size}"
            )
        return image[..., self.y_center, self.x_center]

    def adjoint(self, image):
        if image.shape[-2:] != self.size:
            raise ValueError(
                f"image shape {image.shape} "
                f"does not match sensor size {self.size}"
            )
        return F.pad(image, self.pad_args)


class Convolution:
    def __init__(self, padded_psf):
        """
        Multiplication in fourier domain.
        """
        self.h = ft(padded_psf)

    def forward(self, image):
        return ift(self.h * ft(image))

    def adjoint(self, image):
        return ift(self.h.conj() * ft(image))

    def autocorrelation(self):
        return (self.h * self.h.conj()).real


def ft(image):
    return fft.rfft2(fft.ifftshift(image, dim=(-2, -1)), norm='ortho')
    

def ift(image):
    return fft.fftshift(fft.irfft2(image, norm='ortho'), dim=(-2, -1))


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, output_padding=(0, 0, 0, 0)):
        super().__init__()

        features = init_features
        self.encoder1 = unet_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = unet_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = unet_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = unet_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = unet_block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            in_channels=features * 16,
            out_channels=features * 8,
            kernel_size=2,
            stride=2,
            output_padding=output_padding[0],
        )
        self.decoder4 = unet_block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8,
            features * 4,
            kernel_size=2,
            stride=2,
            output_padding=output_padding[1],
        )
        self.decoder3 = unet_block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4,
            features * 2,
            kernel_size=2,
            stride=2,
            output_padding=output_padding[2],
        )
        self.decoder2 = unet_block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2,
            features,
            kernel_size=2,
            stride=2,
            output_padding=output_padding[3],
        )
        self.decoder1 = unet_block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)


def unet_block(in_channels, features, name):
    layers = [
        (
            name + "conv1",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ),
        (name + "relu1", nn.ReLU(inplace=True)),
        (
            name + "conv2",
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ),
        (name + "relu2", nn.ReLU(inplace=True)),
    ]
    return nn.Sequential(OrderedDict(layers))


class PrimalBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()

        w = width

        p = w + w
        self.primal_conv = torch.nn.Sequential(*[
            nn.Conv2d(p, p * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(p * 2, w, 1),
        ])


    def forward(self, inputs):
        camera, primal, dual, image = inputs

        dual_fwd = color_pack(camera.forward(color_unpack(primal)))
        dual = image - dual_fwd

        primal_adj = color_pack(camera.adjoint(color_unpack(dual)))
        primal_cat = torch.cat([primal_adj, primal], dim=1)
        primal = primal + self.primal_conv(primal_cat)

        return primal, dual


class PrimalDualBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()

        w = width

        d = w + w + 1
        self.dual_conv = torch.nn.Sequential(*[
            nn.Conv2d(d, d * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d * 2, w, 1),
        ])

        p = w + w
        self.primal_conv = torch.nn.Sequential(*[
            nn.Conv2d(p, p * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(p * 2, w, 1),
        ])
        
    def forward(self, inputs):
        camera, primal, dual, image = inputs

        dual_fwd = color_pack(camera.forward(color_unpack(primal)))
        dual_cat = torch.cat([dual_fwd, dual, image], dim=1)
        dual = dual + self.dual_conv(dual_cat)

        primal_adj = color_pack(camera.adjoint(color_unpack(dual)))
        primal_cat = torch.cat([primal_adj, primal], dim=1)
        primal = primal + self.primal_conv(primal_cat)

        return primal, dual


class UPDN(nn.Module):

    def __init__(self, width=5, depth=10, learned_models=5, primal_only=False):
        super().__init__()

        MWDNS = True
        if MWDNS:
            psf = Image.open('MWDNs_psf.png').convert('RGB')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).permute(2, 0, 1)
        else:
            psf = Image.open('Mirflickr_psf.tiff').convert('RGB')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            ds = 4
            psf = resize(
                psf, (psf.shape[0]//ds, psf.shape[1]//ds), mode='constant', anti_aliasing=True)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).permute(2, 0, 1)
        self.depth = depth
        self.width = width

        psfs = torch.tile(psf.unsqueeze(1), (1, 1, learned_models or 1, 1, 1))
        self.psfs = torch.nn.Parameter(psfs, requires_grad=learned_models > 0)

        block = PrimalBlock if primal_only else PrimalDualBlock

        # Just one of these is enough
        self.layers = torch.nn.Sequential(*[
            block(self.width)
            for i in range(0, self.depth)
        ])

        self.unet = UNet(
            in_channels=width,
            out_channels=1,
            init_features=width * 2,
            # output_padding=[(1, 0), (1, 0), (1, 0), (0, 0)],  # diffusercam full
            output_padding=[(0, 0), (0, 0), (0, 0), (0, 0)] if MWDNS else [(1, 0), (1, 0), (1, 0), (0, 0)], # MWDNs
        )

    def model_images(self):
        b, c, v, h, w = self.psfs.shape
        return {
            "PSFs": self.psfs.reshape(b * v, c, h, w),
        }

    def forward(self, image, denoise=True, depth=None, kernel=0):
        b, c, h, w = image.shape
        
        v = self.width
        k = kernel

        # Reload camera on each epoch
        camera = LenslessCamera(self.psfs)
        image  = image.reshape(b * c, 1, h, w)
        dual   = torch.zeros(b * c, v, h, w, device=image.device)
        primal = torch.zeros(b * c, v, h * 2, w * 2, device=image.device)

        for i in range(0, depth or self.depth):
            primal, dual = self.layers[i]((camera, primal, dual, image))

        output = camera.crop(primal)

        if denoise:
            output = output[:, [k]] + self.unet(output)
        else:
            output = output[:, [k]]

        output = output.reshape((b, c, h, w))
        output = torch.sigmoid(output)
        return output


def color_unpack(x):
    """
    Unpack colors from the batch channel
    """
    c = 3
    bc, k, h, w = x.shape
    return x.reshape((bc // c, c, k, h, w))


def color_pack(x):
    """
    Hide colors in batch channel
    """
    b, c, k, h, w = x.shape
    assert c == 3
    return x.reshape((b * c, k, h, w))


if __name__ == '__main__':
    t = torch.rand(1, 3, 320, 320)
    model = UPDN()
    model.eval()
    with torch.no_grad():
        res = model(t)
        print(res.shape)
