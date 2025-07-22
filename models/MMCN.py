import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage.transform import resize


class MMCN(nn.Module):
    def __init__(self, K=5):
        super(MMCN, self).__init__()
        self.K = K  # Number of unrolled ADMM iterations
        self.input_channels = 3

        # Define the PSF as an nn.Parameter
        # For 3 channels, we can define a PSF for each channel
        # If the PSF is the same across channels, we can replicate it
        MWDNS = True
        if MWDNS:
            psf = Image.open('MWDNs_psf.png').convert('RGB')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).permute(2, 0, 1).unsqueeze(1)
            self.input_size = (320, 320)
            self.psf_size = (320, 320)
        else:
            psf = Image.open('Mirflickr_psf.tiff').convert('RGB')
            psf = np.array(psf)
            psf = np.clip(psf, a_min=0, a_max=psf.max()).astype(np.float32)
            ds = 4
            psf = resize(
                psf, (psf.shape[0]//ds, psf.shape[1]//ds), mode='constant', anti_aliasing=True)
            psf /= np.linalg.norm(psf.ravel())
            psf = torch.from_numpy(psf).permute(2, 0, 1).unsqueeze(1)
            self.input_size = (270, 480)
            self.psf_size = (270, 480)

        self.psf = nn.Parameter(psf, requires_grad=False)

        # Define the unrolled ADMM blocks
        self.admm_blocks = nn.ModuleList()
        for _ in range(K):
            self.admm_blocks.append(ADMMBlock(self.psf_size, self.input_channels))

        # Measurement processing layer (for the compensation branch)
        self.measurement_layer = nn.Sequential(
            nn.Conv2d(self.input_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # No pooling to maintain spatial size
        )

        # Compensation branch layers for processing intermediate s^(k)
        self.compensation_layers = nn.ModuleList()
        for _ in range(K):
            self.compensation_layers.append(nn.Sequential(
                nn.Conv2d(self.input_channels, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                # No pooling
            ))

        # U-Net decoder
        # Input channels adjusted based on the number of features concatenated
        self.decoder = UNetDecoder(input_channels=24 * (K + 1))

    def forward(self, b):
        """
        Forward pass of the MMCN.
        :param b: Measurement input tensor of shape [batch_size, 3, H, W]
        :return: Reconstructed image tensor of shape [batch_size, 3, H, W]
        """
        # Initialize variables
        batch_size = b.size(0)
        s = torch.zeros_like(b)  # Shape: [batch_size, 3, H, W]
        u_x = torch.zeros_like(b)
        u_y = torch.zeros_like(s)
        u_z = torch.zeros_like(s)
        s_intermediates = []

        # Process the measurement
        measurement_features = self.measurement_layer(b)

        # Unrolled ADMM iterations
        for k in range(self.K):
            # ADMM update
            s, u_x, u_y, u_z = self.admm_blocks[k](b, s, u_x, u_y, u_z, self.psf)
            s_intermediates.append(s)

        # Compensation branch: process measurement and s_intermediates
        features = [measurement_features]
        for k in range(self.K):
            s_k = s_intermediates[k]
            # Process s_k through the compensation layer
            s_k_features = self.compensation_layers[k](s_k)
            features.append(s_k_features)

        # Concatenate all features
        concat_features = torch.cat(features, dim=1)

        # Pass through U-Net decoder
        output = self.decoder(concat_features)

        return output


class ADMMBlock(nn.Module):
    def __init__(self, psf_size, input_channels):
        super(ADMMBlock, self).__init__()

        # ADMM parameters initialized as learnable parameters
        self.rho_x = nn.Parameter(torch.tensor(1.0))
        self.rho_y = nn.Parameter(torch.tensor(1.0))
        self.rho_z = nn.Parameter(torch.tensor(1.0))
        self.lambd = nn.Parameter(torch.tensor(0.1))  # Regularization parameter

        self.psf_size = psf_size
        self.input_channels = input_channels

    def forward(self, b, s, u_x, u_y, u_z, psf):
        """
        Forward pass of an ADMM block.
        """
        # Convolve s with psf for each channel
        # Using groups=input_channels to perform channel-wise convolution
        Hs = F.conv2d(s, psf, padding='same', groups=self.input_channels)

        # Update x
        x_numerator = b + self.rho_x * Hs - u_x
        x_denominator = 1 + self.rho_x
        x = x_numerator / x_denominator

        # Update y
        y = torch.relu(s + u_y / self.rho_y)

        # Update z (using soft-thresholding)
        z = soft_threshold(s + u_z / self.rho_z, self.lambd / self.rho_z)

        # Compute H^T * x
        psf_transpose = torch.flip(psf, [2, 3])  # Flip PSF for transpose
        Ht_x = F.conv2d(x, psf_transpose, padding='same', groups=self.input_channels)

        # Update s
        s_numerator = self.rho_x * Ht_x + self.rho_y * y - u_y + self.rho_z * z - u_z
        s_denominator = self.rho_x + self.rho_y + self.rho_z + 1e-6
        s = s_numerator / s_denominator

        # Update dual variables
        Hs = F.conv2d(s, psf, padding='same', groups=self.input_channels)
        u_x = u_x + self.rho_x * (Hs - x)
        u_y = u_y + self.rho_y * (s - y)
        u_z = u_z + self.rho_z * (s - z)

        return s, u_x, u_y, u_z


def soft_threshold(x, lam):
    """
    Soft-thresholding operator.
    """
    return torch.sign(x) * F.relu(torch.abs(x) - lam)


class UNetDecoder(nn.Module):
    def __init__(self, input_channels):
        super(UNetDecoder, self).__init__()

        # Adjusted decoder to output 3 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Final output layer to get back to 3-channel image
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net decoder.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_conv(x)
        return x.clamp(0, 1)
    

if __name__ == '__main__':

    # Define the network
    mmcn = MMCN(K=5).cuda()

    # Example measurement tensor
    b = torch.randn(1, 3, 270, 480).cuda()  # Batch size of 1

    mmcn.eval()

    with torch.no_grad():
        # Forward pass
        reconstructed_image = mmcn(b)
        print(reconstructed_image.shape)