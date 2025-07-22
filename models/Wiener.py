from skimage.restoration import wiener, unsupervised_wiener
import numpy as np
import os
from PIL import Image
from skimage import exposure
from tqdm import tqdm
from skimage import io


psf = Image.open('MWDNs_psf.png').convert('RGB')
psf = np.array(psf)
psf = psf / np.max(psf)
input_folder = 'input'
target_folder = 'output'
imgs = os.listdir(input_folder)
for img in tqdm(imgs):
    image = Image.open(os.path.join(input_folder, img)).convert('RGB')
    image = np.array(image)
    image = image / np.max(image)
    res = []
    for i in range(3):
        # deconvolved = wiener(image[:, :, i], psf[:, :, i], 0.01)
        deconvolved, _ = unsupervised_wiener(image[:, :, i], psf[:, :, i])
        res.append(deconvolved)
    res = np.array(res).transpose(1, 2, 0)
    # res = (res + 1) * 127.5
    rescaled = exposure.rescale_intensity(res, in_range=(res.min(), res.max()))
    normalized = (rescaled + 1) / 2
    normalized = (normalized * 255).astype(np.uint8)
    io.imsave(os.path.join(target_folder, img), normalized)