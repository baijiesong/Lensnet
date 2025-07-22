import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_file, img_options=None):
        super(DataLoaderTrain, self).__init__()

        with np.load(rgb_file) as data:
            self.inp_files = data['blur']
            self.tar_files = data['gt']

        self.img_options = img_options
        self.sizex = len(self.inp_files)  # get the size of target

        self.transform = A.Compose([
            A.Transpose(p=0.3),
            A.Flip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(p=0.3),
            A.Resize(height=img_options['h'], width=img_options['w']),
            ],
            additional_targets={
                'target': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_img = np.transpose(self.inp_files[index_], (1, 2, 0))
        tar_img = np.transpose(self.tar_files[index_], (1, 2, 0))

        transformed = self.transform(image=inp_img, target=tar_img)

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        return inp_img, tar_img


class DataLoaderVal(Dataset):
    def __init__(self, rgb_file, img_options=None):
        super(DataLoaderVal, self).__init__()

        with np.load(rgb_file) as data:
            self.inp_files = data['blur']
            self.tar_files = data['gt']

        self.img_options = img_options
        self.sizex = len(self.inp_files)  # get the size of target

        self.transform = A.Compose([
            A.Resize(height=img_options['h'], width=img_options['w']), ],
            additional_targets={
                'target': 'image',
            }
        )

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_img = np.transpose(self.inp_files[index_], (1, 2, 0))
        tar_img = np.transpose(self.tar_files[index_], (1, 2, 0))

        transformed = self.transform(image=inp_img, target=tar_img)

        inp_img = F.to_tensor(transformed['image'])
        tar_img = F.to_tensor(transformed['target'])

        return inp_img, tar_img
