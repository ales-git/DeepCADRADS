import CAD_Dataset as dat
from torchvision import transforms
import random
import torchvision.transforms.functional as TF
import torch
from config import Config_03_Train as config

def load_data(data_path, train_txt_path, val_txt_path, dim=None):

    class MyRotationTransform:
        """Rotate by one of the given angles."""
        def __init__(self, angles):
            self.angles = angles
        def __call__(self, x):
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)

    torch.manual_seed(23)
    transform = transforms.Compose(
        [MyRotationTransform(angles=config.rot_angles),
         transforms.RandomHorizontalFlip(p=config.hflip_prob),
         transforms.RandomVerticalFlip(p=config.vflip_prob)
         ])

    traindata = dat.CAD_Dataset(data_path,train_txt_path, dim, transform)
    valdata = dat.CAD_Dataset(data_path,val_txt_path, dim)

    return traindata, valdata