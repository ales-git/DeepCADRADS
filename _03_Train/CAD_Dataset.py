import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import re


class CAD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, file_name, dim, transform=None, extract_names = False):
        self.path_img = data_path
        self.transform = transform
        self.extract_names = extract_names
        self.file = file_name
        self.dim = dim
        self.list_uniqID = []
        self.label = []
        self.data_list = []

        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        f = open(self.file, "r")
        for line in f:
            line = line.strip()
            path_file, l = line.split(" ")[0], line.split(" ")[1]
            self.label.append(l)
            self.list_uniqID.append(path_file.split('/')[-1])
        f.close()

        for uniqID,l in zip(self.list_uniqID,self.label):
            list_pat_img = [f for f in os.listdir(self.path_img) if f.startswith(uniqID)]
            list_pat_img.sort(key=natural_keys)
            views = list(set([s.split('_')[-1] for s in list_pat_img]))
            views.sort(key=natural_keys)
            for view in views:
                im_view = []
                for img in list_pat_img:
                    if view in img:
                        im_view.append(img)
                case = {'uniqID':uniqID,'img_names':im_view,'label':l}
                self.data_list.append(case)



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # print(self.img[idx])
        # print(self.dir_path)
        item = self.data_list[idx]
        img_LAD = read_image(os.path.join(self.path_img,item['img_names'][0]))
        img_LCX = read_image(os.path.join(self.path_img,item['img_names'][1]))
        img_RCA = read_image(os.path.join(self.path_img,item['img_names'][2]))
        resize_img = transforms.Resize((self.dim, self.dim))
        img_LAD = resize_img(img_LAD).squeeze(0)
        img_LCX = resize_img(img_LCX).squeeze(0)
        img_RCA = resize_img(img_RCA).squeeze(0)

        # final 3-channel img
        i = torch.stack([img_LAD.type(torch.float), img_LCX.type(torch.float), img_RCA.type(torch.float)], dim=0)
        i /= 255

        # normalize with imagenet weights
        normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229, 0.224, 0.225])
        i = normalize(i)

        # label
        l = torch.tensor(float(item['label']))

        # name
        img_name = item['uniqID']

        if self.transform:
            i = self.transform(i)
        if self.extract_names:

            return i, l, img_name # transformed img,label,uniqID
        else:
            return i, l
