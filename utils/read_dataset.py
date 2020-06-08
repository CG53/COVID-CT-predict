'''
@Article{he2020sample,
  author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie, Pengtao},
  title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
  journal = {medrxiv},
  year    = {2020},
}
'''

import os
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['COVID-CT', 'NonCOVID-CT']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transformer = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = CovidCTDataset(root_dir='../COVID-CT-Image/',
                              txt_COVID='../COVID-CT-Image/Dataset-Split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='../COVID-CT-Image/Dataset-Split/NonCOVID/trainCT_NonCOVID.txt',
                              transform=train_transformer)
    valset = CovidCTDataset(root_dir='../COVID-CT-Image/',
                            txt_COVID='../COVID-CT-Image/Dataset-Split/COVID/testCT_COVID.txt',
                            txt_NonCOVID='../COVID-CT-Image/Dataset-Split/NonCOVID/testCT_NonCOVID.txt',
                            transform=test_transformer)
    testset = CovidCTDataset(root_dir='../COVID-CT-Image/',
                             txt_COVID='../COVID-CT-Image/Dataset-Split/COVID/valCT_COVID.txt',
                             txt_NonCOVID='../COVID-CT-Image/Dataset-Split/NonCOVID/valCT_NonCOVID.txt',
                             transform=test_transformer)
    batchsize = 5
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    
    # display some images
    print(len(train_loader.dataset)) # 425
    print(len(val_loader.dataset)) # 203
    print(len(test_loader.dataset)) # 118
    for batch_index, batch_samples in enumerate(train_loader): 
        data, target = batch_samples['img'], batch_samples['label']
        print(data.shape, target.shape) # torch.Size([5, 3, 224, 224]), torch.Size([5])
        images, labels = [], []
        for i in range(batchsize):
            images.append(data[i].permute(1, 2, 0).numpy())
            if target[i] == 0:
                labels.append('COVID-19')
            else:
                labels.append('NonCOVID-19')
        _, figs = plt.subplots(1, len(images)) 
        for f, img, label in zip(figs, images, labels):
            f.imshow(img)
            f.set_title(label)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)
        plt.show()
        break
    