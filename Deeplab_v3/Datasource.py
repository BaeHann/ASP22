import linecache
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np

class TrainData(Dataset):
    def __init__(self,img_dir,label_dir,str_info,transform=None):
        super(TrainData,self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.address_of_info = str_info
    def __len__(self):
        return len(os.listdir(self.img_dir))-1
    def get_tensor_of_the_image(self,img):
        return self.transform(img)
    def __getitem__(self, idx):
        name_this_time = linecache.getline(self.address_of_info, idx+1).split()[0]
        img = Image.open(self.img_dir + name_this_time + '.jpg')
        img = img.resize((500,500))
        tensor_of_img = self.get_tensor_of_the_image(img)
        label = Image.open(self.label_dir + name_this_time + '.png')
        label = label.resize((500,500))
        array_of_label = np.array(label)
        for i in range(array_of_label.shape[0]):
            for j in range(array_of_label.shape[1]):
                if array_of_label[i][j] == 255:
                    array_of_label[i][j] = 0
        tensor_of_label = torch.tensor(array_of_label).long()
        return tensor_of_img,tensor_of_label

class TestData(Dataset):
    def __init__(self,img_dir,label_dir,transform=None):
        super(TestData,self).__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
    def __len__(self):
        #return len(os.listdir(self.img_dir))
        return 5
    def get_tensor_of_the_image(self,img):
        return self.transform(img)
    def __getitem__(self, idx):
        img = Image.open(self.img_dir + str(idx) + '.jpg')
        img = img.resize((500,500))
        tensor_of_img = self.get_tensor_of_the_image(img)
        label = Image.open(self.label_dir + str(idx) + '.png')
        label = label.resize((500,500))
        array_of_label = np.array(label)
        for i in range(array_of_label.shape[0]):
            for j in range(array_of_label.shape[1]):
                if array_of_label[i][j] == 255:
                    array_of_label[i][j] = 0
        tensor_of_label = torch.tensor(array_of_label).long()
        return tensor_of_img,tensor_of_label

train_data = TrainData("Data/train_img/","Data/train_label/",'trainval.txt',ToTensor())
test_data = TestData("Data/test_img/","Data/test_label/",ToTensor())
train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)

"""
for batch,(X,y) in  enumerate(train_dataloader):
    y = y.int()
    y = np.array(y)
    a = y[1] * 255
    label_this_time = Image.fromarray(a)
    label_this_time.show()
"""
