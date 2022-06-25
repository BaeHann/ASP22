from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
import os
import numpy as np

class TrainData(Dataset):
    def __init__(self,img_dir_1,img_dir_2,transform=None,target_transform=None):
        super(TrainData,self).__init__()
        self.img_dir = img_dir_1
        self.label_dir = img_dir_2
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(os.listdir(self.img_dir))
        #return 2000
    def get_tensor_of_the_image(self,img_path:str):
        img=Image.open(img_path)
        return self.transform(img)
    def get_tensor_of_the_label(self,label_path:str):
        img = Image.open(label_path)
        img = img.resize((388, 388))
        return self.target_transform(img).long()[0]
    def __getitem__(self, idx):
        #print(idx)
        img_path = self.img_dir+str(idx)+".jpg"
        label_path= self.label_dir+str(idx)+".jpg"
        return self.get_tensor_of_the_image(img_path),self.get_tensor_of_the_label(label_path)

class TestData(Dataset):
    def __init__(self,img_dir_1,img_dir_2,transform=None,target_transform=None):
        super(TestData,self).__init__()
        self.img_dir = img_dir_1
        self.label_dir = img_dir_2
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(os.listdir(self.img_dir))-1
    def get_tensor_of_the_image(self,img_path:str):
        img=Image.open(img_path)
        img = img.resize((572,572))
        return self.transform(img)
    def get_tensor_of_the_label(self,label_path:str):
        img = Image.open(label_path)
        img = img.resize((388,388))
        return self.target_transform(img).long()[0]
    def __getitem__(self, idx):
        #print(idx)
        img_path = self.img_dir+"train-volume"+str(idx+25)+".jpg"
        label_path= self.label_dir+"train-labels"+str(idx+25)+".jpg"
        return self.get_tensor_of_the_image(img_path),self.get_tensor_of_the_label(label_path)

train_data = TrainData("data/train_img/","data/train_label/",ToTensor(),ToTensor())
test_data = TestData("data/images/","data/labels/",ToTensor(),ToTensor())
train_dataloader = DataLoader(train_data, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=20, shuffle=True)

"""
for batch, (X, y) in enumerate(test_dataloader):
    print(batch)
    print(y[0].shape)
    a=np.array(y[0])
    print(a)
"""

