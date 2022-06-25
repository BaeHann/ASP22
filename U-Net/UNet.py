import torch.nn as nn
import torch
class conv_and_relu(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(conv_and_relu,self).__init__()
        self.procedures = nn.Sequential(
            nn.Conv2d(channel_in,channel_out,(3,3)),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),    #节约计算资源
            nn.Conv2d(channel_out,channel_out,(3,3)),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.procedures(x)
class U_Net(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(U_Net,self).__init__()
        self.conv1 = conv_and_relu(channel_in,64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv_and_relu(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = conv_and_relu(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = conv_and_relu(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = conv_and_relu(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024,512,(2,2),stride=2)
        self.conv6 = conv_and_relu(1024,512)
        self.up2 = nn.ConvTranspose2d(512,256,(2,2),stride=2)
        self.conv7 = conv_and_relu(512,256)
        self.up3 = nn.ConvTranspose2d(256, 128, (2, 2), stride=2)
        self.conv8 = conv_and_relu(256,128)
        self.up4 = nn.ConvTranspose2d(128,64,(2, 2), stride=2)
        self.conv9 = conv_and_relu(128,64)
        self.last_conv = nn.Conv2d(64,2,(1,1))
    def forward(self,x):
        x1 = self.conv1(x)#开始encode
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        x6 = self.pool3(x5)

        x7 = self.conv4(x6)
        x8 = self.pool4(x7)

        x9 = self.conv5(x8)
        x10_fromdown = self.up1(x9)#开始decode
        x10_fromleft = torch.narrow(x7,2,4,56)
        x10_fromleft = torch.narrow(x10_fromleft,3,4,56)
        x10 = torch.cat((x10_fromleft,x10_fromdown),dim=1)

        x11 = self.conv6(x10)
        x12_fromdown = self.up2(x11)
        x12_fromleft = torch.narrow(x5,2,16,104)
        x12_fromleft = torch.narrow(x12_fromleft,3,16,104)
        x12 = torch.cat((x12_fromleft,x12_fromdown),dim=1)

        x13 = self.conv7(x12)
        x14_fromdown = self.up3(x13)
        x14_fromleft = torch.narrow(x3,2,40,200)
        x14_fromleft = torch.narrow(x14_fromleft,3,40,200)
        x14 = torch.cat((x14_fromleft,x14_fromdown),dim=1)

        x15 = self.conv8(x14)
        x16_fromdown = self.up4(x15)
        x16_fromleft = torch.narrow(x1,2,88,392)
        x16_fromleft = torch.narrow(x16_fromleft,3,88,392)
        x16 = torch.cat((x16_fromleft,x16_fromdown),dim=1)

        x17 = self.conv9(x16)
        x18 = self.last_conv(x17)
        return x18
