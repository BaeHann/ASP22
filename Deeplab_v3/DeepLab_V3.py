import torch.nn as nn
import torch
from torch.nn.functional import interpolate
class Basic_Unit(nn.Module):#ResNet_50的残差块
    def __init__(self,channel_in,channel_mid,channel_out,mid_stride=1,mid_padding=1,mid_dilation=1):
        super(Basic_Unit,self).__init__()
        self.conv1 = nn.Conv2d(channel_in,channel_mid,(1,1))
        self.bnorm1 = nn.BatchNorm2d(channel_mid)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel_mid,channel_mid,(3,3),mid_stride,mid_padding,mid_dilation)#每一个残差块中只有在3*3卷积那里才有可能出现补零、空洞与非1步长
        self.bnorm2 = nn.BatchNorm2d(channel_mid)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channel_mid,channel_out,(1,1))
        self.bnorm3 = nn.BatchNorm2d(channel_out)
        if mid_stride!=1 or channel_in!=channel_out:
            self.bypass = nn.Sequential(
                nn.Conv2d(channel_in,channel_out,(1,1),mid_stride),
                nn.BatchNorm2d(channel_out)
            )
        else:
            self.bypass = None
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self,x):
        original_x = x
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bnorm3(x)
        if self.bypass!=None:
            original_x = self.bypass(original_x)
        x = torch.add(x,original_x)
        x = self.relu3(x)
        return x

class stage0(nn.Module):
    def __init__(self):
        super(stage0,self).__init__()
        self.conv1 = nn.Conv2d(3,64,(7,7),2,3)
        self.bnorm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(3,2,1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        return x
class stage1(nn.Module):
    def __init__(self):
        super(stage1,self).__init__()
        self.resblock1 = Basic_Unit(64,64,256)
        self.resblock2 = Basic_Unit(256,64,256)
        self.resblock3 = Basic_Unit(256,64,256)
    def forward(self,x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return x
class stage2(nn.Module):
    def __init__(self):
        super(stage2,self).__init__()
        self.resblock1 = Basic_Unit(256,128,512,2)
        self.resblock2 = Basic_Unit(512,128,512,1)
        self.resblock3 = Basic_Unit(512,128,512)
        self.resblock4 = Basic_Unit(512,128,512)
    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        return x
class stage3(nn.Module):
    def __init__(self):
        super(stage3, self).__init__()
        self.resblock1 = Basic_Unit(512,256,1024)
        self.resblock2 = Basic_Unit(1024,256,1024,1,2,2)
        self.resblock3 = Basic_Unit(1024, 256, 1024, 1, 2, 2)
        self.resblock4 = Basic_Unit(1024, 256, 1024, 1, 2, 2)
        self.resblock5 = Basic_Unit(1024, 256, 1024, 1, 2, 2)
        self.resblock6 = Basic_Unit(1024, 256, 1024, 1, 2, 2)
    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        return x
class stage4(nn.Module):
    def __init__(self):
        super(stage4, self).__init__()
        self.resblock1 = Basic_Unit(1024,512,2048,1,2,2)
        self.resblock2 = Basic_Unit(2048,512,2048,1,4,4)
        self.resblock3 = Basic_Unit(2048, 512, 2048, 1, 4, 4)
    def forward(self,x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return x

class ASPP(nn.Module):
    def __init__(self):
        super(ASPP,self).__init__()
        self.branch1_conv = nn.Conv2d(2048,256,(1,1),1)
        self.branch1_bnorm = nn.BatchNorm2d(256)
        self.branch1_relu = nn.ReLU(inplace=True)
        self.branch2_conv = nn.Conv2d(2048,256,(3,3),1,12,12)
        self.branch2_bnorm = nn.BatchNorm2d(256)
        self.branch2_relu = nn.ReLU(inplace=True)
        self.branch3_conv = nn.Conv2d(2048,256,(3,3),1,24,24)
        self.branch3_bnorm = nn.BatchNorm2d(256)
        self.branch3_relu = nn.ReLU(inplace=True)
        self.branch4_conv = nn.Conv2d(2048, 256, (3, 3), 1, 36, 36)
        self.branch4_bnorm = nn.BatchNorm2d(256)
        self.branch4_relu = nn.ReLU(inplace=True)
        self.branch5_avgpool = nn.AvgPool2d(63)
        self.branch5_conv = nn.Conv2d(2048,256,(1,1))
        self.branch5_bnorm =nn.BatchNorm2d(256)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Conv2d(1280,256,(1,1))
        self.last_bnorm = nn.BatchNorm2d(256)
        self.last_relu = nn.ReLU(inplace=True)
        #不用Adaptive average pooling是因为对于正方形平面而言没有效果，相当于什么也没做
    def forward(self,x):
        upper_size = x.size()[2]
        x1 = self.branch1_conv(x)
        x1 = self.branch1_bnorm(x1)
        x1 = self.branch1_relu(x1)
        x2 = self.branch2_conv(x)
        x2 = self.branch2_bnorm(x2)
        x2 = self.branch2_relu(x2)
        x3 = self.branch3_conv(x)
        x3 = self.branch3_bnorm(x3)
        x3 = self.branch3_relu(x3)
        x4 = self.branch4_conv(x)
        x4 = self.branch4_bnorm(x4)
        x4 = self.branch4_relu(x4)
        x5 = self.branch5_avgpool(x)
        x5 = self.branch5_conv(x5)
        x5 = self.branch5_bnorm(x5)
        x5 = self.branch5_relu(x5)
        x5 = interpolate(x5,size=[upper_size,upper_size],mode='nearest')
        x_concat = torch.cat((x1,x2,x3,x4,x5),dim=1)
        x_last = self.last_conv(x_concat)
        x_last = self.last_bnorm(x_last)
        x_last = self.last_relu(x_last)
        return x_last

class Deeplab_v3(nn.Module):
    def __init__(self):
        super(Deeplab_v3,self).__init__()
        self.stage0 = stage0()
        self.stage1 = stage1()
        self.stage2 = stage2()
        self.stage3 = stage3()
        self.stage4 = stage4()
        self.ASPP = ASPP()
        self.conv1 = nn.Conv2d(256,256,(3,3),padding=1)
        self.bnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256,21,(1,1))
        self.bnorm2 = nn.BatchNorm2d(21)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self,x):
        upper_size = x.size()[2]
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.ASPP(x)
        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.relu2(x)
        x = interpolate(x,size=[upper_size,upper_size],mode='bilinear',align_corners=True)
        return x

