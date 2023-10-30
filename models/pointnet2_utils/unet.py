import torch.nn as nn
import torch
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channel=128, out_channel=3, training=True):
        super(UNet3D, self).__init__()
        self.training = training
        self.conv3d1 = nn.Sequential(nn.Conv3d(in_channel, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU())
        self.conv3d2 = nn.Sequential(nn.Conv3d(64, 32, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU())                        
        self.conv3d3 = nn.Sequential(nn.Conv3d(32, 32, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU())
        self.conv3d4 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=1, padding=1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU())


        self.conv2d1 = nn.Sequential(nn.Conv2d(64*7, 256,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.conv2d2 = nn.Sequential(nn.Conv2d(256, 64,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2d3 = nn.Sequential(nn.Conv2d(64, 64,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2d4 = nn.Sequential(nn.Conv2d(64, 64,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.conv2d5 = nn.Sequential(nn.Conv2d(64, 3,
                                             kernel_size=1,
                                             stride=1))
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.2)
    def forward(self, x):
        out1 = self.conv3d1(x)
        out2 = self.conv3d2(out1)
        out3 = self.conv3d3(out2)
        out4 = self.conv3d4(out3)
  
        out4=out4.permute(0,1,4,2,3).reshape(out4.shape[0],out4.shape[4]*out4.shape[1],out4.shape[2],out4.shape[3])
        out5=self.conv2d1(out4)
        out6=self.conv2d2(out5)
        out7=self.conv2d3(out6)
        out7=self.dp1(out7)
        feat_pc=self.conv2d4(out7)
        out8=self.dp2(feat_pc)

        out9=self.conv2d5(out8)
        return out9,feat_pc

