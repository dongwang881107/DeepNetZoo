import torch.nn as nn
import torch
from deepzoo.architecture.src import *


class REDCNN(nn.Module):
    # Low-Dose CT With a Residual Encoder-Decoder Convolutional Neural Network
    # Page 2526, Figure 1
    # Conv2d (s=1) + ConvTranspose2d (s=1)
    def __init__(self, bn_flag, sa_flag):
        super(REDCNN, self).__init__()
        print('RENCNN: Residual Encoder-Decoder Convolutional Neural Network in TMI paper')

        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.out_channel = 96
        self.acti = 'relu'
        self.bn_flag = bn_flag # batch normalization
        self.sa_flag = sa_flag # self attention
        if self.sa_flag == True:
            self.sa = SelfAttenBlock(self.out_channel)

        self.layer1 = conv_block('conv', 1, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer2 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer3 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer4 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer5 = conv_block('conv', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer6 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer7 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer8 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer9 = conv_block('trans', self.out_channel, self.out_channel, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)
        self.layer10 = conv_block('trans', self.out_channel, 1, self.kernel_size, self.stride, self.padding, self.acti, self.bn_flag)

    def forward(self, x):
        # encoder
        out = self.layer2(self.layer1(x))
        res1 = out
        out = self.layer4(self.layer3(out))
        res2 = out
        # decoder
        if self.sa_flag == True:
            out = self.layer6(self.sa(self.layer5(out)))
        else:
            out = self.layer6(self.layer5(out))
        out = out + res2
        out = self.layer8(self.layer7(out))
        out = out + res1
        out = self.layer10(self.layer9(out))
        return out

class UNET_EJNMMI(nn.Module):
    # Ref: Artificial intelligence‐based PET denoising could allow a two‐fold reduction in [18F]FDG PET acquisition time in digital PET/CT
    # Conv2d(stride=2) + ConvTranspose2d(stride=2)
    def __init__(self):
        super(UNET_EJNMMI, self).__init__()
        print('UNET_EJNMMI: Unet architecture in Radiology paper')
        
        self.kernel_size = 3
        self.padding = 1
        self.acti = 'relu'
        self.bn_flag = True

        # encoder
        self.layer1 = conv_block('conv', 1, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer2 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=16, out_channels=16, acti=self.acti)
        self.layer3 = conv_block('conv', 16, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer4 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=32, out_channels=32, acti=self.acti)
        self.layer5 = conv_block('conv', 32, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer6 = down_sampling('conv', self.kernel_size, 2, self.padding, in_channels=64, out_channels=128, acti=self.acti)
        # decoder
        self.layer7 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=128, out_channels=64, acti=self.acti)
        self.layer8 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer9 = conv_block('conv', 64, 64, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer10 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=64, out_channels=32, acti=self.acti)
        self.layer11 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer12 = conv_block('conv', 32, 32, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer13 = up_sampling('trans', self.kernel_size, 2, self.padding, in_channels=32, out_channels=16, acti=self.acti)    
        self.layer14 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer15 = conv_block('conv', 16, 16, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)
        self.layer16 = conv_block('conv', 16, 1, self.kernel_size, 1, self.padding, self.acti, self.bn_flag)

    def forward(self, x):
        out = self.layer1(x)
        res1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        res2 = out
        out = self.layer4(out)
        out = self.layer5(out)
        res3 = out
        out = self.layer6(out) 
        out = self.layer7(out) 
        out = out + res3
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = out + res2
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out + res1
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        return out