# Code adapted from: https://github.com/vinceecws/SegNet_PyTorch/blob/master/Pavements/SegNet.py#L7
import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):

    #def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):#, pdo=0.2):
        super(SegNet, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        #self.dropout10 = nn.Dropout(pdo)
        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        #self.dropout11 = nn.Dropout(pdo)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.dropout12 = nn.Dropout(pdo)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        #self.dropout21 = nn.Dropout(pdo)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #self.dropout22 = nn.Dropout(pdo)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        #self.dropout31 = nn.Dropout(pdo)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.dropout32 = nn.Dropout(pdo)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.dropout33 = nn.Dropout(pdo)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        #self.dropout41 = nn.Dropout(pdo)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout42 = nn.Dropout(pdo)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout43 = nn.Dropout(pdo)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout51 = nn.Dropout(pdo)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout52 = nn.Dropout(pdo)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout53 = nn.Dropout(pdo)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)


        #DECODING consists of 5 stages
        #Each stage corresponds to their respective counterparts in ENCODING


        #General Max Pool 2D/Upsampling for DECODING layers
        self.MaxDe = nn.MaxUnpool2d(2, stride=2)
        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout53 = nn.Dropout(pdo)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout52 = nn.Dropout(pdo)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout51 = nn.Dropout(pdo)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout43 = nn.Dropout(pdo)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #self.dropout42 = nn.Dropout(pdo)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        #self.dropout41 = nn.Dropout(pdo)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.dropout33 = nn.Dropout(pdo)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #self.dropout32 = nn.Dropout(pdo)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        #self.dropout31 = nn.Dropout(pdo)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        #self.dropout22 = nn.Dropout(pdo)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        #self.dropout21 = nn.Dropout(pdo)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #self.dropout12 = nn.Dropout(pdo)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        #self.dropout11 = nn.Dropout(pdo)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

    def forward(self, x):

        #ENCODE LAYERS
        #Stage 1
        #x = self.dropout10(x)
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        #x = self.dropout11(x)
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        #x = self.dropout12(x)
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        #x = self.dropout21(x)
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        #x = self.dropout22(x)
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        #x = self.dropout31(x)
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        #x = self.dropout32(x)
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        #x = self.dropout33(x)
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        #x = self.dropout41(x)
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        #x = self.dropout42(x)
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        #x = self.dropout43(x)
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        #x = self.dropout51(x)
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        #x = self.dropout52(x)
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        #x = self.dropout53(x)
        x, ind5 = self.MaxEn(x)
        size5 = x.size()

        #DECODE LAYERS
        #Stage 5
        x = self.MaxDe(x, ind5, output_size=size4)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        #x = self.dropout53(x)
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        #x = self.dropout52(x)
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        #x = self.dropout51(x)

        #Stage 4
        x = self.MaxDe(x, ind4, output_size=size3)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        #x = self.dropout43(x)
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        #x = self.dropout42(x)
        x = F.relu(self.BNDe41(self.ConvDe41(x)))
        #x = self.dropout41(x)

        #Stage 3
        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        #x = self.dropout33(x)
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        #x = self.dropout32(x)
        x = F.relu(self.BNDe31(self.ConvDe31(x)))
        #x = self.dropout31(x)

        #Stage 2
        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        #x = self.dropout22(x)
        x = F.relu(self.BNDe21(self.ConvDe21(x)))
        #x = self.dropout21(x)

        #Stage 1
        x = self.MaxDe(x, ind1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        #x = self.dropout12(x)
        x = self.ConvDe11(x)
        #x = self.dropout11(x)

        ###x = F.softmax(x, dim=1)
        x = F.sigmoid(x)

        return x
