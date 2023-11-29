import json
import os
import time
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv, dotenv_values
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import rasterio as rio
import itertools
from itertools import product
import utils
import torch.nn as nn
import torch.nn.functional as F
from focal_utils import get_weights

# BUILDING UNET MODEl (Source: https://www.classcentral.com/course/youtube-pytorch-image-segmentation-tutorial-with-u-net-everything-from-scratch-baby-126811)
class DoubleConv(nn.Module):
    #def __init__(self, in_channels, out_channels,p):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #nn.Dropout(p),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            #nn.Dropout(p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),#inplace=True),
            #nn.Dropout(p),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            #nn.Dropout(p),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),#inplace=True,
            #nn.Dropout(p)
            )#)

    def forward(self, x):
        ###x = self.dropout(x)
        return self.conv(x)

class UNET(nn.Module):
    #def __init__(self, in_channels=3, out_channels=1,features = [32 , 64, 128, 256], p=0.2):#features=[64, 128, 256, 512]):
    def __init__(self, in_channels=3, out_channels=1,features = [32 , 64, 128, 256]):#features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            #self.downs.append(DoubleConv(in_channels, feature,p))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2,))
            self.ups.append(DoubleConv(feature*2, feature))
            #self.ups.append(DoubleConv(feature*2, feature,p))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #self.bottleneck = DoubleConv(features[-1], features[-1]*2,p)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return torch.sigmoid(self.final_conv(x))
