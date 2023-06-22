import torch.nn as nn
import torch
import torchvision.transforms.functional as TF
import math


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        skip = self.conv(x)
        pool = self.pool(skip)

        return pool, skip


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder1 = EncoderBlock(channels[0], channels[1])
        self.encoder2 = EncoderBlock(channels[1], channels[2])
        self.encoder3 = EncoderBlock(channels[2], channels[3])
        self.encoder4 = EncoderBlock(channels[3], channels[4])

        self.bottleneck = ConvBlock(channels[4], channels[5])

    def forward(self, x):
        skips = [None] * 4
        x, skips[0] = self.encoder1(x)
        x, skips[1] = self.encoder2(x)
        x, skips[2] = self.encoder3(x)
        x, skips[3] = self.encoder4(x)

        x = self.bottleneck(x)

        return x, skips


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = self.crop(x, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.conv(x)

        return x

    def crop(self, x, skip):
        return TF.resize(x, size=skip.shape[2:], interpolation=2)


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.decoder1 = DecoderBlock(channels[0], channels[1])
        self.decoder2 = DecoderBlock(channels[1], channels[2])
        self.decoder3 = DecoderBlock(channels[2], channels[3])
        self.decoder4 = DecoderBlock(channels[3], channels[4])

    def forward(self, x, skips):
        x = self.decoder1(x, skips[3])
        x = self.decoder2(x, skips[2])
        x = self.decoder3(x, skips[1])
        x = self.decoder4(x, skips[0])

        return x


class UNet(nn.Module):
    def __init__(self, encoder_channels=(3, 64, 128, 256, 512, 1024), decoder_channels=(1024, 512, 256, 128, 64),
                 n_classes=1):
        super().__init__()
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.classifier = nn.Conv2d(in_channels=decoder_channels[-1], out_channels=n_classes, kernel_size=(1, 1))

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        out = self.classifier(x)
        out = torch.sigmoid(out)

        return out


class ConvModel(nn.Module):
    def __init__(self, unet, kernel_size):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=kernel_size, stride=1, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.unet = unet

    def forward(self, x):
        img = self.conv1(x)
        img = self.relu(img)
        img = self.conv2(img)
        img = self.relu(img)
        out = self.unet(img)
        return out


class Autoencoder(nn.Module):
    def __init__(self, input_size, depth, bottleneck_size, kernel_size, output_size):
        super(Autoencoder, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_size = input_size
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3

        for layer in range(depth):
            out_size = int(bottleneck_size / math.pow(2, (depth - (layer + 1))))
            self.encoder.extend([nn.Conv2d(in_size, out_size, kernel_size=kernel_size, padding=padding, stride=2, bias=False)])
            in_size = out_size

        out_size = output_size
        for layer in range(depth):
            in_size = int(bottleneck_size / math.pow(2, (depth - (layer + 1))))
            self.decoder.extend([nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, padding=padding, output_padding=1, stride=2, bias=False)])
            out_size = in_size

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
            x = self.relu(x)

        for layer in reversed(self.decoder):
            x = layer(x)
            x = self.relu(x)

        return x


class AutoencoderModel(nn.Module):
    def __init__(self, autoencoder, unet):
        super(AutoencoderModel, self).__init__()
        self.autoencoder = autoencoder
        self.unet = unet

    def forward(self, x):
        x = self.autoencoder(x)
        out = self.unet(x)
        return out