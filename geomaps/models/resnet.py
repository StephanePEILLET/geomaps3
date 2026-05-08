import torch
import torch.nn as nn
import torch.nn.functional as F


def add_block_layers(in_ch, bloc_type, n_blocs):
    layers = []
    for i in range(int(n_blocs)):
        layers.append(bloc_type(in_ch))

    return nn.Sequential(*layers)


class InputConv(nn.Module):
    def __init__(self, in_ch, out_ch, bloc_type, n_blocs):
        super(InputConv, self).__init__()
        self.inc = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.blocs = add_block_layers(out_ch, bloc_type, n_blocs)

    def forward(self, x):
        x = self.inc(x)
        x = self.blocs(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_ch):
        super(BasicBlock, self).__init__()
        self.building_block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.building_block(x) + x
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_ch):
        super(Bottleneck, self).__init__()
        inter_ch = in_ch // 4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, 1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, inter_ch, 3, padding=1),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bottleneck(x) + x
        x = self.relu(x)
        return x


class ResEncoder(nn.Module):
    """

    Parameters
    ---------
    in_ch : int
        number of input channels
    out_ch : int
        number of output classes
    block_type: class
        class of residual bloc to implement : BasicBlock or Bottleneck
    n_blocks : int
        number of residual blocs to put in the encoder
    """

    def __init__(self, in_ch, out_ch, block_type, n_blocks):
        super(ResEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.blocs = add_block_layers(out_ch, block_type, n_blocks)

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocs(x)
        return x


class ResDecoder(nn.Module):
    """

    Parameters
    ---------
    in_ch : int
        number of input channels
    out_ch : int
        number of output classes
    block_type: class
        class of residual bloc to implement : BasicBlock or Bottleneck
    n_blocks : int
        number of residual blocs to put in the decode
    """

    def __init__(self, in_ch, out_ch, block_type, n_blocks, bilinear=True):
        super(ResDecoder, self).__init__()
        self.bilinear = bilinear

        if self.bilinear:
            self.up = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.blocs = add_block_layers(out_ch, block_type, n_blocks)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=True)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        x = self.blocs(x)
        return x


class ResBlockNet(nn.Module):
    """ResBlokNet = U-Net with reduced number of filters and residual blocs

    Parameters
    ----------
    in_channels : int
        number of input channels
    classes : int
        number of output classes
    model_name : string
        name of the model with architecture to apply

    model_name should be written as follows:
    'rnet_' + first layer size + '_' + bloc_type (basicblock or bottleneck) +\
    '_' + number of blocs per encoder and decoder
    Example : 'rnet_8_basicbloc_8' : there will be 8 basicblocs at each encoder / decoder\
    and the width of U-Net will start at 8 until 128
    """

    def __init__(self, in_channels, classes, model_name, bilinear=True):
        super(ResBlockNet, self).__init__()
        if "basicblock" in model_name:
            block_type = BasicBlock
        elif "bottleneck" in model_name:
            block_type = Bottleneck

        n_blocks = int(model_name[-1])
        self.bilinear = bilinear

        if model_name.split("_")[1] == "8":
            layer_size = [8, 16, 32, 64, 128]
        elif model_name.split("_")[1] == "16":
            layer_size = [16, 32, 64, 128, 256]
        elif model_name.split("_")[1] == "32":
            layer_size = [32, 64, 128, 256, 512]
        elif model_name.split("_")[1] == "64":
            layer_size = [64, 128, 256, 512, 1024]

        # first layer
        self.inc = InputConv(in_channels, layer_size[0], block_type, n_blocks)

        # encoder
        self.down1 = ResEncoder(layer_size[0], layer_size[1], block_type, n_blocks)
        self.down2 = ResEncoder(layer_size[1], layer_size[2], block_type, n_blocks)
        self.down3 = ResEncoder(layer_size[2], layer_size[3], block_type, n_blocks)
        self.down4 = ResEncoder(layer_size[3], layer_size[4], block_type, n_blocks)

        # decoder
        self.up1 = ResDecoder(
            layer_size[4], layer_size[3], block_type, n_blocks, bilinear=self.bilinear
        )
        self.up2 = ResDecoder(
            layer_size[3], layer_size[2], block_type, n_blocks, bilinear=self.bilinear
        )
        self.up3 = ResDecoder(
            layer_size[2], layer_size[1], block_type, n_blocks, bilinear=self.bilinear
        )
        self.up4 = ResDecoder(
            layer_size[1], layer_size[0], block_type, n_blocks, bilinear=self.bilinear
        )

        # last layer
        self.outc = nn.Conv2d(layer_size[0], classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
