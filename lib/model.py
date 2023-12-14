import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(Conv3DBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.pooling:
            pooled = self.pooling(x)
            return pooled, x
        return x, x

class UpConv3DBlock(nn.Module):
    def __init__(self, in_channels, res_channels=0, last_layer=False, 
                 num_classes=None):
        super(UpConv3DBlock, self).__init__()
        mid_channels = in_channels // 2
        self.upconv = nn.ConvTranspose3d(in_channels, mid_channels, 
                                         kernel_size=2, stride=2)
        self.conv1 = nn.Conv3d(mid_channels + res_channels, mid_channels, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        if last_layer:
            self.final_conv = nn.Conv3d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x, residual=None):
        x = self.upconv(x)
        if residual is not None:
            x = torch.cat((x, residual), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if hasattr(self, 'final_conv'):
            x = self.final_conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], 
                 bottleneck_channel=512):
        super(UNet3D, self).__init__()
        self.down_blocks = nn.ModuleList([
            Conv3DBlock(in_channels, level_channels[0]),
            Conv3DBlock(level_channels[0], level_channels[1]),
            Conv3DBlock(level_channels[1], level_channels[2])
        ])
        self.bottleNeck = Conv3DBlock(level_channels[2], bottleneck_channel, 
                                      bottleneck=True)
        self.up_blocks = nn.ModuleList([
            UpConv3DBlock(bottleneck_channel, level_channels[2]),
            UpConv3DBlock(level_channels[2], level_channels[1]),
            UpConv3DBlock(level_channels[1], level_channels[0], last_layer=True, 
                          num_classes=num_classes)
        ])
    
    def forward(self, x):
        residuals = []
        for block in self.down_blocks:
            x, res = block(x)
            residuals.append(res)
        x, _ = self.bottleNeck(x)
        for block, res in zip(reversed(self.up_blocks), reversed(residuals)):
            x = block(x, res)
        return x
