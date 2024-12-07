import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class Court_TN(nn.Module):
    def __init__(self, input_size=1, output_size=15):
        super(Court_TN, self).__init__()

        input_layers = input_size * 3

        # Define layers
        self.conv1 = ConvBlock(input_layers, 64)
        self.conv2 = ConvBlock(64, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = ConvBlock(512 + 256, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = ConvBlock(256 + 128, 128)
        self.conv15 = ConvBlock(128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv16 = ConvBlock(128 + 64, 64)
        self.conv17 = ConvBlock(64, 64)

        self.conv18 = nn.Conv2d(64, output_size, kernel_size=1, padding=0)

        self._init_weights()
    

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        x1 = self.conv1(x)
        
        x1 = self.conv2(x1)
        x2 = self.pool1(x1)
    
        x2 = self.conv3(x2)
    
        x2 = self.conv4(x2)
        x3 = self.pool2(x2)
    
        x3 = self.conv5(x3)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x4 = self.pool3(x3)
    
        x4 = self.conv8(x4)
        x4 = self.conv9(x4)
        x4 = self.conv10(x4)
    
        # Upsampling path
        x5 = self.up1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.conv11(x5)
        x5 = self.conv12(x5)
        x5 = self.conv13(x5)
    
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.conv14(x6)
        x6 = self.conv15(x6)
    
        x7 = self.up3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.conv16(x7)
        x7 = self.conv17(x7)

        output = torch.sigmoid(self.conv18(x7))
        return output


    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)   