import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))
    
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = ConvBlock(1, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        self.down_conv5 = ConvBlock(512, 1024)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
    
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)

        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))

        out = self.final(up4)
        
        return out