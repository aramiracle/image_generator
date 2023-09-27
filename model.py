import torch
import torch.nn as nn

class PCAGenerator(nn.Module):
    def __init__(self):
        super(PCAGenerator, self).__init__()
        
        # Define additional layers for upsampling to reach 50x50 resolution
        self.deconv1 = self.deconv_block(320, 160)
        self.deconv2 = self.deconv_block(160, 80)
        self.deconv3 = self.deconv_block(80, 40)
        self.deconv4 = self.deconv_block(40, 20)
        self.deconv5 = self.deconv_block(20, 3)
        
    def deconv_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        x0 = x.unsqueeze(2).unsqueeze(3)  # Input: 320x1x1
        x1 = self.deconv1(x0)  # Output: 160x2x2
        x2 = self.deconv2(x1)  # Output: 80x4x4
        x3 = self.deconv3(x2)  # Output: 40x8x8
        x4 = self.deconv4(x3)  # Output: 20x16x16
        x5 = self.deconv5(x4)  # Output: 3x32x32
        output = torch.sigmoid(x5)
        return output