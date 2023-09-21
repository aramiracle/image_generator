import torch
import torch.nn as nn
    

class PCAGenerator(nn.Module):
    def __init__(self):
        super(PCAGenerator, self).__init__()
        
        # Define additional layers for upsampling to reach 50x50 resolution

        self.deconv1 = self.deconv_block(256, 128)
        self.deconv2 = self.deconv_block(128, 64)
        self.deconv3 = self.deconv_block(64, 32)
        self.deconv4 = self.deconv_block(32, 16)
        self.deconv5 = self.deconv_block(16, 3)
        
        
    def deconv_block(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        x0 = x.unsqueeze(2).unsqueeze(3) #256x1x1
        x1 = self.deconv1(x0) #128x2x2
        x2 = self.deconv2(x1) #64x4x4
        x3 = self.deconv3(x2) #32x8x8
        x4 = self.deconv4(x3) #16x16x16
        x5 = self.deconv5(x4) #3x32x32
        output = torch.sigmoid(x5)

        return output



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 1),  # Fully connected layer to output a single scalar
            nn.Sigmoid()  # Sigmoid activation to squash the output to [0, 1]
        )

    def forward(self, x):
        return self.model(x)

class FeatureOpimizer(nn.Module):
    def __init__(self):
        super(FeatureOpimizer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128,256),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)