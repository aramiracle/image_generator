import torch.nn as nn
import torchvision.models as models


# Define the Generator network
class PretrainGenerator(nn.Module):
    def __init__(self):
        super(PretrainGenerator, self).__init__()

        # Load the pre-trained ResNet-50 model
        efficientnet = models.efficientnet_b3(weights='EfficientNet_B3_Weights.DEFAULT')
        
        # Remove the fully connected layers at the end
        self.efficientnet_features = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Define additional layers for upsampling to reach 50x50 resolution
        self.fc = nn.Linear(6000, 128)
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        
    def forward(self, x):
        x0 = self.fc(x).unsqueeze(2).unsqueeze(3) #128x1x1
        x1 = self.deconv1(x0) #64x2x2
        x2 = self.deconv2(x1) #32x4x4
        x3 = self.deconv3(x2) #16x8x8
        x4 = self.deconv4(x3) #8x16x16
        output = self.deconv5(x4) #3x32x32

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
            nn.Dropout(0.5),
            nn.Sigmoid()  # Sigmoid activation to squash the output to [0, 1]
        )

    def forward(self, x):
        return self.model(x)

class FeatureOpimizer(nn.Module):
    def __init__(self):
        super(FeatureOpimizer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(6000, 1000),
            nn.Linear(1000, 6000)
        )

    def forward(self, x):
        return self.fc(x)