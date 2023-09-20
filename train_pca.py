import os
import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from model import *
import re
import shutil
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from data_loader import ImageDataset
import torchvision.models as models
import torch.nn.functional as F


train_root_dir = 'data/resized'

transform = transforms.Compose([transforms.ToTensor()])

image_names = os.listdir(train_root_dir)

# Calculate images feature with pretrained models
model_1 = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT').eval()
model_2 = models.shufflenet_v2_x2_0(weights='ShuffleNet_V2_X2_0_Weights.DEFAULT').eval()
model_3 = models.regnet_y_1_6gf(weights='RegNet_Y_1_6GF_Weights.DEFAULT').eval()
model_4 = models.densenet121(weights='DenseNet121_Weights.DEFAULT').eval()
model_5 = models.mnasnet1_3(weights='MNASNet1_3_Weights.DEFAULT').eval()
model_6 = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT').eval()
model_7 = models.regnet_x_1_6gf(weights='RegNet_X_1_6GF_Weights.DEFAULT').eval()

print('Starting calculating features and transform it.')

features_list = []
for image_name in tqdm(image_names):
    image = Image.open(f'{train_root_dir}/{image_name}').convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    f1 = model_1(image_tensor)
    f2 = model_2(image_tensor)
    f3 = model_3(image_tensor)
    f4 = model_4(image_tensor)
    f5 = model_5(image_tensor)
    f6 = model_6(image_tensor)
    f7 = model_7(image_tensor)

    feature = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)
    features_list.append(feature)

print('Features calculated.')

features_arr = torch.cat(features_list, dim=0).detach().numpy()
pca = PCA(n_components=96)
X_pca = torch.tensor(pca.fit_transform(features_arr))
X_pca_normalized = normalize(X_pca)

#Training

# Define hyperparameters
batch_size = 64
learning_rate = 3e-3
epochs = 300

train_dataset = ImageDataset(train_root_dir, transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


generated_images_dir = 'generated_images'
shutil.rmtree(generated_images_dir, ignore_errors=True)
os.makedirs(generated_images_dir, exist_ok=True)

# Define the path to the checkpoint directory
checkpoint_dir = 'saved_models/phase1/'
os.makedirs(checkpoint_dir, exist_ok=True)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
psnr_metric = PeakSignalNoiseRatio()

generator = PCAGenerator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = ImageDataset(train_root_dir, transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


generated_images_dir = 'generated_images'
shutil.rmtree(generated_images_dir, ignore_errors=True)
os.makedirs(generated_images_dir, exist_ok=True)

# Define the path to the checkpoint directory
checkpoint_dir = 'saved_models/pca/'
os.makedirs(checkpoint_dir, exist_ok=True)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
psnr_metric = PeakSignalNoiseRatio()



# List all checkpoint files in the directory
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'gan_checkpoint_epoch\d+\.pth', f)]

if checkpoint_files:
    # Extract and sort the epoch numbers
    epoch_numbers = [int(re.search(r'gan_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files]
    epoch_numbers.sort()
    
    # Load the latest checkpoint
    latest_epoch = epoch_numbers[-1]
    latest_checkpoint = f'gan_checkpoint_epoch{latest_epoch:04d}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    # Load the generator and discriminator models and their states
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    loss_discriminator = checkpoint['loss_discriminator']
    loss_generator = checkpoint['loss_generator']

    # Make sure to set the mode for generator and discriminator
    generator.train()

    print(f"PCA - Loaded checkpoint from epoch {epoch}. Resuming training...")
else:
    epoch = 0
    print("PCA - No checkpoint found. Starting training from epoch 1...")


best_ssim = 0

# Training loop phase I
for epoch in range(epoch, epochs):
    running_psnr = 0.0
    running_ssim = 0.0
    running_lnl1 = 0.0
    running_loss = 0.0

    # Wrap train_loader with tqdm for progress bar
    for batch_idx, real_images in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):

        current_batch_size = real_images.shape[0]
        
        features = X_pca_normalized[batch_idx*batch_size:batch_idx*batch_size + current_batch_size]

        # Generate fake images from the generator
        fake_images = generator(features)
        if not((batch_idx + 1)%4):
            save_image(torch.cat((fake_images, real_images), dim=0), f'{generated_images_dir}/fake_image_epoch{epoch + 1:04d}_batch{batch_idx + 1:02d}.png')

        # Calculate PSNR for this batch and accumulate
        psnr = psnr_metric(fake_images, real_images)
        running_psnr += psnr
    
        # Calculate SSIM for this batch and accumulate
        ssim_value = ssim_metric(fake_images, real_images)
        running_ssim += ssim_value

        # Calculate LNL1 for this batch and accumulate
        lnl1_value = lnl1_metric(fake_images, real_images)
        running_lnl1 += lnl1_value

        
        # Train Generator
        optimizer_G.zero_grad()
        
        # Calculating custom loss
        loss = PSNR_SSIM_LNL1_loss(fake_images, real_images)
        running_loss += loss

        loss.backward()
        optimizer_G.step()
        
    
    average_psnr = running_psnr / len(train_loader)
    average_ssim = running_ssim / len(train_loader)
    average_lnl1 = running_lnl1 / len(train_loader)
    average_loss = running_loss / len(train_loader)

    print(f"PCA - Epoch [{epoch + 1}/{epochs}] Loss G: {average_loss:.4f}")
    print(f'PCA - Epoch [{epoch + 1}/{epochs}] Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} LNL1: {average_lnl1:.4f}')

    # Check the condition and save models if met
    if not((epoch + 1) % 5):
        save_path = f'saved_models/phase1/gan_checkpoint_epoch{epoch + 1:04d}.pth'
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
        }, save_path)
        print('Phase I - Model is saved.')

    if best_ssim < average_ssim:
        best_ssim = average_ssim
        save_path = 'saved_models/phase1/best_gan_checkpoint.pth'
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict()
        }, save_path)
        print('$$$ Best model is saved according to SSIM. $$$')


