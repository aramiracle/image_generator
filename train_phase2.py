import os
import re
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from utils import *
from data_loader import ImageDataset


train_root_dir = 'data/resized'

# Define hyperparameters
batch_size = 64
learning_rate = 3e-4
epochs = 300

# Initialize the generator and discriminator
generator = PretrainGenerator()
discriminator = Discriminator()

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = ImageDataset(train_root_dir, transform=transform, train=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)
psnr_metric = PeakSignalNoiseRatio()

generated_images_dir = 'generated_images'
shutil.rmtree(generated_images_dir, ignore_errors=True)
os.makedirs(generated_images_dir, exist_ok=True)

# Define the path to the checkpoint directory
checkpoint_dir_phase1 = 'saved_models/phase1/'
checkpoint_dir_phase2 = 'saved_models/phase2/'
os.makedirs(checkpoint_dir_phase2, exist_ok=True)

# Check if the checkpoint directory exists

checkpoint_files_phase1 = [f for f in os.listdir(checkpoint_dir_phase1) if re.match(r'gan_checkpoint_epoch\d+\.pth', f)]
checkpoint_files_phase2 = [f for f in os.listdir(checkpoint_dir_phase2) if re.match(r'gan_checkpoint_epoch\d+\.pth', f)]

if checkpoint_files_phase2:
    # Extract and sort the epoch numbers
    epoch_numbers = [int(re.search(r'gan_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files_phase2]
    epoch_numbers.sort()
    
    # Load the latest checkpoint
    latest_epoch = epoch_numbers[-1]
    latest_checkpoint = f'gan_checkpoint_epoch{latest_epoch:04d}.pth'
    checkpoint_path = os.path.join(checkpoint_dir_phase2, latest_checkpoint)
    
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
    discriminator.train()

    print(f"Phase II - Loaded checkpoint from epoch {epoch}. Resuming training...")

elif checkpoint_files_phase1:
    epoch = 0
    print("Phase II - No checkpoint found in phase 2. Loading phase 1 generator and starting training from epoch 1...")

    
    # Load the latest checkpoint
    best_checkpoint = 'best_gan_checkpoint.pth'
    checkpoint_path = os.path.join(checkpoint_dir_phase1, best_checkpoint)
    
    # Load the generator and discriminator models and their states
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])

    # Make sure to set the mode for generator and discriminator
    generator.train()
    discriminator.train()
else:
    raise Exception('You have to run phase 1 before starting phase 2.')


# Calculate images feature with pretrained models
model_1 = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')
model_2 = models.shufflenet_v2_x2_0(weights='ShuffleNet_V2_X2_0_Weights.DEFAULT')
model_3 = models.regnet_y_1_6gf(weights='RegNet_Y_1_6GF_Weights.DEFAULT')
model_4 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
model_5 = models.mnasnet1_3(weights='MNASNet1_3_Weights.DEFAULT')
model_6 = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
model_7 = models.regnet_x_1_6gf(weights='RegNet_X_1_6GF_Weights.DEFAULT')

best_ssim = 0.0

# Training loop phase II
for epoch in range(epoch, epochs):
    running_psnr = 0.0
    running_ssim = 0.0
    running_lnl1 = 0.0
    running_loss_generator = 0.0
    running_loss_discrimonator = 0.0

    # Wrap train_loader with tqdm for progress bar
    for batch_idx, real_images in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):

        current_batch_size = real_images.shape[0]
        
        f1 = model_1(real_images)
        f2 = model_2(real_images)
        f3 = model_3(real_images)
        f4 = model_4(real_images)
        f5 = model_5(real_images)
        f6 = model_6(real_images)
        f7 = model_7(real_images)

        features = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=1)

        # Train Discriminator
        optimizer_D.zero_grad()
 
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

        # Calculate the loss for real and fake images
        real_labels = torch.ones(current_batch_size, 1)
        fake_labels = torch.zeros(current_batch_size, 1)
        
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)
        
        output_fake = discriminator(fake_images)
        loss_fake = criterion(output_fake, fake_labels)
        
        loss_discriminator = loss_real + loss_fake
        running_loss_discrimonator += loss_discriminator

        loss_discriminator.backward(retain_graph=True)
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        output_fake = discriminator(fake_images)
        bce_loss = criterion(output_fake, real_labels)  # BCELoss for generator
        
        # Calculating custom loss
        custom_loss = PSNR_SSIM_LNL1_loss(fake_images, real_images)
        
        # Combine BCELoss and CustomLoss for the generator
        loss_generator = bce_loss + custom_loss
        running_loss_generator += loss_generator
        
        loss_generator.backward()
        optimizer_G.step()
        
    
    average_psnr = running_psnr / len(train_loader)
    average_ssim = running_ssim / len(train_loader)
    average_lnl1 = running_lnl1 / len(train_loader)
    average_loss_generator = running_loss_generator / len(train_loader)
    average_loss_discriminator = running_loss_discrimonator / len(train_loader)

    print(f"Phase 2 - Epoch [{epoch + 1}/{epochs}] Loss D: {average_loss_discriminator:.4f} Loss G: {average_loss_generator:.4f}")
    print(f'Phase 2 - Epoch [{epoch + 1}/{epochs}] Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} LNL1: {average_lnl1:.4f}')

    # Check the condition and save models if met
    if not((epoch + 1) % 5):
        save_path = f'saved_models/phase2/gan_checkpoint_epoch{epoch + 1:04d}.pth'
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'loss_discriminator': loss_discriminator.item(),
            'loss_generator': loss_generator.item()
        }, save_path)
        print('Phase II - Model is saved.')

    if best_ssim < average_ssim:
        best_ssim = average_ssim
        save_path = 'saved_models/phase2/best_gan_checkpoint.pth'
        torch.save({
            'generator_state_dict': generator.state_dict()
        }, save_path)
        print('$$$ Best model is saved according to SSIM. $$$')