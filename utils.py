import torch
import math
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import os
from model import *  # Import custom model
from torchvision import transforms
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
from data_loader import ImageDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import re
import shutil
import torch.optim as optim
from sklearn.decomposition import PCA
from torchvision.utils import save_image
from tqdm import tqdm

# Custom metric for LNL1 calculation
def lnl1_metric(prediction, target):
    max = torch.maximum(prediction, target) + 1e-3 * torch.ones_like(prediction)
    norm1 = torch.absolute(prediction - target)
    normalize_norm1 = torch.mean(torch.mul(norm1, max.pow(-1)))
    lnl1_value = -20*torch.log10(normalize_norm1)
    return lnl1_value

# Combined loss function incorporating PSNR, SSIM, and LNL1
def PSNR_SSIM_LNL1_loss(prediction, target):
    psnr = PeakSignalNoiseRatio()
    psnr_value = psnr(prediction, target)
    psnr_loss = -psnr_value

    ssim = StructuralSimilarityIndexMeasure(data_range=1)
    ssim_value = ssim(prediction, target)
    ssim_loss = torch.tan(math.pi / 2 * (1 - (1 + ssim_value) / 2))

    lnl1_loss = -lnl1_metric(prediction, target)

    return psnr_loss + 50 * ssim_loss + lnl1_loss

# Function to generate images using a trained generator
def image_generator(test_loader, test_embeddings, model_dir, results_dir, alpha=0.1):
    for i, images in enumerate(test_loader):
        if not(i):
            test_images = images
        else:
            test_images = torch.cat(test_images, images)

    generator = PCAGenerator()  # Create an instance of the generator model

    os.makedirs(results_dir, exist_ok=True)

    best_checkpoint_path = f'{model_dir}/best_model_checkpoint.pth'
    checkpoint = torch.load(best_checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])  # Load the generator's weights

    generated_images = generator(test_embeddings)  # Generate images using the loaded generator
    noise = torch.rand_like(test_embeddings)
    generated_noisy_images = generator(test_embeddings + alpha * noise)
    save_image(test_images, f'{results_dir}/test_images.png')
    save_image(generated_images, f'{results_dir}/generated_images.png')  # Save generated images
    save_image(generated_noisy_images, f'{results_dir}/generated_noisy_images.png')
    
# Function to load and preprocess images
def loading_images(root_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageDataset(root_dir, transform=transform, train=True)  # Create a custom dataset for loading images
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.05, shuffle=False)  # Split the dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to train a PCA-based generator
def train_pca_generator(train_loader, transformed_features, batch_size, learning_rate, epochs):
    generator = PCAGenerator()  # Create an instance of the generator model
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)  # Use Adam optimizer for training

    generated_images_dir = 'generated_images'
    shutil.rmtree(generated_images_dir, ignore_errors=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    checkpoint_dir = 'saved_models/pca/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1)  # SSIM metric for evaluation
    psnr_metric = PeakSignalNoiseRatio()  # PSNR metric for evaluation

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'model_checkpoint_epoch\d+\.pth', f)]

    if checkpoint_files:
        epoch_numbers = [int(re.search(r'model_checkpoint_epoch(\d+)\.pth', f).group(1)) for f in checkpoint_files]
        epoch_numbers.sort()

        latest_epoch = epoch_numbers[-1]
        latest_checkpoint = f'model_checkpoint_epoch{latest_epoch:04d}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['generator_state_dict'])  # Load the generator's weights
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])

        generator.train()

        print(f"PCA - Loaded checkpoint from epoch {epoch}. Resuming training...")
    else:
        epoch = 0
        print("PCA - No checkpoint found. Starting training from epoch 1...")

    best_ssim = 0

    for epoch in range(epoch, epochs):
        running_psnr = 0.0
        running_ssim = 0.0
        running_lnl1 = 0.0
        running_loss = 0.0

        for batch_idx, real_images in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):

            current_batch_size = real_images.shape[0]

            features = transformed_features[batch_idx*batch_size:batch_idx*batch_size + current_batch_size]

            fake_images = generator(features)  # Generate fake images using the generator
            if not((batch_idx + 1)%4):
                save_image(torch.cat((fake_images, real_images), dim=0), f'{generated_images_dir}/fake_image_epoch{epoch + 1:04d}_batch{batch_idx + 1:02d}.png')

            psnr = psnr_metric(fake_images, real_images)  # Calculate PSNR for evaluation
            running_psnr += psnr

            ssim_value = ssim_metric(fake_images, real_images)  # Calculate SSIM for evaluation
            running_ssim += ssim_value

            lnl1_value = lnl1_metric(fake_images, real_images)  # Calculate LNL1 metric
            running_lnl1 += lnl1_value

            optimizer_G.zero_grad()

            loss = PSNR_SSIM_LNL1_loss(fake_images, real_images)  # Calculate the combined loss
            running_loss += loss

            loss.backward()
            optimizer_G.step()

        average_psnr = running_psnr / len(train_loader)
        average_ssim = running_ssim / len(train_loader)
        average_lnl1 = running_lnl1 / len(train_loader)
        average_loss = running_loss / len(train_loader)

        print(f"PCA - Epoch [{epoch + 1}/{epochs}] Loss G: {average_loss:.4f}")
        print(f'PCA - Epoch [{epoch + 1}/{epochs}] Metrics: PSNR: {average_psnr:.4f} SSIM: {average_ssim:.4f} LNL1: {average_lnl1:.4f}')

        if not((epoch + 1) % 5):
            save_path = f'{checkpoint_dir}/model_checkpoint_epoch{epoch + 1:04d}.pth'
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict()
            }, save_path)
            print('PCA - Model is saved.')

        if best_ssim < average_ssim:
            best_ssim = average_ssim
            save_path = f'{checkpoint_dir}/best_model_checkpoint.pth'
            torch.save({
                'generator_state_dict': generator.state_dict()
            }, save_path)
            print('$$$ Best model is saved according to SSIM. $$$')

# Function to calculate embeddings and apply PCA
def calculate_embeddings(train_loader, test_loader, features_dir):
    
    print('Starting calculating features and transform it with PCA if it is not saved.')
    os.makedirs(features_dir, exist_ok=True)
    if not(os.listdir(features_dir)):
        train_features = calculate_features(train_loader).detach().numpy()
        test_features = calculate_features(test_loader).detach().numpy()
        print('Features calculated.')
        
        mean_list = []
        std_list = []
        for i in range(train_features.shape[0]):
            mean = np.mean(train_features[i, ...])
            std = np.std(train_features[i, ...])
            train_features[i, ...] = (train_features[i, ...] - mean)/std
            mean_list.append(mean)
            std_list.append(std)
        
        pca = PCA(n_components=320)  # Create a PCA object with 256 components
        X_pca_train = torch.tensor(pca.fit_transform(train_features))  # Fit and transform PCA on training features
        print(f'Explained variance: {pca.explained_variance_ratio_.sum()}')
        
        min = X_pca_train.min()
        max = X_pca_train.max()
        X_pca_normalized_train = (X_pca_train - min)/(max - min)  # Normalize PCA-transformed features
        
        for i in range(test_features.shape[0]):
            test_features[i, ...] = (test_features[i, ...] - mean_list[i])/std_list[i]

        X_pca_test = torch.tensor(pca.transform(test_features))  # Transform test features using the same PCA
        X_pca_normalized_test = (X_pca_test - min)/(max - min)  # Normalize test features    

        torch.save(X_pca_normalized_train, f'{features_dir}/transformed_features_train.pt')  # Save transformed training features
        torch.save(X_pca_normalized_test, f'{features_dir}/transformed_features_test.pt')  # Save transformed test features
        print('Transformed features are saved.')

    else:
        X_pca_normalized_train = torch.load(f'{features_dir}/transformed_features_train.pt')  # Load transformed training features
        X_pca_normalized_test = torch.load(f'{features_dir}/transformed_features_test.pt')  # Load transformed test features

    return X_pca_normalized_train, X_pca_normalized_test

# Function to calculate image features using pretrained models
def calculate_features(loader):

    # Create instances of various pretrained models for feature extraction
    model_1 = models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.DEFAULT').eval()
    model_2 = models.shufflenet_v2_x2_0(weights='ShuffleNet_V2_X2_0_Weights.DEFAULT').eval()
    model_3 = models.regnet_y_3_2gf(weights='RegNet_Y_3_2GF_Weights.DEFAULT').eval()
    model_4 = models.densenet201(weights='DenseNet201_Weights.DEFAULT').eval()
    model_5 = models.mnasnet1_3(weights='MNASNet1_3_Weights.DEFAULT').eval()
    model_6 = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT').eval()
    model_7 = models.regnet_x_3_2gf(weights='RegNet_X_3_2GF_Weights.DEFAULT').eval()
    model_8 = models.swin_v2_t(weights='Swin_V2_T_Weights.DEFAULT').eval()
    model_9 = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.DEFAULT').eval()

    features_list = []
    for images in tqdm(loader):

        f1 = model_1(images)  # Extract features using model 1
        f2 = model_2(images)  # Extract features using model 2
        f3 = model_3(images)  # Extract features using model 3
        f4 = model_4(images)  # Extract features using model 4
        f5 = model_5(images)  # Extract features using model 5
        f6 = model_6(images)  # Extract features using model 6
        f7 = model_7(images)  # Extract features using model 7
        f8 = model_8(images)  # Extract features using model 8
        f9 = model_9(images)  # Extract features using model 9

        feature = torch.cat((f1, f2, f3, f4, f5, f6, f7, f8, f9), dim=1)  # Concatenate features from different models
        features_list.append(feature)
    features = torch.cat(features_list, dim=0)  # Concatenate features from all batches
    return features
