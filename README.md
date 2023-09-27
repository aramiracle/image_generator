# image_generator
Trying to generate images with features adpoted from pretrained models

# Project README

This project comprises a collection of Python scripts for image preprocessing, PCA-based image generation, and image quality evaluation. The primary objective is to transform input images into PCA space, train a generator model to produce enhanced images, and assess the quality of the generated images. Below, we provide an in-depth overview of each script and its functionalities:

## preprocess.py

This script manages the preprocessing of input images. Key functionalities include:

- Resizing: Input images, located in the `data/original` directory, are resized to a specified target size of 32x32 pixels.
- Multi-Processing: To improve efficiency, the resizing process is parallelized using multi-processing.
- Output Directory: If it doesn't exist, the script creates an output directory named `data/resized` and saves the resized images there.

## data_loader.py

This script defines a custom PyTorch dataset class called `ImageDataset`. The dataset class is designed for image enhancement tasks and includes the following features:

- Loading and Transformation: It loads images and applies optional image transformations, such as resizing and data augmentation, depending on the specified parameters.
- Train/Test Split: The dataset can be split into training and testing sets.
- Customization: The dataset allows customization of image transformations and other data loading parameters.

Additionally, the script provides the `get_data_loaders` function to create data loaders for training datasets.

## model.py

The `model.py` script contains the definition of a PyTorch neural network model called `PCAGenerator`. This model is designed for image generation based on PCA-transformed features. Key features of this model include:

- Upsampling Layers: The model employs a series of deconvolutional layers to upsample low-dimensional features to the desired image size (32x32 pixels).

## utils.py

The `utils.py` script serves as a utility module for various functions and metrics used in the project:

- Custom Metrics: It defines a custom metric `lnl1_metric` for LNL1 calculation and a combined loss function `PSNR_SSIM_LNL1_loss` that incorporates PSNR, SSIM, and LNL1 metrics for image quality assessment.
- Image Generation: The script includes an `image_generator` function for generating images using a trained generator model and saving the results.
- Data Loading: The `loading_images` function is responsible for loading and preprocessing images, as well as splitting them into training and testing sets.
- Model Training: It contains the `train_pca_generator` function for training the PCA-based generator model with provided data.
- Feature Calculation: The `calculate_embeddings` function computes image embeddings and applies PCA for dimensionality reduction.
- Feature Extraction: The `calculate_features` function extracts image features using pretrained models.

## train_pca.py

The `train_pca.py` script serves as the main entry point for the project. It coordinates various aspects of the project:

- Directory Paths: Specifies paths for directories where data, features, models, and results are stored.
- Training Parameters: Allows for customization of batch size, learning rate, and the number of training epochs.
- Data Loading: Uses the functions from `utils.py` to load and preprocess images, as well as calculate image embeddings.
- Model Training: Initiates training of the PCA-based image generator using the specified parameters.
- Image Generation: Utilizes the trained generator model to generate enhanced images and save them in the designated directory.

## Usage

To effectively use this project, follow these steps:

1. Ensure you have all required dependencies installed, including PyTorch, torchvision, PIL, and other libraries used in the scripts.

2. Organize your input images in the `data/original` directory. These are the images that will undergo preprocessing and serve as input for the generator.

3. Run `preprocess.py` to resize the input images to the desired size and save the resized images in the `data/resized` directory.

4. Run `train_pca.py` to perform the following tasks:
   - Load the resized and preprocessed images.
   - Calculate PCA-transformed features for the images.
   - Train the PCA-based image generator.
   - Generate enhanced images using the trained model and save them in the `results/cnn` directory.

5. Examine the generated images in the `results/cnn` directory for image quality assessment or further analysis.

## Note

This project assumes access to a dataset of images organized as specified above. It's essential to adjust the training parameters, model architecture, and preprocessing steps to align with your specific image processing and generation needs.

Although the results are not very good but it was entertaining try to generate image from features extracted from pretrained models.
