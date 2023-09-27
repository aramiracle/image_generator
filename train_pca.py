from model import *
from utils import *

if __name__ == "__main__":
    # Directory paths
    root_dir = 'data/resized'
    features_dir = 'features'
    model_dir = 'saved_models/pca'
    results_dir = 'results/cnn'

    # Training parameters
    batch_size = 64
    learning_rate = 5e-3
    epochs = 5000

    # Load data
    train_loader, test_loader = loading_images(root_dir, batch_size)

    # Calculate embeddings for train and test data
    X_pca_normalized_train, X_pca_normalized_test = calculate_embeddings(train_loader, test_loader, features_dir)

    # Train the PCA-based generator
    train_pca_generator(train_loader, X_pca_normalized_train, batch_size, learning_rate, epochs)

    # Generate images using the trained model
    image_generator(test_loader, X_pca_normalized_test, model_dir, results_dir, alpha=0.2)
