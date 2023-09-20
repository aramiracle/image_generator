from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch

# Define a custom dataset class for image enhancement tasks.
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Initialize the ImageEnhancementDataset.

        Args:
            input_root_dir (str): Path to the directory containing input images.
            output_root_dir (str): Path to the directory containing output (target) images.
            transform (callable, optional): Optional image transformations to apply.
            train (bool, optional): Set to True for training data; applies data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # List all input and output image file paths
        self.image_paths = [os.path.join(root_dir, fname) for fname in sorted(os.listdir(root_dir))]
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def apply_data_augmentation(self, input_image):
        """
        Apply data augmentation to input and output images.

        Args:
            input_image (PIL.Image): Input image.
            output_image (PIL.Image): Output (target) image.

        Returns:
            PIL.Image: Augmented input image.
            PIL.Image: Augmented output (target) image.
        """

        if self.train:

            # Color Jitter
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            input_image = color_jitter(input_image)

        return input_image

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            PIL.Image: Augmented input image.
            PIL.Image: Augmented output (target) image.
        """
        input_img_path = self.image_paths[idx]

        input_image = self.transform(Image.open(input_img_path).convert('RGB'))

        # input_image = self.apply_data_augmentation(input_image)

        return input_image

def get_data_loaders(train_root_dir, batch_size):
    """
    Create data loaders for training and testing datasets.

    Args:
        train_input_root_dir (str): Path to the directory containing training input images.
        train_output_root_dir (str): Path to the directory containing training output images.
        batch_size (int): Batch size for data loaders.

    Returns:
        DataLoader: Training data loader.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = ImageDataset(train_root_dir, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
