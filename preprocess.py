import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from multiprocessing import Pool

# Define the input and output directories
original_dir = "data/original"
resized_dir = "data/resized"

# Create the output directory if it doesn't exist
os.makedirs(resized_dir, exist_ok=True)

# Define the desired image size
desired_size = (64, 64)

# Define a transformation to resize the images
transform = transforms.Compose([
    transforms.Resize(desired_size),
    transforms.ToTensor(),  # Converts images to PyTorch tensors
])

# Define a function to process a single image
def process_image(filename):
    # Construct the paths
    original_path = os.path.join(original_dir, filename)
    output_path = os.path.join(resized_dir, filename)

    # Read the original image using Pillow
    img = Image.open(original_path)

    # Apply the transformation
    img = transform(img)

    # Save the resized image
    save_image(img, output_path)

    return filename

# List all files in the original directory
file_list = os.listdir(original_dir)

# Detect the number of CPU cores available
num_cores = os.cpu_count()

# Create a Pool of worker processes using all available CPU cores
with Pool(processes=num_cores) as pool:
    result = pool.map(process_image, file_list)

print("Resizing and saving complete.")
