import os
import torch
from model import *
from piq import brisque
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm


generator = PretrainGenerator()
model = FeatureOpimizer()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Testing loop with tqdm
gan_images_dir = 'results/gan'
os.makedirs(gan_images_dir, exist_ok=True)

best_checkpoint_path = 'saved_models/phase1/best_gan_checkpoint.pth'
checkpoint = torch.load(best_checkpoint_path)
generator.load_state_dict(checkpoint['generator_state_dict'])

generated_images_num = 100
num_epochs = 300

generator.eval()

for i in tqdm(range(generated_images_num)):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        random_feature = torch.rand(10).unsqueeze(0)
        generated_feature = model(random_feature)
        generated_image_tensor = generator(generated_feature)
        loss = brisque(generated_image_tensor)
        if loss < best_loss:
            best_generated_image = generated_image_tensor
            best_loss = loss
        loss.backward()
        optimizer.step()
        print(f'Image {i+1} Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

    print(f'Optimizing for {i+1}-th random feature is finished.')

    generated_image = transforms.ToPILImage()(best_generated_image.squeeze().cpu().detach())
    generated_image.save(f'{gan_images_dir}/generated_{i+1:04d}.png')




