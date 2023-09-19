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

best_checkpoint_path = 'saved_models/gan/best_gan_checkpoint.pth'
checkpoint = torch.load(best_checkpoint_path)
generator.load_state_dict(checkpoint['generator_state_dict'])

generated_images_num = 100
num_epochs = 1000

generator.eval()

for i in tqdm(range(generated_images_num)):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        random_feature = torch.rand(1, 3000)
        generated_feature = model(random_feature)
        generated_image_tensor = generator(generated_feature)
        loss = brisque(generated_image_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')

    print(f'Optimizing for {i+1}-th random feature is finished.')

    model.eval()
    with torch.no_grad():
        generated_image_tensor = generator(model(random_feature))

        generated_image = transforms.ToPILImage()(generated_image_tensor.squeeze().cpu().detach())
        generated_image.save(f'{gan_images_dir}/generated_{i+1:04d}.png')




