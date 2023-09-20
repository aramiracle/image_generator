import os
import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

train_root_dir = 'data/resized'

transform = transforms.Compose([transforms.ToTensor()])

image_names = os.listdir(train_root_dir)

model_1 = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')
model_2 = models.shufflenet_v2_x2_0(weights='ShuffleNet_V2_X2_0_Weights.DEFAULT')
model_3 = models.regnet_y_1_6gf(weights='RegNet_Y_1_6GF_Weights.DEFAULT')
model_4 = models.densenet121(weights='DenseNet121_Weights.DEFAULT')
model_5 = models.mnasnet1_3(weights='MNASNet1_3_Weights.DEFAULT')
model_6 = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
model_7 = models.regnet_x_1_6gf(weights='RegNet_X_1_6GF_Weights.DEFAULT')

model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()
model_6.eval()
model_7.eval()

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

features_array = torch.cat(features_list, dim=0).detach().numpy()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_array)
print('Features transformed')

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Projection')
plt.grid(True)
plt.show()
