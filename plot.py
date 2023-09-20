import os
import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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
pca = PCA(n_components=5)
X_pca = pca.fit_transform(features_array)

# Min-max scaling to map values between 0 and 1
min_max_scaler = MinMaxScaler()
X_pca_scaled = min_max_scaler.fit_transform(X_pca)

print(f'Explained variance: {pca.explained_variance_ratio_.cumsum()}')
print('Features transformed')

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1])
plt.xlabel('Principal Component 1 (Scaled)')
plt.ylabel('Principal Component 2 (Scaled)')
plt.title('PCA 2D Projection (Scaled)')
plt.grid(True)
plt.show()

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], X_pca_scaled[:, 2])
ax.set_xlabel('Principal Component 1 (Scaled)')
ax.set_ylabel('Principal Component 2 (Scaled)')
ax.set_zlabel('Principal Component 3 (Scaled)')
ax.set_title('PCA 3D Projection (Scaled)')

plt.show()
