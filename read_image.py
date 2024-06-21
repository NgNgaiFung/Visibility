import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, excel_file, image_folder, transform=None):
        self.data = pd.read_excel(excel_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx, 1]

        width, height = image.size
        piece_width = width // 5
        piece_height = height // 4
        upper = 2 * piece_height
        lower = 3 * piece_height

        image1 = image.crop((0, upper, piece_width, lower))
        image2 = image.crop((piece_width, upper, 2 * piece_width, lower))
        image3 = image.crop((2 * piece_width, upper, 3 * piece_width, lower))
        image4 = image.crop((3 * piece_width, upper, 4 * piece_width, lower))
        image5 = image.crop((4 * piece_width, upper, 5 * piece_width, lower))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
            image4 = self.transform(image4)
            image5 = self.transform(image5)

        return image1, image2, image3, image4, image5, label

# Define the transformation pipeline
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

excel_file = "data.xlsx"
image_folder = "images/"
dataset = CustomDataset(excel_file, image_folder, transform=transform)

print(dataset[0][5])

# Create a data loader
# data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

import torchvision.transforms.functional as F

if not os.path.exists("segmented_images"):
    os.mkdir("segmented_images")
    os.mkdir("segmented_images/image1")
    os.mkdir("segmented_images/image2")
    os.mkdir("segmented_images/image3")
    os.mkdir("segmented_images/image4")
    os.mkdir("segmented_images/image5")

for i in range(len(dataset)):
    image1, image2, image3, image4, image5, labels = dataset[i]
    image1_path = f"segmented_images/image1/pic{i}_{labels}.jpg"
    image1_i = transforms.ToPILImage()(image1)
    image1_i.save(image1_path)

    image2_path = f"segmented_images/image2/pic{i}_{labels}.jpg"
    image2_i = transforms.ToPILImage()(image2)
    image2_i.save(image2_path)

    image3_path = f"segmented_images/image3/pic{i}_{labels}.jpg"
    image3_i = transforms.ToPILImage()(image3)
    image3_i.save(image3_path)

    image4_path = f"segmented_images/image4/pic{i}_{labels}.jpg"
    image4_i = transforms.ToPILImage()(image4)
    image4_i.save(image4_path)

    image5_path = f"segmented_images/image5/pic{i}_{labels}.jpg"
    image5_i = transforms.ToPILImage()(image5)
    image5_i.save(image5_path)

# num_rows = 2
# num_cols = 5

# plt.figure(figsize=(10, 2))

# for i in range(num_rows):
#     plt.subplot(num_rows, num_cols, i * num_cols + 1)
#     plt.imshow(image1[i].permute(1, 2, 0))
#     plt.title(labels[i].item())
#     plt.axis("off")

#     plt.subplot(num_rows, num_cols, i * num_cols + 2)
#     plt.imshow(image2[i].permute(1, 2, 0))
#     plt.title(labels[i].item())
#     plt.axis("off")

#     plt.subplot(num_rows, num_cols, i * num_cols + 3)
#     plt.imshow(image3[i].permute(1, 2, 0))
#     plt.title(labels[i].item())
#     plt.axis("off")

#     plt.subplot(num_rows, num_cols, i * num_cols + 4)
#     plt.imshow(image4[i].permute(1, 2, 0))
#     plt.title(labels[i].item())
#     plt.axis("off")

#     plt.subplot(num_rows, num_cols, i * num_cols + 5)
#     plt.imshow(image5[i].permute(1, 2, 0))
#     plt.title(labels[i].item())
#     plt.axis("off")
    
# plt.show()