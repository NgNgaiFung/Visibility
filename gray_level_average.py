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

        width, height = image.size
        image = image.crop((0, 30, width, height))

        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transformation pipeline
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
])

excel_file = "data.xlsx"
image_folder = "images/"
dataset = CustomDataset(excel_file, image_folder, transform=transform)

# Create a DataLoader
batch_size = 32  # Set your desired batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def gray_level_averaging(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_images = len(dataloader.dataset)
    processed_images = 0
    result = None
    for images, _ in dataloader:
        images = images.to(device)
        if result is None:
            result = torch.zeros(images.shape[1:], device=device)
        result += torch.sum(images, dim=0)
        processed_images += batch_size
        print(f"Batch processed ({processed_images}/{total_images} images)", f"{device} Memory allocated:", torch.cuda.memory_allocated() / 1024 / 1024, "MiB")
    return result / len(dataset)

result = gray_level_averaging(dataloader) 
torch.save(result, "averaged_gray_level.pt")

result_image = transforms.ToPILImage()(result.squeeze(0).cpu())
result_image.save("averaged_gray_level.jpg")

plt.imshow(result.squeeze(0).cpu(), cmap="gray")
plt.show()