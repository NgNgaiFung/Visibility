# modeling
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from scipy.io import savemat

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

class SplitImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(image_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path)
        label = int(image_file.split("_")[1].split(".")[0])

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

[[[123],[123],[123]],[123]]

class FeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = os.listdir(root_dir)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.root_dir, self.samples[idx])
        feature = torch.load(feature_path)
        label = int(self.samples[idx].split('_')[-1].split('.')[0])
        return feature, label

def vgg16_feature_extraction(data_loader):
    vgg16 = models.vgg16(weights="VGG16_Weights.IMAGENET1K_FEATURES").eval().cuda()
    # vgg16.classifier = vgg16.classifier[:-1]
    num = 1
    if not os.path.exists('./features'):
        os.makedirs('./features')
    for batch_idx, sample in enumerate(data_loader):
        image, label = sample[0].cuda(), sample[1].cuda()
        print("before vgg:", image.shape)
        feature = vgg16(image)
        print("after vgg:", feature)
        for i, (feature, label) in enumerate(zip(feature, label)):
            print(f"Feature shape: {feature}, Label: {label}")
            result = {"feature": feature.cpu().detach().numpy(), "label": label.cpu().detach().numpy()}
            savemat(f'./features/feature_{num}.mat', result)
            # torch.save(feature, f'./features/feature_{num}_{label}.pt')
            num += 1
        print("=====================================")
        print("Memory allocated by CUDA:", torch.cuda.memory_allocated(device) / 1024 / 1024, "MiB")
        print("Max memory allocated by CUDA:", torch.cuda.max_memory_allocated(device) / 1024 / 1024, "MiB")
        print("Process Percentage:", (batch_idx+1)/len(data_loader)*100, "%")


image_dataset = "segmented_images/image1"
split_dataset = SplitImageDataset(image_dataset, transform=transform)

batch_size = 3
data_loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    vgg16_feature_extraction(data_loader)