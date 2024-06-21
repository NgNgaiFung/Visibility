import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        num_files = len(os.listdir(folder_path))
        self.transform = transform

        self.data = []
        for i in range(num_files):
            file_name = folder_path + f'/feature_{i+1}.mat'
            loaded_data = loadmat(file_name)
            feature = loaded_data['feature'][0]
            label = loaded_data['label'][0][0]
            self.data.append({'feature': feature, 'label': label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        return self.data[index]

class ToTensor(object):
    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        return {'feature': torch.from_numpy(feature), 'label':label}

# SVR_model
##############################################################
class SVR(nn.Module):
    def __init__(self):
        super(SVR, self).__init__()
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

model = SVR()

# Loss function
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
##############################################################

# check the torch dataset is working
folder_path = 'features'
dataset = MyDataset(folder_path='features', transform=ToTensor())
print(type(dataset[0]['feature']), dataset[0]['feature'].shape, dataset[0]['label'], type(dataset[0]['label']))

# dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# split the dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(len(train_dataset), len(test_dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)
print(next(iter(train_dataloader))['feature'].type())
print()
# train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        feature = data['feature'].float()
        label = data['label'].float()
        optimizer.zero_grad()
        output = model(feature)
        loss_output = loss(output, label)
        loss_output.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss_output.item()}")
    print("=====================================")
