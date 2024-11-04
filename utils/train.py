import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import os
import matplotlib.pyplot as plt
import time

# optimized for machine with the following specs
# CPU: 13th Gen Intel(R) Core(TM) i5-13500HX   2.50 GHz
# RAM: 16.0 GB
# GPU: NVIDIA GeForce RTX 4050 laptop GPU
# VRAM: 6141 MB 

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class PTImageFolder(Dataset):
    """ all the images are converted to a tensor and saved to reduce the time taken to load the images 
        this class is used to load the images from the directory and apply the transformations to the images"""
    def __init__(self, root):
        self.root = root
        self.data = []
        self.labels = []
        
        # Traverse the directory
        for label, subfolder in enumerate(os.listdir(root)):
            subfolder_path = os.path.join(root, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith('.pt'):
                        self.data.append(os.path.join(subfolder_path, file))
                        self.labels.append(label)  # Use subfolder index as label

        # we use a bunch of transformations to augment the data
        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.5)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
        transforms.Normalize(mean=[0.4608, 0.4610, 0.4559], std=[0.2312, 0.2275, 0.2638]),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        AddGaussianNoise(0., 0.06)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the tensor and corresponding label
        # print(self.data[idx])
        tensor = torch.load(self.data[idx])
        label = self.labels[idx]

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label    

def load_model(model_path, num_classes=10):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def plot_loss(train_losses, output_path="loss.png"):
    plt.plot(train_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.savefig(output_path)
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        # using mixed precision training to speed up training
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(train_loader)

def train_model(
        model_path="resnet50_building_classifier.pth",
        dataset_path='E:/home/works/Desktop/projects/vision/dataset/mergedataset_pt/',
        out_folder='C:/Users/works/OneDrive/Desktop/projects/vision/output/',
        batch_size=64,
        learning_rate=0.001,
        num_epochs=100
):
    print("Initializing training...")
    model = load_model(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # get the dataloader
    dataset = PTImageFolder(dataset_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

    # defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    train_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device="cuda" if torch.cuda.is_available() else "cpu")
        train_losses.append(avg_loss)

        # Save the best model
        if avg_loss == min(train_losses):
            torch.save(model.state_dict(), os.path.join(out_folder, 'resnet50_building_classifier_best.pth'))

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(out_folder, f'resnet50_building_classifier_epoch{epoch + 1}.pth'))

        plot_loss(train_losses, os.path.join(out_folder, "loss.png"))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of utils directory
    base_directory = os.path.dirname(script_directory)

    # path to the dataset folder, containing subfolders for each class
    # the data should be in the form of a pytorch tensor
    dataset_path = os.path.join(base_directory,"dataset\\MergeDataset_pt\\")
    model_path = os.path.join(base_directory,"output\\model.pth")
    out_folder = os.path.join(base_directory,"output\\")

    train_model(model_path, dataset_path, out_folder)
