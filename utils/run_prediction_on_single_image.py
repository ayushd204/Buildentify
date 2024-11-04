import torch
from torchvision import models, transforms
from PIL import Image
import os

# a simple script to predict the class of a single image or a folder containing images

# Define class names
CLASS_NAMES = [
    "Butler Hall", "Carpenter Hall", "Lee Hall", "McCain Hall", "McCool Hall", 
    "Old Main", "Simrall Hall", "Student Union", "Swalm Hall", "Walker Hall"
]

# Define the data transformations for a single image
def get_transform():
    return transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4608, 0.4610, 0.4559], std=[0.2312, 0.2275, 0.2638]),
    ])

# Load and prepare the model
def load_model(model_path, num_classes=10):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device

# Load and preprocess a single image
def load_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Predict the class for a single image
def predict_image(model, image_path,transform, device):
    image = load_image(image_path, transform, device)
    with torch.no_grad():
        output = model(image)
        output = torch.nn.functional.softmax(output, dim=1)  # Get class probabilities
        _, pred = torch.max(output, 1)
        
        # if pred.item() != 3:
        #     print(f"Predicted class: {CLASS_NAMES[pred.item()]} for image {image_path}")

        return CLASS_NAMES[pred.item()]

# Process all images in a folder
def process_folder(model, image_folder,transform, device):
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        print(predict_image(model, image_path, transform, device))

def main():

    model_path = "C:/Users/works/OneDrive/Desktop/projects/vision/outptut/resnet50_building_classifier39.pth"
    image_path = "C:/Users/works/Downloads/test.jpg"    # path to your image
    image_folder = "E:/home/works/Desktop/projects/vision/dataset/MergeDataset/McCain Hall" # path to your image folder

    # Initialize transformations, model, and device
    transform = get_transform()
    model, device = load_model(model_path)

    # get prediction for a single image
    print(predict_image(model, image_path, transform, device))

    # get predictions for images in the folder
    process_folder(model, image_folder, transform, device)

if __name__ == "__main__":
    main()
