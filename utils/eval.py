import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import os

def get_class_names():
    return [
        "Butler Hall", "Carpenter Hall", "Lee Hall", "McCain Hall",
        "McCool Hall", "Old Main", "Simrall Hall", "Student Union",
        "Swalm Hall", "Walker Hall"
    ]

def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4608, 0.4610, 0.4559], std=[0.2312, 0.2275, 0.2638]),
    ])

def load_data(dataset_path, transform=get_data_transforms()):
    eval_data = ImageFolder(root=dataset_path, transform=transform)
    eval_loader = DataLoader(eval_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    return eval_data, eval_loader

def load_model(model_path, num_classes=10):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)

def evaluate_model(model, eval_loader, eval_data):
    all_preds, all_labels = [], []

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(eval_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def calculate_metrics(all_labels, all_preds, class_names):
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return accuracy, f1, conf_matrix, class_report

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of utils directory
    root_dir = os.path.dirname(script_directory)

    # path to the dataset folder, containing subfolders for each class
    dataset_path = os.path.join(root_dir,'/dataset/test/')

    # path to the trained model
    model_path = os.path.join(root_dir,'/output/resnet50_building_classifier.pth')

    class_names = get_class_names()
    transform = get_data_transforms()
    eval_data, eval_loader = load_data(dataset_path, transform)
    model = load_model(model_path)

    all_labels, all_preds = evaluate_model(model, eval_loader, eval_data)

    # Calculate and print metrics
    accuracy, f1, conf_matrix, class_report = calculate_metrics(all_labels, all_preds, class_names)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

if __name__ == "__main__":
    main()