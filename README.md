# README

## Project Overview
The aim of this project is to develop a machine learning model that classifies images of buildings at Mississippi State University. Using a ResNet-50 architecture, the model predicts which building is shown in a given image. The project implements image data augmentation and mixed-precision training to optimize performance on limited computational resources.

### Folder Structure
- `dataset/MergeDataset_pt`: Contains the images of buildings saved as PyTorch tensors used for training the model.
- `dataset/test`: Contains the images of buildings for the test split, they can be png or jpeg.
- `output/`: Stores the trained model checkpoints and training metrics, such as loss plots.

### Files
- `train.py`: Contains functions to train the model using data augmentations and optimized settings for the available hardware.
- `eval.py`: Provides functions to evaluate the trained model, calculate performance metrics, and output results.
- `convert_to_pt.py`: Converts png images into pytorch tensors and saves them in an identically structured folder/subfolders.
- `run_prediction_on_single_image.py`: Simple script to run the model on a few images used for manual testing.
- `train_test_split.py`: Split the dataset into train and test.
- `heic_to_png.py`: Convert HEIC image format used by ios to png.
- `extract_frames.py`: Extract individual frames from a mov file.
---

## Instructions

### 1. Requirements
Ensure you have the following Python packages installed:
```bash
torch torchvision scikit-learn matplotlib
```

### 2. Training the Model

#### Model Architecture
- **Base Architecture**: ResNet-50, a deep CNN model pre-trained on ImageNet, is fine-tuned for this classification task.
- **Final Layer**: The fully connected (FC) layer is modified to predict 10 classes, each representing a unique building.

#### Training Steps
1. **Pre-trained Model Initialization**: The ResNet-50 model is initialized with pre-trained weights, and the final layer is adjusted for 10 classes.
2. **Data Augmentation**: The dataset undergoes transformations to improve model robustness:
    - Random resized cropping, horizontal flipping, and rotation.
    - Color jitter, Gaussian blur, and additive Gaussian noise for realistic variations.
3. **Mixed Precision Training**: `torch.cuda.amp.autocast` is used to accelerate training without compromising accuracy.
4. **Training**: 
   - Run `train.py` to start training.
   - The script saves the best-performing model based on loss reduction.
   - Loss plot updates each epoch and is saved in the `output/` directory.

#### Running Training
Run the `train.py` script as follows:
```bash
python train.py
```
Configure model and dataset paths within `train_model()` in `train.py` if needed.

---

### 3. Evaluating the Model

#### Evaluation Metrics
The evaluation metrics include:
- **Accuracy**: Measures overall classification correctness.
- **F1 Score**: Weighted average of precision and recall, accounting for class imbalance.
- **Confusion Matrix**: Displays true positives, false positives, and other metrics per class.
- **Classification Report**: Includes precision, recall, and F1 scores for each building class.

#### Running Evaluation
Run the `eval.py` script as follows:
```bash
python eval.py
```

Ensure the `dataset_path` and `model_path` variables point to the correct dataset and model checkpoint in `main()` in `eval.py`.

---

## Output
- **Training Loss Plot**: Saved as `loss.png` in the `output/` folder.
- **Model Checkpoints**: Saved after each epoch and for the best model in `output/`.
- **Evaluation Report**: Prints the accuracy, F1 score, confusion matrix, and classification report in the console.
