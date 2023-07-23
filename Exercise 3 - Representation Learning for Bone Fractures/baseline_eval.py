# -*- coding: utf-8 -*-

# general
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50


# sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


SEED = 222
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

results_dir = 'hw3_results_baseline_full_bs128/'
save_path = 'metrics_baseline_full/'

# def load_data(data_dir):
#
#   relevant_categories = ["XR_ELBOW", "XR_HAND", "XR_SHOULDER"]
#
#   dataset = []
#   for category in os.listdir(data_dir):
#
#     if category in relevant_categories: # Only for relevant categories
#       category_path = os.path.join(data_dir, category)
#
#       normal_patients, abnormal_patients = [], []
#
#       for patient in os.listdir(category_path): # Iterate through patients
#         patient_path = os.path.join(category_path, patient)
#
#         for study_type in os.listdir(patient_path): # Iterate through pos, nega
#
#           if "positive" in study_type:
#             abnormal_patients.append(patient_path)
#           elif "negative" in study_type:
#             normal_patients.append(patient_path)
#
#       patients = normal_patients + abnormal_patients
#       random.shuffle(patients)  # Shuffle the patients
#
#       for patient_path in patients:
#         for study_type in os.listdir(patient_path):
#           study_path = os.path.join(patient_path, study_type)
#
#           if "positive" in study_type:
#             label = "abnormal"
#           elif "negative" in study_type:
#             label = "normal"
#
#           for image in os.listdir(study_path):
#             image_path = os.path.join(study_path, image)
#             dataset.append(
#                 {
#                     'label': label,
#                     'body_part': category,
#                     'image_path': image_path
#                 }
#             )
#
#   return pd.DataFrame(dataset)

# create train and test dfs
def load_data(data_dir):

    dataset = []
    for category in os.listdir(data_dir):

        category_path = os.path.join(data_dir, category)

        normal_patients, abnormal_patients = [], []

        for patient in os.listdir(category_path): # Iterate through patients
            patient_path = os.path.join(category_path, patient)

            for study_type in os.listdir(patient_path): # Iterate through pos, nega

                if "positive" in study_type:
                    abnormal_patients.append(patient_path)
                elif "negative" in study_type:
                    normal_patients.append(patient_path)

        patients = normal_patients + abnormal_patients
        random.shuffle(patients)  # Shuffle the patients

        for patient_path in patients:
            for study_type in os.listdir(patient_path):

                study_path = os.path.join(patient_path, study_type)

                if "positive" in study_type:
                    label = "abnormal"
                elif "negative" in study_type:
                    label = "normal"

                for image in os.listdir(study_path):
                    image_path = os.path.join(study_path, image)
                    if image.startswith("._"):
                        continue  # Skip files starting with "._"
                    dataset.append(
                        {
                            'label': label,
                            'body_part': category,
                            'image_path': image_path
                        }
                    )

    return pd.DataFrame(dataset)


train_df = load_data('MURA-v1.1/train')
test_df = load_data('MURA-v1.1/valid')

# convert labels to numeric
label_mapping = {'normal': 0, 'abnormal': 1}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

class XRaysataset(Dataset):
    def __init__(self, df, transform=None, train=True):
        self.df = df
        self.train = train
        self.transform = transform
        self.classes = df['label'].unique()
        self.targets = df['label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['image_path']
        # patient_id = self.df['image_path'].str.extract(r'patient(\d+)')
        label = self.df.iloc[idx]['label']
        # body_part = self.df.iloc[idx]['body_part']

        image = Image.open(image_path)
        # image = cv2.imread(image_path)

        if self.transform:
            img = self.transform(image)

        return img, torch.tensor(label)

test_transform = transforms.Compose([
    # convert to 3 channels (duplicate the single channel XRay)
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_dataset = XRaysataset(test_df, transform=test_transform, train=False)

# create data loader
batch_size = 128
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

labels = test_df['label'].unique()

# Load the pre-trained ResNet50 model - without last fc layer
base_model = resnet50(pretrained=True)
num_features = base_model.fc.in_features
base_model.fc = nn.Identity()

# Add custom fully connected layers
model = nn.Sequential(
    base_model,
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Linear(128, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.Sigmoid()
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model.load_state_dict(torch.load(results_dir + "baselinemodel_epoch10.pt", map_location=device))

model.eval()
y_val = test_df['label'].values
y_predictions = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = (outputs >= 0.5).view(-1).cpu().numpy()
        y_predictions.extend(predictions.tolist())


def calculate_metrics(y_true, y_pred, save_path):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1 score:  {f1}")

    metrics = pd.DataFrame.from_dict(metrics, orient='index')
    metrics.to_csv(save_path + 'metrics.csv', header=False)

def plot_confusion_matrix(y_test, y_pred, class_labels, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,  # just use class_labels directly
                yticklabels=class_labels)  # just use class_labels directly
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path + 'confusion_matrix.png')

# predictions = np.array(predictions)
# true_labels = np.array(y_val)
# probabilities = np.array(probabilities)

# Calculate and save metrics
calculate_metrics(y_val, y_predictions, save_path)

# Plot confusion matrix
plot_confusion_matrix(y_val, y_predictions, labels, save_path)


# def plot_baseline(df, title, save_path):
#     epochs = df['epoch'].values
#     train_loss = df['train_loss'].values
#     val_loss = df['val_loss'].values
#     train_acc = df['train_acc@1'].values
#     val_acc = df['val_acc@1'].values
#
#     plt.figure(figsize=(14, 6))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss, label='Train Loss')
#     plt.plot(epochs, val_loss, label='Validation Loss')
#     plt.title('Loss vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, train_acc, label='Train Accuracy')
#     plt.plot(epochs, val_acc, label='Validation Accuracy')
#     plt.title('Accuracy vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(save_path + 'baseline.png')
#
#
# def plot_model_performance(df, title, save_path):
#     epochs = df['epoch'].values
#     train_loss = df['train_loss'].values
#     test_acc = df['test_acc@1'].values
#
#     plt.figure(figsize=(14, 6))
#
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_loss, label='Train Loss')
#     plt.title('Loss vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, test_acc, label='Test Accuracy')
#     plt.title('Accuracy vs. Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(save_path + 'model.png')
#
# baseline_full_bs128_logs_path = 'log.csv'
# baseline_full_bs128_logs = pd.read_csv(baseline_full_bs128_logs_path)
# baseline_full_bs128_logs
#
# plot_baseline(baseline_full_bs128_logs, 'Baseline (All data, batch size 128) Performance')
#
# """# Model (All data, batch size 8)"""
#
# model_full_bs8_logs_path = '/content/drive/MyDrive/AI_for_Healthcare/Exercise3/VM_RESULTS/hw3_results_full_bs8/log.csv'
# model_full_bs8_logs = pd.read_csv(model_full_bs8_logs_path)
# model_full_bs8_logs

# plot_model_performance(model_full_bs8_logs, 'Model (All data, batch size 8) Performance')
#
# """# Model (All data, batch size 256)"""
#
# model_full_bs256_logs_path = '/content/drive/MyDrive/AI_for_Healthcare/Exercise3/VM_RESULTS/hw3_results_full_bs256/log.csv'
# model_full_bs256_logs = pd.read_csv(model_full_bs256_logs_path)
# model_full_bs256_logs
#
# plot_model_performance(model_full_bs256_logs, 'Model (All data, batch size 256) Performance')
#
# """# Model (All data, batch size 256, batch norm 8)"""
#
# model_full_bs256_bn8_logs_path = '/content/drive/MyDrive/AI_for_Healthcare/Exercise3/VM_RESULTS/hw3_results_full_bs256_bn8/log.csv'
# model_full_bs256_bn8_logs = pd.read_csv(model_full_bs256_bn8_logs_path)
# model_full_bs256_bn8_logs
#
# plot_model_performance(model_full_bs256_bn8_logs, 'Model (All data, batch size 256, batch norm 8) Performance')
#
# def compare_models_performance(model_data, model_names):
#     plt.figure(figsize=(14, 6))
#
#     # Plotting train loss for all models
#     plt.subplot(1, 2, 1)
#     for i, df in enumerate(model_data):
#         plt.plot(df['epoch'], df['train_loss'], label=model_names[i])
#     plt.xlabel('Epochs')
#     plt.ylabel('Train Loss')
#     plt.title('Train Loss vs. Epochs')
#     plt.legend()
#
#     # Plotting test accuracy for all models
#     plt.subplot(1, 2, 2)
#     for i, df in enumerate(model_data):
#         plt.plot(df['epoch'], df['test_acc@1'], label=model_names[i])
#     plt.xlabel('Epochs')
#     plt.ylabel('Test Accuracy')
#     plt.title('Test Accuracy vs. Epochs')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# compare_models_performance([model_full_bs8_logs[0:10], model_full_bs256_logs, model_full_bs256_bn8_logs], ['Model (All data, batch size 8) Performance', 'Model (All data, batch size 256) Performance', 'Model (All data, batch size 256, batch norm 8) Performance'])