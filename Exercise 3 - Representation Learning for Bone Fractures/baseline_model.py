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

# # convert labels to numeric
# label_mapping = {'normal': 0, 'abnormal': 1}
# train_df['label'] = train_df['label'].map(label_mapping)
# test_df['label'] = test_df['label'].map(label_mapping)

# convert labels to numeric
label_mapping = {'normal': 0, 'abnormal': 1}
train_df['label'] = train_df['label'].map(label_mapping)
test_df['label'] = test_df['label'].map(label_mapping)

# Extract the patient ID from the image path
train_df['patient_id'] = train_df['image_path'].str.extract(r'patient(\d+)')

# Count the unique patient IDs per body part and view count
view_count_per_body_part = train_df.groupby(['body_part', 'patient_id']).size().reset_index(name='view_count')
# Count the number of patients per unique view count per body part
patient_count_per_view_count = view_count_per_body_part.groupby(['body_part', 'view_count']).size().reset_index(name='patient_count')
patient_count_per_view_count.to_csv(results_dir + 'patient_counts.csv')

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

train_transform = transforms.Compose([
    # convert to 3 channels (duplicate the single channel XRay)
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    # convert to 3 channels (duplicate the single channel XRay)
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# create datasets
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED) # split train to train and val

train_dataset = XRaysataset(train_df, transform=train_transform, train=True)
val_dataset = XRaysataset(val_df, transform=test_transform, train=True)
test_dataset = XRaysataset(test_df, transform=test_transform, train=False)

# create data loaders
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

labels = train_df['label'].unique()

# Extract the patient ID from the image path
train_df['patient_id'] = train_df['image_path'].str.extract(r'patient(\d+)')

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

model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# load model if resume
epoch_start = 1
# if args.resume != '':
    # checkpoint = torch.load(args.resume)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch_start = checkpoint['epoch'] + 1
    # print('Loaded from: {}'.format(args.resume))
    #
# Training loop
results = {'train_loss': [], 'train_acc@1': [],'val_loss': [], 'val_acc@1': []}
num_epochs = 10

for epoch in range(epoch_start, num_epochs + 1):
    train_bar = tqdm(train_loader)
    val_bar = tqdm(val_loader)
    # Training
    model.train()

    total_num = 0
    train_loss_total = 0.0
    correct_train_total = 0
    total_train_total = 0
    train_accuracy = 0

    for images, labels in train_bar:
        correct_train = 0
        total_train = 0

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        train_loss = criterion(outputs, labels.float().unsqueeze(1))
        train_loss.backward()
        optimizer.step()

        train_loss_total += train_loss.item() * train_loader.batch_size
        total_num += train_loader.batch_size

        # Calculate train accuracy
        predicted_labels = (outputs >= 0.5).view(-1).cpu().numpy()
        correct_train += (predicted_labels == labels.cpu().numpy()).sum()
        correct_train_total += correct_train
        total_train += labels.size(0)
        total_train_total += total_train

        train_accuracy = correct_train_total / total_train_total
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(epoch, num_epochs,
                                                                                                  optimizer.param_groups[0]['lr'],
                                                                                                  train_loss_total / total_num,
                                                                                                  train_accuracy * 100))
    results['train_loss'].append(train_loss_total / total_num)
    results['train_acc@1'].append(train_accuracy * 100)

    # Evaluation
    model.eval()

    val_loss_total = 0.0
    correct_val_total = 0
    total_val_total = 0
    val_accuracy = 0

    with torch.no_grad():
        for images, labels in val_loader:
            correct_val = 0
            total_val = 0

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            val_loss = criterion(outputs, labels.float().unsqueeze(1))
            val_loss_total += val_loss.item() * val_loader.batch_size

            # Calculate validation accuracy
            predicted_labels = (outputs >= 0.5).view(-1).cpu().numpy()
            correct_val += (predicted_labels == labels.cpu().numpy()).sum()
            correct_val_total += correct_val
            total_val += labels.size(0)
            total_val_total += total_val

            val_accuracy = correct_val_total / total_val_total
            val_bar.set_description('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format((val_loss_total / total_num), val_accuracy*100))

    results['val_loss'].append(val_loss_total / total_num)
    results['val_acc@1'].append(val_accuracy * 100)
    data_frame = pd.DataFrame(data=results, index=range(epoch_start, epoch + 1))
    data_frame.to_csv(results_dir + 'log.csv', index_label='epoch')
    # Save model checkpoint
    checkpoint_path = os.path.join(results_dir, f'baselinemodel_epoch{epoch}.pt')
    torch.save(model.state_dict(), checkpoint_path)

# Testing
model.eval()
y_val = test_df['label'].map({'normal': 0, 'abnormal': 1}).values
y_pred = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = (outputs >= 0.5).view(-1).cpu().numpy()
        y_pred.extend(predictions.tolist())

# Save the model weights
model_weights_path = os.path.join(results_dir, 'ResNet50_BodyParts.pt')
torch.save(model.state_dict(), model_weights_path)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
