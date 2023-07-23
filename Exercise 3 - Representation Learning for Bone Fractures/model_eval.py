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
import json

#pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import resnet

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

def load_args_from_json(file_path):
    with open(file_path, 'r') as f:
        args = json.load(f)
    return args

results_dir = 'hw3_results_hes_bs8/'
save_path = 'metrics_model_hes_bs8/'

loaded_args = load_args_from_json(results_dir + 'args.json')

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
            im_1 = self.transform(image)
            im_2 = self.transform(image)

        if self.train:
            return im_1, im_2
        else:
            return im_1, label

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

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)

class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x

class ModelMoCo(nn.Module):
    def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)
        self.encoder_k = ModelBase(feature_dim=dim, arch=arch, bn_splits=bn_splits)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss

# create model
model = ModelMoCo(
    dim=loaded_args["moco_dim"],
    K=loaded_args["moco_k"],
    m=loaded_args["moco_m"],
    T=loaded_args["moco_t"],
    arch=loaded_args["arch"],
    bn_splits=loaded_args["bn_splits"],
    symmetric=loaded_args["symmetric"],
).cuda()

# Load the saved model
checkpoint = torch.load(results_dir + '/model_last.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
#
# # iterate over test set to get predictions
# true_labels = []
# predicted_labels = []
#
# for images, labels in test_loader:
#     images = images.cuda()
#     labels = labels.cuda()
#
#     # Forward pass
#     outputs = model(images)
#
#     # Compute predicted labels
#     _, predicted = torch.max(outputs.data, 1)
#
#     true_labels.extend(labels.cpu().numpy())
#     predicted_labels.extend(predicted.cpu().numpy())
#
# # Calculate metrics
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels)
# recall = recall_score(true_labels, predicted_labels)
# f1 = f1_score(true_labels, predicted_labels)
# roc_auc = roc_auc_score(true_labels, predicted_labels)
#
# print("Accuracy: {:.4f}".format(accuracy))
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1-Score: {:.4f}".format(f1))
# print("ROC AUC: {:.4f}".format(roc_auc))
#
# #visualize sample scans with predictions
# # Choose a few random samples
# sample_indices = np.random.choice(len(test_dataset), size=4, replace=False)
#
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#
# for i, idx in enumerate(sample_indices):
#     image, label = test_dataset[idx]
#     image = image.permute(1, 2, 0).numpy()
#
#     axes[i // 2, i % 2].imshow(image)
#     axes[i // 2, i % 2].set_title("True: {}, Predicted: {}".format(label, predicted_labels[idx]))
#
# plt.tight_layout()
# plt.savefig(save_path + 'sample.png')

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

