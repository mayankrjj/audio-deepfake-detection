import os
import torch
import librosa
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd


# Dataset Class
class AudioDataset(Dataset):
    def __init__(self, tsv_file, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        self.file_paths = []
        self.labels = []

        # Load the metadata file (TSV) using pandas
        data = pd.read_csv(tsv_file, sep='\t')

        # Assuming the .tsv file has columns 'file_path' and 'label'
        for _, row in data.iterrows():
            self.file_paths.append(os.path.join(data_dir, row['file_path']))
            self.labels.append(row['label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load and process audio file
        audio = self.load_audio(self.file_paths[idx])
        mel_spec = self.extract_mel_spectrogram(audio)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return mel_spec, label

    def load_audio(self, file_path, sr=16000):
        audio, _ = librosa.load(file_path, sr=sr)
        if self.augment:
            audio = self.augment_audio(audio)
        return audio

    def augment_audio(self, audio):
        # Apply time stretching, pitch shifting, and noise injection for augmentation
        if np.random.rand() < 0.5:
            audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
        if np.random.rand() < 0.5:
            audio = librosa.effects.pitch_shift(audio, 16000, n_steps=np.random.randint(-2, 2))
        if np.random.rand() < 0.5:
            noise = np.random.randn(len(audio))
            audio += 0.005 * noise
        return audio

    def extract_mel_spectrogram(self, audio, sr=16000, n_mels=128):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db


# CNN Model (unchanged)
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Training and Evaluation Functions (unchanged)
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds * 100
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


def evaluate_model(model, test_loader):
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds * 100
    print(f'Test Accuracy: {accuracy:.2f}%')


# Define dataset paths
train_tsv = '/Users/mohitrajput/Desktop/ASVspoof5_protocols/ASVspoof5.train.tsv'
test_tsv = '/Users/mohitrajput/Desktop/ASVspoof5_protocols/ASVspoof5.eval.track_2.trial.tsv'
data_dir = '/Users/mohitrajput/Desktop/ASVspoof5_protocols'  # Path to the folder containing audio files

# Create datasets and data loaders
train_dataset = AudioDataset(train_tsv, data_dir, augment=True)
test_dataset = AudioDataset(test_tsv, data_dir, augment=False)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# Evaluate the model
evaluate_model(model, test_loader)
