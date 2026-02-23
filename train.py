import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration & Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH = 200
IMG_HEIGHT = 50
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_CSV = 'data.csv'
IMG_DIR = 'captchas/'

# Alphanumeric character set for SASTRA webstream
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHARS)}
REV_MAP = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token

# --- Data Preparation ---
def setup_dataset():
    # Check if folders already exist and contain files
    if os.path.exists('dataset/train') and os.path.exists('dataset/test'):
        if len(os.listdir('dataset/train')) > 1: # Checking for more than just labels.csv
            print(">>> Dataset already split and organized. Skipping setup.")
            train_df = pd.read_csv('dataset/train/labels.csv')
            test_df = pd.read_csv('dataset/test/labels.csv')
            return train_df, test_df

    print(">>> Organizing dataset for the first time...")
    df = pd.read_csv(DATA_CSV)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    for folder in ['dataset/train', 'dataset/test']:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)

    # Save splits and copy files
    train_df.to_csv('dataset/train/labels.csv', index=False)
    test_df.to_csv('dataset/test/labels.csv', index=False)
    
    for _, row in df.iterrows():
        dest = 'dataset/train' if row['filename'] in train_df['filename'].values else 'dataset/test'
        shutil.copy(os.path.join(IMG_DIR, row['filename']), os.path.join(dest, row['filename']))
    
    return train_df, test_df

class CaptchaDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx, 0]))
        image = Image.open(img_name).convert('L') 
        label_str = str(self.df.iloc[idx, 1])
        label = torch.IntTensor([CHAR_MAP[c] for c in label_str])
        
        if self.transform: image = self.transform(image)
        return image, label, torch.IntTensor([len(label)])

def collate_fn(batch):
    images, targets, target_lens = zip(*batch)
    images = torch.stack(images, 0)
    # Pad sequences to handle varying lengths (e.g., 5 vs 6 chars)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    target_lens = torch.cat(target_lens, 0)
    return images, targets_padded, target_lens

# --- Model Architecture ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(256, 128, bidirectional=True, num_layers=2, batch_first=False)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.cnn(x) 
        x = x.permute(3, 0, 1, 2) 
        w, b, c, h = x.size()
        x = x.view(w, b, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)

def train():
    train_df, test_df = setup_dataset()
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(CaptchaDataset(train_df, 'dataset/train', transform), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CaptchaDataset(test_df, 'dataset/test', transform), 
                             batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')

    print(f"Starting training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        loop = tqdm(train_loader, leave=False, desc=f"Epoch [{epoch}/{EPOCHS}]")
        for imgs, targets, target_lens in loop:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full(size=(imgs.size(0),), fill_value=logits.size(0), dtype=torch.int32)
            
            loss = criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, targets, target_lens in test_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                input_lens = torch.full(size=(imgs.size(0),), fill_value=logits.size(0), dtype=torch.int32)
                v_loss += criterion(log_probs, targets, input_lens, target_lens).item()

        avg_train = t_loss / len(train_loader)
        avg_test = v_loss / len(test_loader)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)

        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")
        
        if avg_test < best_loss:
            best_loss = avg_test
            torch.save(model.state_dict(), "best_model.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Training Metrics - Captcha Solver')
    plt.legend()
    plt.savefig('training_metrics.png')
    print("Training Complete. Model saved as best_model.pth")

if __name__ == "__main__":
    train()