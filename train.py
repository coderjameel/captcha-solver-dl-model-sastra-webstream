import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 200, 50
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005 # Middle ground for stability
DATA_CSV = 'data_clean.csv'

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHARS)}
REV_MAP = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  

# ... (Keep setup_dataset, CaptchaDataset, and collate_fn from previous version) ...

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Simplified CNN to preserve more spatial info for the LSTM
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)) # Only pool height
        )
        # New Input size: 128 channels * 12 height = 1536
        self.rnn = nn.LSTM(1536, 256, bidirectional=True, num_layers=2, batch_first=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).view(w, b, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

def train():
    train_df, test_df = setup_dataset()
    # Stronger Augmentation
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5), # Handle webstream lighting
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(CaptchaDataset(train_df, 'dataset/train', transform), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # scheduler to cut LR if stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        for imgs, targets, target_lens in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
            
            loss = criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            
            # --- GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            t_loss += loss.item()

        avg_loss = t_loss/len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Periodic Save
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()