import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 200, 50
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0003 # Fine-tuned for BiLSTM stability
DATA_CSV = 'data_clean.csv'

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHARS)}
REV_MAP = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  

# --- Custom Preprocessing for Touching Characters ---
class CaptchaTransform:
    def __call__(self, img):
        # Invert if the background is dark, ensure black text on white
        img = ImageOps.grayscale(img)
        # Add slight contrast boost to separate touching letters
        return transforms.functional.adjust_contrast(img, 2.0)

class CaptchaDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df, self.root_dir, self.transform = df, root_dir, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx, 0]))
        image = Image.open(img_name).convert('L')
        label_str = str(self.df.iloc[idx, 1])
        label_indices = [CHAR_MAP[c] for c in label_str if c in CHAR_MAP]
        if not label_indices: label_indices = [1]
        label = torch.IntTensor(label_indices)
        if self.transform: image = self.transform(image)
        return image, label, torch.IntTensor([len(label)])

def collate_fn(batch):
    images, targets, target_lens = zip(*batch)
    images = torch.stack(images, 0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    target_lens = torch.cat(target_lens, 0)
    return images, targets_padded, target_lens

# --- CRNN Architecture ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1), nn.LeakyReLU(0.2), nn.MaxPool2d((2, 1))
        )
        self.rnn_input_size = 1536 
        self.rnn = nn.LSTM(self.rnn_input_size, 256, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).view(w, b, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)

def train():
    # Load and Split
    df = pd.read_csv(DATA_CSV)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    transform = transforms.Compose([
        CaptchaTransform(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(CaptchaDataset(train_df, 'captchas/', transform), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    print(f"Starting training on {DEVICE} for SASTRA Captchas...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        for imgs, targets, target_lens in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
            
            loss = criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            
            # Clip gradients to stop the 3.5 plateau
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        avg_loss = t_loss/len(train_loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    train()