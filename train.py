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

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 200, 50
BATCH_SIZE = 64 # DGX can handle larger batches easily
EPOCHS = 100    # Increased epochs for better convergence
LEARNING_RATE = 0.0001 # Reduced to fix the plateau
DATA_CSV = 'data_clean.csv' 
IMG_DIR = 'captchas/'

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  

def setup_dataset():
    if os.path.exists('dataset/train') and os.path.exists('dataset/test'):
        if len(os.listdir('dataset/train')) > 1:
            print(">>> Using existing split dataset.")
            return pd.read_csv('dataset/train/labels.csv'), pd.read_csv('dataset/test/labels.csv')

    df = pd.read_csv(DATA_CSV)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    for folder in ['dataset/train', 'dataset/test']:
        if os.path.exists(folder): shutil.rmtree(folder)
        os.makedirs(folder)
    train_df.to_csv('dataset/train/labels.csv', index=False)
    test_df.to_csv('dataset/test/labels.csv', index=False)
    
    train_filenames = set(train_df['filename'].values)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting"):
        dest = 'dataset/train' if row['filename'] in train_filenames else 'dataset/test'
        shutil.copy(os.path.join(IMG_DIR, row['filename']), os.path.join(dest, row['filename']))
    return train_df, test_df

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

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn_input_size = 1536 
        self.rnn = nn.LSTM(self.rnn_input_size, 256, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights for faster convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.cnn(x) 
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).view(w, b, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)

def train():
    train_df, test_df = setup_dataset()
    # Adding Augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(CaptchaDataset(train_df, 'dataset/train', train_transform), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CaptchaDataset(test_df, 'dataset/test', val_transform), 
                             batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        loop = tqdm(train_loader, leave=False, desc=f"Epoch [{epoch}/{EPOCHS}]")
        for imgs, targets, target_lens in loop:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
            loss = criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, targets, target_lens in test_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                logits = model(imgs)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
                v_loss += criterion(log_probs, targets, input_lens, target_lens).item()

        avg_train, avg_test = t_loss/len(train_loader), v_loss/len(test_loader)
        history['train_loss'].append(avg_train)
        history['test_loss'].append(avg_test)
        scheduler.step(avg_test) # Adjust LR if stuck
        
        print(f"Epoch {epoch} | Train: {avg_train:.4f} | Test: {avg_test:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_test < best_loss:
            best_loss = avg_test
            torch.save(model.state_dict(), "best_model.pth")
    
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['test_loss'], label='Test')
    plt.savefig('training_metrics.png')

if __name__ == "__main__":
    train()