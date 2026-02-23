import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration for DGX Node ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WIDTH, IMG_HEIGHT = 200, 50
BATCH_SIZE = 32         # Reduced to force more frequent weight updates
EPOCHS = 200
LEARNING_RATE = 0.0005  # Steady initial learning rate
DATA_CSV = 'data_clean.csv'

CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHARS)}
REV_MAP = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  

# --- Advanced Preprocessing ---
class CaptchaNoiseFilter:
    """Applies a median filter to blur out thin background lines."""
    def __call__(self, img):
        img = img.convert('L')
        return img.filter(ImageFilter.MedianFilter(size=3))

class CaptchaDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df, self.root_dir, self.transform = df, root_dir, transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.df.iloc[idx, 0]))
        image = Image.open(img_name)
        
        label_str = str(self.df.iloc[idx, 1])
        label_indices = [CHAR_MAP[c] for c in label_str if c in CHAR_MAP]
        if not label_indices: label_indices = [1]
        
        label = torch.IntTensor(label_indices)
        if self.transform: image = self.transform(image)
        return image, label, torch.IntTensor([len(label)]), label_str

def collate_fn(batch):
    images, targets, target_lens, raw_labels = zip(*batch)
    images = torch.stack(images, 0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    target_lens = torch.cat(target_lens, 0)
    return images, targets_padded, target_lens, raw_labels

# --- CTC Decoder & Metrics ---
def greedy_decoder(logits):
    pred_indices = logits.argmax(2) 
    pred_indices = pred_indices.transpose(0, 1).cpu().numpy() 
    
    decoded_strings = []
    for seq in pred_indices:
        char_list = []
        for i in range(len(seq)):
            if seq[i] != 0 and (i == 0 or seq[i] != seq[i-1]):
                char_list.append(REV_MAP[seq[i]])
        decoded_strings.append("".join(char_list))
    return decoded_strings

def calculate_metrics(predictions, targets):
    correct_words = 0
    correct_chars = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct_words += 1
            
        min_len = min(len(pred), len(target))
        correct_chars += sum(1 for p, t in zip(pred[:min_len], target[:min_len]) if p == t)
        total_chars += max(len(pred), len(target))
        
    word_acc = (correct_words / len(targets)) * 100
    char_acc = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    return word_acc, char_acc

# --- Deep CRNN Architecture ---
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout(0.2) 
        )
        self.rnn_input_size = 512 * 6 
        self.rnn = nn.LSTM(self.rnn_input_size, 256, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).view(w, b, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)

def train():
    df = pd.read_csv(DATA_CSV)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    train_transform = transforms.Compose([
        CaptchaNoiseFilter(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(3, fill=255),
        transforms.ColorJitter(contrast=2.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        CaptchaNoiseFilter(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = DataLoader(CaptchaDataset(train_df, 'captchas/', train_transform), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(CaptchaDataset(test_df, 'captchas/', test_transform), 
                             batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CRNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Patient scheduler: Drops LR only if validation loss stops improving for 15 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.5)

    best_word_acc = 0.0

    print("="*80)
    print(f"🚀 INITIATING STABILIZED TRAINING ON {DEVICE.type.upper()}")
    print(f"Total Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Dataset: {len(df)} images")
    print("="*80)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch [{epoch}/{EPOCHS}]")
        for imgs, targets, target_lens, raw_labels in loop:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
            
            loss = criterion(log_probs, targets, input_lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(greedy_decoder(logits))
            train_targets.extend(raw_labels)

        avg_train_loss = train_loss / len(train_loader)
        train_w_acc, train_c_acc = calculate_metrics(train_preds, train_targets)

        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for imgs, targets, target_lens, raw_labels in test_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                logits = model(imgs)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.int32)
                
                v_loss = criterion(log_probs, targets, input_lens, target_lens)
                val_loss += v_loss.item()
                
                val_preds.extend(greedy_decoder(logits))
                val_targets.extend(raw_labels)

        avg_val_loss = val_loss / len(test_loader)
        val_w_acc, val_c_acc = calculate_metrics(val_preds, val_targets)
        
        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        print(f"\n[EPOCH {epoch:03d}/{EPOCHS}] LR: {curr_lr:.6f}")
        print(f" ┣▶ TRAIN | Loss: {avg_train_loss:.4f} | Char Acc: {train_c_acc:5.2f}% | Word Acc: {train_w_acc:5.2f}%")
        print(f" ┗▶ TEST  | Loss: {avg_val_loss:.4f} | Char Acc: {val_c_acc:5.2f}% | Word Acc: {val_w_acc:5.2f}%")
        
        if epoch % 5 == 0 or val_w_acc > 5.0:
            print(f"    * Sample - True: '{val_targets[0]}' | Pred: '{val_preds[0]}'")

        if val_w_acc >= best_word_acc and val_w_acc > 0:
            best_word_acc = val_w_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f" 🌟 NEW BEST MODEL SAVED! Word Accuracy: {best_word_acc:.2f}%")

    print("\n" + "="*80)
    print(f"🎉 TRAINING COMPLETE. Maximum Test Word Accuracy achieved: {best_word_acc:.2f}%")
    print("="*80)

if __name__ == "__main__":
    train()