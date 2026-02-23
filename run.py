import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# Must match train.py exactly
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
REV_MAP = {i + 1: char for i, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

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

def decode_prediction(output):
    # Greedy decoding for CTC
    output = output.argmax(2) # [W, B]
    output = output.squeeze(1).cpu().numpy()
    
    char_list = []
    for i in range(len(output)):
        if output[i] != 0 and (not (i > 0 and output[i] == output[i-1])):
            char_list.append(REV_MAP[output[i]])
    return "".join(char_list)

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(NUM_CLASSES).to(device)
    
    if not os.path.exists("best_model.pth"):
        print("Error: best_model.pth not found. Please run train.py first.")
        return

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((50, 200)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prediction = decode_prediction(logits)

    print("-" * 30)
    print(f"FILE: {os.path.basename(image_path)}")
    print(f"PREDICTED CAPTCHA: {prediction}")
    print("-" * 30)

if __name__ == "__main__":
    # Test on the first file from the test set if available, else ask user
    test_csv = 'dataset/test/labels.csv'
    if os.path.exists(test_csv):
        sample = pd.read_csv(test_csv).iloc[0]
        predict(os.path.join('dataset/test', sample['filename']))
    else:
        path = input("Enter path to captcha image: ")
        if os.path.exists(path):
            predict(path)
        else:
            print("Invalid path.")

print("NOTICE: This software is for academic research only. "
      "Use for automating student portals (SWI) is strictly prohibited.")