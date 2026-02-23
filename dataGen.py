import os
import random
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- Config ---
NUM_SAMPLES = 50000  # Generate 50k images to blast the DGX
IMG_WIDTH, IMG_HEIGHT = 200, 50
OUTPUT_DIR = 'synthetic_captchas/'
CSV_FILE = 'synthetic_data.csv'

# SASTRA characters
CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def generate_random_string(length=5):
    """Generates a random alphanumeric string (5 or 6 chars)."""
    return ''.join(random.choice(CHARS) for _ in range(length))

def create_synthetic_captcha(text, filename):
    # 1. Create a pure white background
    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    # 2. Load a thick sans-serif font
    # Note: On Linux/DGX, you might need to point this to a specific TTF file
    # like '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36) 
    except IOError:
        font = ImageFont.load_default()

    # 3. Draw tightly spaced text
    # We add a slight random X offset, and keep Y centered
    start_x = random.randint(10, 30)
    for char in text:
        # Draw character
        draw.text((start_x, 2), char, font=font, fill=0)
        # Advance X by a squished amount to simulate overlap (e.g., "ymm")
        start_x += random.randint(22, 28) 

    # 4. Draw the signature SASTRA thick black distractor line
    # It usually starts middle-left and ends bottom-right
    line_start = (random.randint(0, 20), random.randint(30, 40))
    line_end = (random.randint(160, 190), random.randint(35, 45))
    draw.line([line_start, line_end], fill=0, width=random.randint(2, 4))

    # Save the image
    img.save(os.path.join(OUTPUT_DIR, filename))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    data = []
    print(f"Generating {NUM_SAMPLES} synthetic SASTRA captchas...")
    
    for i in tqdm(range(NUM_SAMPLES)):
        # Randomly choose 5 or 6 characters
        length = random.choice([5, 6])
        text = generate_random_string(length)
        filename = f"synth_{i}.png"
        
        create_synthetic_captcha(text, filename)
        data.append({"filename": filename, "captcha_value": text})
        
    # Save perfectly clean labels
    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False)
    print(f"Done! Dataset saved to {OUTPUT_DIR} and labels to {CSV_FILE}")

if __name__ == "__main__":
    main()