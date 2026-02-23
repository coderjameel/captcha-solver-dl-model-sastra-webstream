import os
import csv
import base64
import time
from openai import OpenAI
from PIL import Image
import io

# 1. Configuration
OPENROUTER_API_KEY = "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"
PRIMARY_MODEL = "google/gemini-2.0-flash-001"
FALLBACK_MODEL = "qwen/qwen-2.5-vl-72b-instruct" # Less restrictive
FOLDER_PATH = 'captchas'
OUTPUT_FILE = 'data.csv'

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def encode_png_properly(image_path):
    """Converts PNG to a flat JPEG to remove transparency/alpha issues."""
    with Image.open(image_path) as img:
        # PNGs often have alpha channels that turn black in some viewers
        # We convert to RGB with a white background
        canvas = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            canvas.paste(img, mask=img.split()[3])
        else:
            canvas.paste(img)
            
        buffered = io.BytesIO()
        canvas.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

# 2. Filter for PNG and limit to 100
image_files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith('.png')])
image_files = image_files[1001:2000]

print(f"Processing {len(image_files)} PNG files...")

with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'captcha_value'])

    for index, filename in enumerate(image_files):
        img_path = os.path.join(FOLDER_PATH, filename)
        
        try:
            b64_img = encode_png_properly(img_path)
            
            def get_prediction(model_name):
                return client.chat.completions.create(
                    model=model_name,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What are the characters in this captcha? Output ONLY the text."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        ]
                    }],
                    temperature=0.1
                ).choices[0].message.content.strip()

            # Attempt with Gemini
            result = get_prediction(PRIMARY_MODEL)

            # If Gemini gives a long "Safety/Refusal" response, try Qwen
            if len(result) > 10 or "unable" in result.lower():
                print(f"[{index+1}] Gemini refused/failed. Retrying with Qwen...")
                result = get_prediction(FALLBACK_MODEL)

            writer.writerow([filename, result])
            print(f"[{index+1}/100] {filename}: {result}")
            time.sleep(1) # Safety delay

        except Exception as e:
            print(f"Failed {filename}: {e}")
            writer.writerow([filename, "ERROR"])

print(f"\nTask Complete! Data saved to {OUTPUT_FILE}")
