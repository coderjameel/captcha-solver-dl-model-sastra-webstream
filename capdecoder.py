import json
import base64
import io
from PIL import Image
import pytesseract

def decode_with_tesseract(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    decoded_list = []
    
    for i in range(2000):
        if i < len(data):
            b64_string = data[i].get("captcha", "")
            try:
                img_bytes = base64.b64decode(b64_string)
                img = Image.open(io.BytesIO(img_bytes))
                
                # psm 8 tells tesseract to treat the image as a single word
                captcha_text = pytesseract.image_to_string(img, config='--psm 8').strip()
                decoded_list.append(captcha_text)
            except Exception as e:
                decoded_list.append(None)
                
    return decoded_list

if __name__ == "__main__":
    # Run the function and get the list
    captcha_list = decode_with_tesseract('data.json')
    
    # Print the resulting Python list
    print("\n--- Final Decoded List ---")
    print(captcha_list)