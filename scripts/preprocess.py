import pandas as pd
import os

def load_raw_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def basic_clean(text):
    return text.strip()

def save_cleaned(data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')

if __name__ == "__main__":
    raw_path = "data/raw/telegram_messages.txt"
    cleaned_path = "data/processed/cleaned_text.txt"
    
    if os.path.exists(raw_path):
        raw_lines = load_raw_text(raw_path)
        cleaned = [basic_clean(line) for line in raw_lines if line.strip()]
        save_cleaned(cleaned, cleaned_path)
        print(f"[✓] Saved cleaned data to {cleaned_path}")
    else:
        print(f"[!] File not found: {raw_path}")
