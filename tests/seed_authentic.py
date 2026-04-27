import os
import cv2
import torch
import sys

# Add the directory to path so we can import the detector
sys.path.append(os.path.join(os.getcwd(), 'fake_screenshot_detector'))
from fake_screenshot_detector_ocr import process_image, ScreenshotCNN, log_status

def seed_database():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScreenshotCNN().to(device)
    
    # Define directories containing authentic screenshots
    authentic_dirs = [
        "Payment_screenshots/Easypaisa",
        "Payment_screenshots/Nayapay",
        "Payment_screenshots/Sadapay"
    ]
    
    total_processed = 0
    total_saved = 0

    print("--- SEEDING DATABASE WITH AUTHENTIC SCREENSHOTS ---")

    for folder in authentic_dirs:
        if not os.path.exists(folder):
            log_status(f"Directory {folder} not found, skipping...", "WARNING")
            continue
            
        log_status(f"Processing folder: {folder}", "INFO")
        
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    log_status(f"Could not read {filename}", "ERROR")
                    continue
                
                # process_image automatically saves to transaction_history.json if verdict is REAL / VALID
                result = process_image(model, device, img)
                
                total_processed += 1
                if "REAL" in result['verdict']:
                    total_saved += 1
                    log_status(f"Saved: {filename} (ID: {result['tx_id']})", "SUCCESS")
                else:
                    log_status(f"Skipped (Verdict: {result['verdict']}): {filename}", "WARNING")

    print("\n" + "="*40)
    print(f"Seeding Complete!")
    print(f"Total Images Processed: {total_processed}")
    print(f"Total Transactions Added: {total_saved}")
    print("="*40)

if __name__ == "__main__":
    seed_database()
