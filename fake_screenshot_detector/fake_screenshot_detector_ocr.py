import cv2
import pytesseract
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import os
import sys
import time
import json
from datetime import datetime

# ---------------------------- CONFIG ----------------------------

# List of known Pakistani financial/payment apps
# Used to detect which app the receipt belongs to
PAK_APPS = [
    "Easypaisa", "JazzCash", "NayaPay", "SadaPay", "Meezan Bank",
    "Faysal Bank", "UBL", "HBL", "Bank Alfalah", "Raast"
]

# Keywords commonly found in transaction receipts
# Used to determine if text looks like a real receipt
RECEIPT_KEYWORDS = [
    "successful", "sent", "transaction", "id", "ref", "amount", "pkr", "rs",
    "date", "time", "receiver", "sender", "paid", "approved", "completed"
]

# File where previous transactions are stored (for duplicate detection)
HISTORY_FILE = "transaction_history.json"

# ---------------------------- UTILS ----------------------------

# Function to print colored logs for better CLI visibility
def log_status(msg, type="INFO"):
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "WARNING": "\033[93m", "ERROR": "\033[91m", "RESET": "\033[0m"}
    print(f"{colors.get(type, '')}[{type}] {msg}{colors['RESET']}")

# Check if Tesseract OCR engine is installed
def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False

# Load previous transaction history from JSON file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

# Save a new verified transaction to history
def save_transaction(tx_id, amount, app):
    history = load_history()
    history[tx_id] = {
        "amount": amount,
        "app": app,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ---------------------------- STRUCTURAL VERIFICATION ----------------------------

# This function validates transaction ID structure based on detected app
def validate_id_structure(app_str, tx_id):
    if tx_id == "Unknown":
        return False, "Transaction ID not found or unreadable"

    # Convert app string into list for easy comparison
    apps = [a.strip().lower() for a in app_str.split(",")]

    # Rule 1: NayaPay → expects long alphanumeric (hex-like)
    if "nayapay" in apps:
        if len(tx_id) >= 20 and any(c.isalpha() for c in tx_id):
            return True, "Valid NayaPay Structure"
        return False, f"Invalid NayaPay ID (Expected 20+ hex, got {len(tx_id)})"

    # Rule 2: Easypaisa → exactly 11 digits
    if "easypaisa" in apps:
        if tx_id.isdigit() and len(tx_id) == 11:
            return True, "Valid Easypaisa Structure"
        return False, f"Invalid Easypaisa ID (Expected 11 digits, got {len(tx_id)})"

    # Rule 3: SadaPay → minimum 6 characters
    if "sadapay" in apps:
        if len(tx_id) >= 6:
            return True, "Valid SadaPay Structure"
        return False, "Invalid SadaPay ID structure"

    # Default rule for other apps
    if len(tx_id) >= 8:
        return True, "Generic Structure Passed"

    return False, "Transaction ID too short/atypical"

# ---------------------------- CNN MODEL ----------------------------

# CNN model to classify if screenshot looks real or fake visually
class ScreenshotCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # First convolution layer (extract low-level features)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        # Second convolution layer (deeper features)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # Max pooling reduces spatial size
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout prevents overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply conv → ReLU → pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten tensor for fully connected layer
        x = x.view(-1, 64 * 56 * 56)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# ---------------------------- DETECTION ENGINE ----------------------------

# Extract text from image using OCR
def extract_text(img_path_or_frame):
    if not check_tesseract():
        return "DEMO MODE: Tesseract not found."

    # If input is path → open image
    if isinstance(img_path_or_frame, str):
        img = Image.open(img_path_or_frame)
    else:
        # If input is frame → convert OpenCV (BGR) to RGB
        img = Image.fromarray(cv2.cvtColor(img_path_or_frame, cv2.COLOR_BGR2RGB))

    # Run OCR
    return pytesseract.image_to_string(img)

# Analyze extracted text and compute OCR-based score
def analyze_text(text):
    score = 0.0
    detected_apps = []

    # Detect app names inside text
    for app in PAK_APPS:
        if app.lower() in text.lower():
            detected_apps.append(app)

    # Convert detected apps list to string
    app_str = ", ".join(detected_apps) if detected_apps else "Unknown"

    # If app detected → increase confidence
    if detected_apps: score += 0.4

    # Check how many receipt keywords are present
    matches = [kw for kw in RECEIPT_KEYWORDS if kw.lower() in text.lower()]
    if matches:
        keyword_ratio = len(matches) / 8.0
        score += min(0.4, keyword_ratio)

    # Extract transaction amount using regex
    amount_match = re.search(r"(?:Rs|PKR|Amount)[:\s\.]*([\d,.]+)", text, re.IGNORECASE)
    amount = amount_match.group(1) if amount_match else "Unknown"
    if amount != "Unknown": score += 0.2

    # ---------------- ID EXTRACTION ----------------

    tx_id = "Unknown"

    # Patterns for transaction ID
    patterns = [
        r"(?:Transaction\sID|TID|ID#|Reference\snumber)[:\s#]+([a-zA-Z0-9-]{6,30})",
        r"(?:ID|Ref)[:\s#]*([a-zA-Z0-9-]{10,30})"
    ]

    # Try extracting ID using patterns
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            potential = match.group(1).strip()

            # Filter out invalid words and phone numbers
            if len(potential) > 5:
                if potential.lower() not in ["successful", "complete", "sent", "status", "amount", "number", "erence"]:
                    if not (potential.startswith("03") or potential.startswith("92")):
                        tx_id = potential
                        break

    # Fallback for Easypaisa (11-digit numbers)
    if tx_id == "Unknown" and "easypaisa" in app_str.lower():
        standalone = re.findall(r"\b(\d{11})\b", text)
        for num in standalone:
            if not (num.startswith("03") or num.startswith("92")):
                tx_id = num
                break

    return score, app_str, amount, tx_id

# Get CNN-based probability that image is real
def get_cnn_score(model, device, frame):
    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(frame).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)

        # Convert logits → probabilities
        probs = torch.softmax(outputs, dim=1)

        # Return probability of "real" class
        return probs[0][1].item()

# Main pipeline that processes image
def process_image(model, device, img):
    log_status("Analyzing visual patterns...")

    # CNN-based visual analysis
    cnn_score = get_cnn_score(model, device, img)

    log_status("Reading receipt details...")

    # OCR text extraction
    text = extract_text(img)

    # Text-based analysis
    ocr_score, app, amount, tx_id = analyze_text(text)

    # Structural validation of transaction ID
    is_valid_struct, struct_msg = validate_id_structure(app, tx_id)

    # Check if transaction is already used (duplicate attack)
    is_duplicate = False
    if tx_id != "Unknown":
        history = load_history()
        if tx_id in history:
            is_duplicate = True

    # ---------------- DECISION ENGINE ----------------

    fraud_prob = 0.0
    verdict = ""

    # Case 1: Duplicate → definitely fraud
    if is_duplicate:
        fraud_prob = 1.0
        verdict = "🚨 DUPLICATE DETECTED"

    # Case 2: Invalid structure → likely fake
    elif not is_valid_struct:
        fraud_prob = 0.75 + (cnn_score * 0.25)
        verdict = "❌ FAKE / FABRICATED"

    # Case 3: Missing info → suspicious
    elif app == "Unknown" or amount == "Unknown":
        fraud_prob = 0.60 + (cnn_score * 0.4)
        verdict = "❌ FAKE / SUSPICIOUS"

    # Case 4: Valid → combine OCR + CNN scores
    else:
        fraud_prob = max(0.02, (1.0 - ocr_score) * 0.7 + (cnn_score * 0.3))
        verdict = "✅ REAL / VALID" if fraud_prob < 0.45 else "❌ FAKE / SUSPICIOUS"

    # Save valid transactions
    if verdict == "✅ REAL / VALID" and tx_id != "Unknown":
        save_transaction(tx_id, amount, app)

    return {
        "verdict": verdict,
        "score": fraud_prob * 100,
        "app": app,
        "amount": amount,
        "tx_id": tx_id,
        "msg": struct_msg
    }

# ---------------------------- DETECTOR CLASS ----------------------------

class FakeScreenshotDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ScreenshotCNN().to(self.device)
        # In a real scenario, you'd load weights here:
        # self.model.load_state_dict(torch.load("model_weights.pth", map_location=self.device))
        self.model.eval()

    def analyze(self, img_path_or_frame):
        if isinstance(img_path_or_frame, str):
            img = cv2.imread(img_path_or_frame)
            if img is None:
                return {"error": "Could not read image"}
        else:
            img = img_path_or_frame
            
        return process_image(self.model, self.device, img)

# ---------------------------- MAIN INTERFACE ----------------------------

# Displays formatted output
def display_result(result):
    if "error" in result:
        log_status(result["error"], "ERROR")
        return

    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*17 + "DETECTION REPORT SUMMARY" + " "*17 + "║")
    print("╠" + "═"*58 + "╣")
    print(f"║  Verdict         : {result['verdict']:<38} ║")
    print(f"║  Fraud Probability: {result['score']:>5.1f}%{' '*32} ║")
    print(f"║  Detected App    : {result['app']:<38} ║")
    print(f"║  Extracted Amount: Rs {result['amount']:<35} ║")
    print(f"║  Transaction ID  : {result['tx_id']:<38} ║")
    print(f"║  System Note     : {result['msg']:<38} ║")
    print("╚" + "═"*58 + "╝")

    # Alerts based on result
    if "DUPLICATE" in result['verdict']:
        log_status("ALERT: Replay attack detected!", "ERROR")
    elif "FAKE" in result['verdict']:
        log_status(f"ALERT: {result['msg']}", "WARNING")
    else:
        log_status("Result: Verified and logged.", "SUCCESS")

# Main entry point
def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    print("="*60)
    print("   UNIVERSITY CAFÉ - FAKE SCREENSHOT DETECTOR v2.0")
    print("   Strict Structural Verification Enabled")
    print("="*60)

    detector = FakeScreenshotDetector()

    # Menu
    print("\n[1] Analyze Image File")
    print("[2] Live Camera Scan")
    print("[3] Exit")

    choice = input("\nSelect an option: ").strip()

    # Option 1: Load image from file
    if choice == '1':
        path = input("Enter image path: ").strip()
        if os.path.exists(path):
            res = detector.analyze(path)
            display_result(res)
        else:
            log_status("File not found!", "ERROR")

    # Option 2: Use live camera
    elif choice == '2':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_status("Camera error!", "ERROR")
            return

        log_status("Scanning... Press 'S' to capture.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            cv2.imshow("Café Scanner", frame)

            # Press 's' to capture frame
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.destroyAllWindows()
                res = detector.analyze(frame)
                display_result(res)
                break

        cap.release()
        cv2.destroyAllWindows()

# Run program
if __name__ == "__main__":
    main()
