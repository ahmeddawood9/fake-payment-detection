# Fake Financial Screenshot Detector (Pakistani Banks & Payment Apps)

This project detects **fake payment screenshots** shown by students in a **university café**, using **AI + OCR**.  
It supports screenshots from **Pakistani banks and payment apps** such as:

- EasyPaisa
- JazzCash
- NayaPay
- SadaPay
- Meezan Bank
- Faysal Bank
- Raast, and others.

The system checks if a screenshot is **real** or **fake** and outputs a **fraud score**.

---

## 🎯 Problem

Students sometimes show **edited or fake payment screenshots** to staff in cafés.  
Verification by the naked eye is difficult; this tool provides:
- **OCR** (reads text from the screenshot).  
- **Duplicate Detection** (prevents using the same screenshot twice).
- **AI-based layout/classification** (CNN model).  
- **Heuristic fraud score**.

---

## 🔧 How it Works

### 1. Input
The system accepts image uploads via a **Web Interface** or **CLI**. It can also capture images via a **Live Camera** (CLI mode).

### 2. OCR & Heuristics
- Uses **Tesseract OCR** to extract:
  - **Amount**, **Transaction ID**, **Date/Time**.
  - Keywords like "successful", "sent", "approved".
- Checks for **Duplicate Transaction IDs** against a local history (`transaction_history.json`).

### 3. AI-based CNN model
- A lightweight **CNN** (PyTorch) analyzes the visual layout.
- Detects inconsistencies in UI patterns common in fake screenshots.

### 4. Final Verdict
- Combines AI analysis and OCR heuristics to provide:
  - **Verdict**: `REAL` / `FAKE`
  - **Fraud Score %**
  - **Detected App** & **Estimated Amount**

---

##  Getting Started

### 📋 Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR**:
   - **Ubuntu**: `sudo apt install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

### ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-screenshot-detector.git
   cd fake-screenshot-detector
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

##  Usage

### 1. Web Interface (Recommended)
Launch the Flask application for a user-friendly browser-based experience:
```bash
python app.py
```
Then visit `http://localhost:5000` in your browser.

### . CLI Mode
Run the detector directly from the terminal:
```bash
python fake_screenshot_detector/fake_screenshot_detector_ocr.py
```
- **Option 1**: Live Camera mode (Capture via webcam).
- **Option 2**: Image File mode (Provide path to a file).

---

##  Project Structure

```text
├── app.py                      # Flask Web Application entry point
├── fake_screenshot_detector/   # Core logic
│   └── fake_screenshot_detector_ocr.py  # OCR and CNN implementation
├── templates/                  # HTML templates for Web UI
├── transaction_history.json    # Local database of processed transactions
├── Payment_screenshots/        # Dataset for testing (Real/Fake)
└── requirements.txt            # Python dependencies
```

---

##  Technical Stack

- **Backend**: Python, Flask
- **OCR**: `pytesseract` (Tesseract)
- **Computer Vision**: `OpenCV`, `Pillow`
- **Deep Learning**: `PyTorch` (CNN)
- **Frontend**: HTML5, Vanilla CSS

---

##  Note
This project was developed for an AI Lab assignment. The CNN model provided is a lightweight implementation designed for demonstration purposes. In a production environment, training on a larger, more diverse dataset of Pakistani banking app screenshots is recommended.
