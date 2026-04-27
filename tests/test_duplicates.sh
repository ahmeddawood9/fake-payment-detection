#!/bin/bash
# test_duplicates.sh - Demonstrates catching someone reusing the same screenshot.

echo "--- PREPARING DATABASE WITH AUTHENTIC DATA ---"
rm -f transaction_history.json

# Pick one image from each app to "store" in the system
sample_images=(
    "Payment_screenshots/Easypaisa/easypaisa.jpeg"
    "Payment_screenshots/Nayapay/Nayapay.jpeg"
    "Payment_screenshots/Sadapay/WhatsApp Image 2026-04-26 at 10.20.18 PM.jpeg"
)

for img in "${sample_images[@]}"; do
    if [ -f "$img" ]; then
        echo "Storing $img in database..."
        echo -e "1\n$img" | python fake_screenshot_detector/fake_screenshot_detector_ocr.py > /dev/null
    fi
done

echo -e "\n--- ATTEMPTING TO RE-USE THE SAME SCREENSHOTS (DUPLICATES) ---"
for img in "${sample_images[@]}"; do
    if [ -f "$img" ]; then
        echo "Processing Duplicate Attempt: $img"
        echo -e "1\n$img" | python fake_screenshot_detector/fake_screenshot_detector_ocr.py | grep -E "Verdict|Transaction ID|System Note"
        echo "------------------------------------------------"
    fi
done
