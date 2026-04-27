#!/bin/bash
# test_authentic.sh - Verifies that real screenshots are accepted correctly.

echo "--- RESETTING DATABASE ---"
rm -f transaction_history.json

folders=("Easypaisa" "Nayapay" "Sadapay")

echo "--- RUNNING AUTHENTIC TESTS ---"
for folder in "${folders[@]}"; do
    echo "Testing folder: $folder"
    # Use find to handle spaces in filenames and limit to 2
    find "Payment_screenshots/$folder" -maxdepth 1 -name "*.jpeg" -o -name "*.png" | head -n 2 | while read -r img; do
        if [ -f "$img" ]; then
            echo "Processing: $img"
            echo -e "1\n$img" | python fake_screenshot_detector/fake_screenshot_detector_ocr.py | grep -E "Verdict|Detected App|Transaction ID|System Note"
            echo "------------------------------------------------"
        fi
    done
done
