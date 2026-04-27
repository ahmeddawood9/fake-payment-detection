#!/bin/bash
# test_fakes.sh - Demonstrates catching fabricated IDs and structurally invalid receipts.

echo "--- RUNNING FRAUD DETECTION ON FAKE/FABRICATED SCREENSHOTS ---"

# Use find to handle spaces in filenames
find "Payment_screenshots/Fake_screenshots" -maxdepth 1 -name "*.jpeg" -o -name "*.png" | while read -r img; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        echo "Testing Fabricated Image: $filename"
        echo -e "1\n$img" | python fake_screenshot_detector/fake_screenshot_detector_ocr.py | grep -E "Verdict|Fraud Probability|Transaction ID|System Note"
        echo "------------------------------------------------"
    fi
done
