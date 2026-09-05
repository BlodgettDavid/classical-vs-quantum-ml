# tests/verify_csv_headers.py

import os
import sys
import csv

# Set up path to import FIELDNAMES directly from logger.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from utils.logger import FIELDNAMES

def verify_csv_headers(results_dir: str = "results"):
    target_dir = os.path.join(ROOT_DIR, results_dir)
    
    if not os.path.exists(target_dir):
        print(f"[!] Directory not found: {target_dir}")
        return

    csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
    
    if not csv_files:
        print(f"[!] No CSV files found in {target_dir}")
        return

    print(f"=== Verifying {len(csv_files)} CSV File(s) in '{results_dir}/' ===")
    print(f"Expected Column Count: {len(FIELDNAMES)}\n")

    all_passed = True

    for filename in csv_files:
        filepath = os.path.join(target_dir, filename)
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                print(f"❌ {filename}: File is empty.")
                all_passed = False
                continue

        missing = [col for col in FIELDNAMES if col not in header]
        extra = [col for col in header if col not in FIELDNAMES]
        order_match = (header == FIELDNAMES)

        if order_match:
            print(f"✅ {filename}: PERFECT MATCH ({len(header)} columns)")
        else:
            all_passed = False
            print(f"❌ {filename}: MISMATCH")
            if missing:
                print(f"   - Missing columns ({len(missing)}): {missing}")
            if extra:
                print(f"   - Unexpected columns ({len(extra)}): {extra}")
            if not missing and not extra and not order_match:
                print("   - All columns present, but column ordering is incorrect.")

    print("\n" + ("=" * 50))
    if all_passed:
        print("🎉 ALL CSV FILES ARE ALIGNED WITH LOGGER SCHEMA.")
    else:
        print("⚠️  SOME FILES DO NOT MATCH THE CURRENT SCHEMA.")

if __name__ == "__main__":
    verify_csv_headers()