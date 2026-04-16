"""
run_merge.py

Merges extracted part JSONs and fixes restaurant names from Excel.

Usage:
    python run_merge.py
    python run_merge.py extracted/ restaurants.xlsx
"""

import os
import sys
import json
from batch_extract import merge_parts

output_folder = sys.argv[1] if len(sys.argv) > 1 else "extracted"
excel_path = sys.argv[2] if len(sys.argv) > 2 else "restaurants.xlsx"

# Step 1 — Merge parts
print("Merging parts...")
merge_parts(output_folder)

# Step 2 — Fix restaurant names from Excel if available
if not os.path.exists(excel_path):
    print(f"\nNo Excel file found at {excel_path} — skipping name fix.")
    print("Done.")
    sys.exit(0)

try:
    import openpyxl
except ImportError:
    os.system("pip install openpyxl --break-system-packages")
    import openpyxl

wb = openpyxl.load_workbook(excel_path)
ws = wb.active

restaurant_map = {}
for row in ws.iter_rows(min_row=2, values_only=True):
    rid = str(row[0]).strip().upper() if row[0] else None
    name = str(row[1]).strip() if row[1] else None
    if rid and name:
        restaurant_map[rid] = name

print(f"\nFixing restaurant names from Excel ({len(restaurant_map)} entries)...")

fixed = []
for filename in os.listdir(output_folder):
    if not filename.endswith('.json'):
        continue
    if '_PART' in filename.upper() or '_part' in filename:
        continue

    rid = os.path.splitext(filename)[0].upper()
    if rid not in restaurant_map:
        continue

    filepath = os.path.join(output_folder, filename)
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    correct_name = restaurant_map[rid]
    if data.get('name') != correct_name:
        old_name = data.get('name')
        data['name'] = correct_name
        # Also fix name in all dish IDs if needed
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        fixed.append(f"{rid}: '{old_name}' → '{correct_name}'")

if fixed:
    print(f"Fixed {len(fixed)} restaurant names:")
    for f in fixed:
        print(f"  {f}")
else:
    print("All names already correct.")

print("\nDone.")
