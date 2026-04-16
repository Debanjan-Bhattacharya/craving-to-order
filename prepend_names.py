"""
prepend_names.py

Reads restaurant names from Excel and prepends them to menu txt files.

Usage:
    python prepend_names.py restaurants.xlsx menus/

Excel format expected:
    Column A: Restaurant ID (R18, R19, etc.)
    Column B: Restaurant Name
    Column C: Location
    Column D: Cuisine
"""

import os
import sys

def prepend_names(excel_path, menus_folder):
    try:
        import openpyxl
    except ImportError:
        print("Installing openpyxl...")
        os.system("pip install openpyxl --break-system-packages")
        import openpyxl

    # Load Excel
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    # Build ID → name mapping
    restaurant_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):  # skip header
        rid = str(row[0]).strip().upper() if row[0] else None
        name = str(row[1]).strip() if row[1] else None
        cuisine = str(row[3]).strip() if row[3] else ""
        if rid and name:
            restaurant_map[rid] = {"name": name, "cuisine": cuisine}

    print(f"Loaded {len(restaurant_map)} restaurants from Excel")

    # Process txt files in menus folder
    txt_files = [
        f for f in os.listdir(menus_folder)
        if f.endswith('.txt')
        and '_part' not in f
        and '_clean' not in f
    ]

    updated = []
    skipped = []
    not_found = []

    for filename in sorted(txt_files):
        rid = os.path.splitext(filename)[0].upper()
        filepath = os.path.join(menus_folder, filename)

        if rid not in restaurant_map:
            not_found.append(filename)
            continue

        info = restaurant_map[rid]

        # Read existing content
        with open(filepath, encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Skip if already has restaurant name prepended
        if content.startswith("Restaurant:"):
            skipped.append(filename)
            continue

        # Prepend restaurant info
        header = (
            f"Restaurant: {info['name']}\n"
            f"Location: Delhi NCR\n"
            f"Cuisine: {info['cuisine']}\n\n"
        )

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)

        updated.append(f"{filename} → {info['name']}")

    print(f"\nUpdated: {len(updated)}")
    for u in updated:
        print(f"  {u}")

    if skipped:
        print(f"\nSkipped (already has name): {len(skipped)}")
        for s in skipped:
            print(f"  {s}")

    if not_found:
        print(f"\nNot found in Excel: {len(not_found)}")
        for n in not_found:
            print(f"  {n}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepend_names.py restaurants.xlsx menus/")
        sys.exit(1)

    excel_path = sys.argv[1]
    menus_folder = sys.argv[2]

    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found")
        sys.exit(1)

    if not os.path.exists(menus_folder):
        print(f"Error: Menus folder '{menus_folder}' not found")
        sys.exit(1)

    prepend_names(excel_path, menus_folder)
