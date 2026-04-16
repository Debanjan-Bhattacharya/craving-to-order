"""
menu_splitter.py

Cleans and splits large Zomato copy-paste menu txt files into
smaller chunks suitable for single API extraction passes.

Usage:
    python menu_splitter.py menus/R18.txt
    python menu_splitter.py menus/  (process all txt files in folder)
"""

import os
import sys
import re

MAX_CHUNK_CHARS = 1500

NOISE_PATTERNS = [
    r"Not eligible for coupons",
    r"\.\.\.read more",
    r"\.\.\. read more",
    r"Best seller",
    r"Best Seller",
    r"Must try",
    r"Must Try",
    r"^\d+\s*$",
    r"^Veg$",
    r"^Non-?veg$",
    r"^Non Veg$",
    r"Add to cart",
    r"Customize",
    r"Pure Veg",
    r"Jain available",
]


def fix_encoding(raw_text):
    """Fix UTF-8 mojibake from copy-pasting Zomato menus."""
    try:
        return raw_text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        raw_text = raw_text.replace('â‚¹', '₹')
        raw_text = raw_text.replace('â€™', "'")
        raw_text = raw_text.replace('â€"', '-')
        raw_text = raw_text.replace('â€˜', "'")
        return raw_text


def extract_header(text):
    """Extract Restaurant/Location/Cuisine header lines from top of text."""
    lines = text.split('\n')
    header_lines = []
    for line in lines:
        if line.startswith(('Restaurant:', 'Location:', 'Cuisine:')):
            header_lines.append(line)
        elif header_lines:
            break
    return '\n'.join(header_lines) + '\n\n' if header_lines else ''


def clean_menu(raw_text):
    """Fix encoding, remove noise, deduplicate consecutive repeated lines."""
    raw_text = fix_encoding(raw_text)
    lines = raw_text.split('\n')
    cleaned = []
    prev_line = None

    for line in lines:
        line = line.strip()

        if not line:
            if cleaned and cleaned[-1] != '':
                cleaned.append('')
            continue

        skip = False
        for pattern in NOISE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                skip = True
                break
        if skip:
            continue

        if line == prev_line:
            continue

        cleaned.append(line)
        prev_line = line

    while cleaned and cleaned[-1] == '':
        cleaned.pop()

    return '\n'.join(cleaned)


def split_into_chunks(cleaned_text, max_chars=MAX_CHUNK_CHARS):
    """Split cleaned menu into chunks. Splits on blank lines to avoid cutting mid-dish."""
    blocks = re.split(r'\n\s*\n', cleaned_text)
    blocks = [b.strip() for b in blocks if b.strip()]

    # Skip header blocks from chunking
    start_idx = 0
    for i, block in enumerate(blocks):
        if not any(block.startswith(p) for p in ('Restaurant:', 'Location:', 'Cuisine:')):
            start_idx = i
            break

    blocks = blocks[start_idx:]

    chunks = []
    current_chunk = []
    current_len = 0

    for block in blocks:
        block_len = len(block)

        if block_len > max_chars:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_len = 0
            lines = block.split('\n')
            sub_chunk = []
            sub_len = 0
            for line in lines:
                if sub_len + len(line) > max_chars and sub_chunk:
                    chunks.append('\n'.join(sub_chunk))
                    sub_chunk = [line]
                    sub_len = len(line)
                else:
                    sub_chunk.append(line)
                    sub_len += len(line)
            if sub_chunk:
                chunks.append('\n'.join(sub_chunk))
            continue

        if current_len + block_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [block]
            current_len = block_len
        else:
            current_chunk.append(block)
            current_len += block_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def process_file(filepath, output_dir=None):
    """Clean and split a single menu file."""
    if output_dir is None:
        output_dir = os.path.dirname(filepath) or '.'

    base = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, encoding='utf-8', errors='replace') as f:
        raw = f.read()

    print(f"\n{base}: {len(raw)} chars raw")

    cleaned = clean_menu(raw)
    print(f"  After cleaning: {len(cleaned)} chars")

    header = extract_header(cleaned)
    content_without_header = cleaned[len(header):] if header else cleaned

    if len(content_without_header) <= MAX_CHUNK_CHARS:
        out_path = os.path.join(output_dir, f"{base}_clean.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"  No split needed → {base}_clean.txt")
        return [out_path]

    chunks = split_into_chunks(cleaned)
    print(f"  Split into {len(chunks)} parts")

    output_paths = []
    for i, chunk in enumerate(chunks, 1):
        out_path = os.path.join(output_dir, f"{base}_part{i}.txt")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(header + chunk)
        print(f"  Part {i}: {len(chunk)} chars → {base}_part{i}.txt")
        output_paths.append(out_path)

    return output_paths


def process_folder(folder):
    """Process all original .txt files in a folder."""
    txt_files = [
        f for f in os.listdir(folder)
        if f.endswith('.txt')
        and '_part' not in f.lower()
        and '_clean' not in f.lower()
    ]

    if not txt_files:
        print(f"No txt files found in {folder}")
        return

    print(f"Found {len(txt_files)} menu files to process")
    all_outputs = []

    for filename in sorted(txt_files):
        filepath = os.path.join(folder, filename)
        outputs = process_file(filepath, folder)
        all_outputs.extend(outputs)

    print(f"\nDone. {len(all_outputs)} output files ready for batch_extract.py")
    print("\nNext step:")
    print("  python batch_extract.py menus/ extracted/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python menu_splitter.py menus/R18.txt")
        print("  python menu_splitter.py menus/")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        process_folder(target)
    elif os.path.isfile(target):
        process_file(target)
    else:
        print(f"Error: '{target}' not found")
        sys.exit(1)
