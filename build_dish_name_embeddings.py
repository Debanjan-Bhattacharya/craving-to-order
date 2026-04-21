"""
build_dish_name_embeddings.py
Embeds all unique dish names for fuzzy deduplication in similar dishes.
One-time operation. Output used at runtime by get_similar_dishes().

Input:  enriched_festival.csv
Output: dish_name_embeddings.json
Format: {"butter chicken": [0.023, -0.14, ...], ...}

Run from: C:\\craving-to-order
Usage:    python build_dish_name_embeddings.py
"""

import os
import csv
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INPUT_FILE  = "enriched_festival.csv"
OUTPUT_FILE = "dish_name_embeddings.json"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 100


def embed_batch(texts: list) -> list:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def build_embeddings():
    # Load all unique dish names
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unique_names = list({r["dish_name"].strip().lower() for r in rows if r.get("dish_name")})
    total = len(unique_names)
    print(f"Unique dish names: {total}")

    embeddings = {}
    for i in range(0, total, BATCH_SIZE):
        batch = unique_names[i: i + BATCH_SIZE]
        vectors = embed_batch(batch)
        for name, vec in zip(batch, vectors):
            embeddings[name] = vec
        if (i + BATCH_SIZE) % 500 == 0 or i + BATCH_SIZE >= total:
            print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total}...")
        time.sleep(0.1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"\nSaved {len(embeddings)} dish name embeddings to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build_embeddings()
