"""
build_dish_lookup.py
Builds a dish aggregation lookup from enriched_festival.csv.
Groups all vectors of the same dish across restaurants.

Output: dish_lookup.json
Format: {
    "butter chicken": {
        "restaurant_count": 6,
        "price_min": 330,
        "price_max": 587,
        "all_restaurants": ["Postman Kitchen", "Chawla's Chicken", ...],
        "cuisine_type": "north-indian",
        "is_veg": "non-veg"
    },
    ...
}

Run from: C:\\craving-to-order
Usage:    python build_dish_lookup.py
"""

import csv
import json
from collections import defaultdict

INPUT_FILE  = "enriched_festival.csv"
OUTPUT_FILE = "dish_lookup.json"


def build_lookup():
    with open(INPUT_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {INPUT_FILE}")

    # Group by dish_name (case-insensitive)
    dish_groups = defaultdict(list)
    for row in rows:
        key = row.get("dish_name", "").strip().lower()
        if key:
            dish_groups[key].append(row)

    lookup = {}
    for dish_key, instances in dish_groups.items():
        prices = []
        restaurants = []
        for inst in instances:
            try:
                p = float(inst.get("price", 0))
                if p > 0:
                    prices.append(p)
            except (ValueError, TypeError):
                pass
            r = inst.get("restaurant_name", "").strip()
            if r and r not in restaurants:
                restaurants.append(r)

        # Use first instance for dish-level metadata
        rep = instances[0]

        lookup[dish_key] = {
            "dish_name":        rep.get("dish_name", "").strip(),
            "restaurant_count": len(restaurants),
            "price_min":        int(min(prices)) if prices else 0,
            "price_max":        int(max(prices)) if prices else 0,
            "all_restaurants":  restaurants,
            "cuisine_type":     rep.get("cuisine_type", ""),
            "cuisine_region":   rep.get("cuisine_region", ""),
            "category_type":    rep.get("category_type", ""),
            "is_veg":           rep.get("is_veg", ""),
            "taste_tags":       rep.get("taste_tags", ""),
            "texture_tags":     rep.get("texture_tags", ""),
            "calorie_band":     rep.get("calorie_band", ""),
            "protein_band":     rep.get("protein_band", ""),
        }

    print(f"Built lookup for {len(lookup)} unique dishes")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_FILE}")

    # Summary
    multi = sum(1 for v in lookup.values() if v["restaurant_count"] > 1)
    print(f"\nDishes available at 2+ restaurants: {multi}")
    print(f"Sample — 'butter chicken':")
    bc = lookup.get("butter chicken")
    if bc:
        print(f"  Restaurants : {bc['restaurant_count']}")
        print(f"  Price range : ₹{bc['price_min']} – ₹{bc['price_max']}")
        print(f"  Restaurants : {bc['all_restaurants']}")


if __name__ == "__main__":
    build_lookup()
