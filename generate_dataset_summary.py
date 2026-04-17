"""
generate_dataset_summary.py

Reads dataset_enriched.json and generates a concise summary
of available cuisines, restaurants, and dish types.

Run this once after each dataset update:
    python generate_dataset_summary.py

Outputs: dataset_summary.txt
retrieval.py reads this at startup to inform query expansion.
"""

import json
from collections import Counter

DATASET_PATH = "dataset_enriched.json"
OUTPUT_PATH = "dataset_summary.txt"


def generate_summary(dataset_path):
    with open(dataset_path, encoding='utf-8') as f:
        data = json.load(f)

    restaurants = []
    cuisine_types = Counter()
    category_types = Counter()
    all_tags = Counter()
    restaurant_types = {}

    for r in data:
        rname = r.get('name', '')
        rcuisine = r.get('cuisine', [])
        restaurants.append(rname)

        for dish in r.get('dishes', []):
            # Cuisine types
            ct = dish.get('cuisine_type', '')
            if ct:
                cuisine_types[ct] += 1

            # Category types
            cat = dish.get('category_type', '')
            if cat:
                category_types[cat] += 1

            # Tags
            for tag in dish.get('tags', []):
                if tag not in ['veg', 'non-veg', 'mild', 'medium-spicy',
                               'spicy', 'very-spicy', 'filling', 'light',
                               'budget', 'premium', 'popular', 'bestseller']:
                    all_tags[tag] += 1

        # Categorise by most common cuisine_type across dishes
        dish_cuisines = [d.get('cuisine_type', '') for d in r.get('dishes', []) if d.get('cuisine_type')]
        dish_cuisines_clean = [c for c in dish_cuisines if c and c != 'null']
        if dish_cuisines_clean:
            from collections import Counter as _Counter
            primary = _Counter(dish_cuisines_clean).most_common(1)[0][0]
        elif rcuisine:
            primary = rcuisine[0]
        else:
            primary = 'other'
        if primary not in restaurant_types:
            restaurant_types[primary] = []
        restaurant_types[primary].append(rname)

    # Build summary
    lines = []

    # Cuisine types available
    top_cuisines = [k for k, v in cuisine_types.most_common(20)]
    lines.append(f"Available cuisine types: {', '.join(top_cuisines)}")

    # Dish categories
    top_cats = [k for k, v in category_types.most_common(15)]
    lines.append(f"Available dish categories: {', '.join(top_cats)}")

    # Notable dish types from tags
    # Only keep food-specific tags, remove metadata noise
    NOISE_TAGS = {'heavy', 'light', 'main', 'meal', 'occasion', 'comfort',
                  'rich', 'healthy', 'indulgent', 'starter', 'snack', 'value',
                  'street food', 'rice', 'noodles', 'roll', 'wrap', 'combo',
                  'complete meal', 'medium', 'handheld', 'breakfast', 'dessert',
                  'beverage', 'budget', 'premium', 'popular', 'bestseller',
                  'filling', 'north-indian', 'south-indian', 'chinese',
                  'fast-food', 'tibetan', 'italian', 'indo-chinese', 'mexican',
                  'indian', 'creamy'}
    notable_tags = [k for k, v in all_tags.most_common(50)
                    if k not in NOISE_TAGS and len(k) > 2]
    lines.append(f"Notable dish types: {', '.join(notable_tags[:20])}")

    # Restaurant names grouped by type
    lines.append(f"\nRestaurants by category:")
    for cuisine_cat, rnames in sorted(restaurant_types.items()):
        lines.append(f"  {cuisine_cat}: {', '.join(rnames)}")

    # All restaurant names flat
    lines.append(f"\nAll restaurants ({len(restaurants)} total): {', '.join(restaurants)}")

    return '\n'.join(lines)


if __name__ == "__main__":
    print("Generating dataset summary...")
    summary = generate_summary(DATASET_PATH)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(summary)

    print(f"Saved to {OUTPUT_PATH}")
    print("\nPreview:")
    print(summary[:800])
