import json
import os
import sys

def merge_restaurants(existing_dataset_path, new_jsons_folder, output_path):
    """
    Merge new restaurant JSONs into existing dataset.
    Skips restaurants whose IDs already exist in the dataset.
    """

    # Load existing dataset
    try:
        with open(existing_dataset_path, encoding='utf-8') as f:
            existing = json.load(f)
        existing_ids = {r["restaurant_id"] for r in existing}
        print(f"Existing dataset: {len(existing)} restaurants ({', '.join(sorted(existing_ids))})")
    except FileNotFoundError:
        existing = []
        existing_ids = set()
        print("No existing dataset found. Starting fresh.")

    # Find all JSON files in folder
    new_files = sorted([
        f for f in os.listdir(new_jsons_folder)
        if f.endswith(".json")
        and not f.endswith("_ERROR.json")
        and "_PART" not in f.upper()
    ])

    if not new_files:
        print(f"No JSON files found in '{new_jsons_folder}'")
        return

    print(f"\nFound {len(new_files)} JSON files in '{new_jsons_folder}'")

    added = []
    skipped = []

    for filename in new_files:
        filepath = os.path.join(new_jsons_folder, filename)
        with open(filepath, encoding="utf-8") as f:
            restaurant = json.load(f)

        rid = restaurant.get("restaurant_id", "UNKNOWN")

        if rid in existing_ids:
            print(f"  [SKIP] {rid} — {restaurant.get('name')} already in dataset")
            skipped.append(rid)
            continue

        existing.append(restaurant)
        existing_ids.add(rid)
        dish_count = len(restaurant.get("dishes", []))
        print(f"  [ADD]  {rid} — {restaurant.get('name')} | {dish_count} dishes")
        added.append(rid)

    # Save merged dataset
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    total_dishes = sum(len(r["dishes"]) for r in existing)

    print(f"\n{'='*50}")
    print(f"MERGE COMPLETE")
    print(f"{'='*50}")
    print(f"Added: {len(added)} restaurants ({', '.join(added) if added else 'none'})")
    print(f"Skipped: {len(skipped)} (already existed)")
    print(f"Total restaurants: {len(existing)}")
    print(f"Total dishes: {total_dishes}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    existing = sys.argv[1] if len(sys.argv) > 1 else "dataset_enriched.json"
    folder = sys.argv[2] if len(sys.argv) > 2 else "extracted"
    output = sys.argv[3] if len(sys.argv) > 3 else "dataset_enriched.json"

    merge_restaurants(existing, folder, output)
