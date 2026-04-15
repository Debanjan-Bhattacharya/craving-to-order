import json

with open("dataset_enriched.json") as f:
    data = json.load(f)

# Collect all enriched dishes
all_dishes = []
for restaurant in data:
    for dish in restaurant["dishes"]:
        if "cuisine_type" in dish:
            all_dishes.append((dish, restaurant["name"]))

print(f"Total enriched dishes: {len(all_dishes)}\n")

NEW_FIELDS = ["cuisine_type", "cuisine_region", "cooking_method", "health_tags",
              "calorie_band", "serving_format", "holds_well", "portability",
              "time_affinity", "occasion_tags", "dietary_tags"]

for i, (dish, r_name) in enumerate(all_dishes, 1):
    print(f"\n[{i}] {dish['name']} @ {r_name} | ₹{dish['price']}")
    print(f"  tags: {dish.get('tags', [])}")
    for field in NEW_FIELDS:
        print(f"  {field}: {dish.get(field, 'MISSING')}")
