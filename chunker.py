import json

def dish_to_text(dish, restaurant_name, cuisine, avg_cost_for_two):
    """Convert a dish dict to a natural language chunk for embedding."""
    tags = dish.get("tags", [])
    diet = next((t for t in tags if t in ["veg", "non-veg"]), "")
    price = dish["price"]

    # Include enriched fields in text if available
    cuisine_type = dish.get("cuisine_type", "")
    cooking_method = dish.get("cooking_method", "")
    health_tags = dish.get("health_tags", [])
    time_affinity = dish.get("time_affinity", [])

    enriched = ""
    if cuisine_type:
        enriched += f" Cuisine style: {cuisine_type}."
    if cooking_method:
        enriched += f" Cooking: {cooking_method}."
    if health_tags:
        enriched += f" Health profile: {', '.join(health_tags)}."
    if time_affinity:
        enriched += f" Best for: {', '.join(time_affinity)}."

    chunk = (
        f"{dish['name']} is a {diet} dish from {restaurant_name}, "
        f"priced at ₹{price}. "
        f"It is a {', '.join(tags)} preparation. "
        f"{dish['description']} "
        f"Cuisine: {', '.join(cuisine)}.{enriched} "
        f"Average cost for two at this restaurant: ₹{avg_cost_for_two}."
    )
    return chunk


def build_chunks(dataset_path):
    with open(dataset_path) as f:
        data = json.load(f)

    chunks = []
    for restaurant in data:
        for dish in restaurant["dishes"]:
            chunk_text = dish_to_text(
                dish,
                restaurant["name"],
                restaurant["cuisine"],
                restaurant["avg_cost_for_two"]
            )

            # Build full metadata — include all enriched fields if present
            metadata = {
                "restaurant_id": restaurant["restaurant_id"],
                "restaurant_name": restaurant["name"],
                "dish_name": dish["name"],
                "price": dish["price"],
                "tags": dish["tags"],
                "cuisine": restaurant["cuisine"],
                "avg_cost_for_two": restaurant["avg_cost_for_two"],
                "category_type": dish.get("category_type", "main"),
            }

            # Add enriched fields if available
            enriched_fields = [
                "cuisine_type", "cuisine_region", "cooking_method",
                "health_tags", "calorie_band", "serving_format",
                "holds_well", "portability", "time_affinity",
                "occasion_tags", "dietary_tags"
            ]
            for field in enriched_fields:
                if field in dish:
                    metadata[field] = dish[field]

            chunks.append({
                "id": dish["id"],
                "text": chunk_text,
                "metadata": metadata
            })

    return chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset_enriched.json"
    chunks = build_chunks(path)

    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Total chunks: {len(chunks)}")
    print("\nSample chunk text:")
    print(chunks[0]["text"])
    print("\nSample metadata fields:")
    for k, v in chunks[0]["metadata"].items():
        print(f"  {k}: {v}")
