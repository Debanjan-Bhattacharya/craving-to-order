import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
BATCH_SIZE = 50
INPUT_PATH = "dataset.json"
OUTPUT_PATH = "dataset_enriched.json"

TAGGING_PROMPT = """You are a food metadata expert. Given a dish from a Delhi restaurant, generate metadata tags.

Return ONLY a JSON object with these exact fields. No explanation, no markdown.

{
  "cuisine_type": "<one of: north-indian, south-indian, indo-chinese, thai, tibetan, chinese, continental, lebanese, mughlai, fast-food, bakery, kerala, middle-eastern>",
  "cuisine_region": "<one of: punjabi, marathi, bengali, tamil, kerala, awadhi, hyderabadi, chettinad, kashmiri, karnataka, or null>",
  "cooking_method": "<one of: fried, grilled, steamed, boiled, simmered, baked, tandoor, raw, slow-cooked, tawa, wok-tossed, mixed, dum>",
  "health_tags": ["<ONLY from: low-cal, high-protein, gut-friendly, light, heavy>"],
  "calorie_band": "<one of: low, medium, high>",
  "serving_format": "<one of: finger-food, handheld, bowl, platter, sit-down, sharing>",
  "holds_well": <true or false>,
  "portability": <true or false>,
  "time_affinity": ["<ONLY from: breakfast, lunch, snack, dinner, late-night>"],
  "occasion_tags": ["<from: party, cheat-day, date-night, holi, eid, sehri, iftar, vrat, diwali — or empty array>"],
  "dietary_tags": ["<from: jain, vrat-friendly, no-onion-garlic, gluten-free — or empty array>"]
}

STRICT RULES — follow exactly:

cuisine_type rules:
- street-food is NOT a valid cuisine_type — assign the dish's actual culinary tradition
- Vada Pav, Pav Bhaji = north-indian
- Thukpa = tibetan
- Momos (any filling, any cooking method) = tibetan
- Manchurian, Chilli Chicken, Chilli Paneer, Hakka Noodles, Fried Rice = indo-chinese
- Andhra Dragon, Chettinad, Telugu dishes = south-indian (NOT indo-chinese)
- Thai curry, Tomyum soup = thai
- Clear soup, Sweet Corn soup, Hot and Sour soup, Manchow soup = chinese
- Continental / Italian / Western dishes = continental
- Biryani at a Kerala restaurant = south-indian (not north-indian)
- Do NOT use street-food as a cuisine_type

cuisine_region rules:
- Rogan Josh = kashmiri always
- Hyderabadi biryani = hyderabadi always
- Bisibele Bhaat = karnataka
- Chettinad dishes = chettinad
- Awadhi / Lucknowi dishes = awadhi
- Biryani at Kerala restaurant = kerala region
- Only assign when clearly regional — otherwise null

cooking_method rules:
- raw = uncooked food ONLY (salads, crudites). Never use for soups, curries, or cooked dishes
- Soups = simmered
- Boiled eggs, tapioca = boiled
- Biryani = dum or slow-cooked
- Fried rice, noodles, stir-fry = wok-tossed (NOT fried)
- Combo dishes or unclear = mixed
- Never return null

health_tags rules:
- The word "medium" is NEVER a valid health_tag under ANY circumstances. Do not use it.
- NEVER assign both "light" and "heavy" to the same dish — pick the dominant one
- light = low oil, mild, easy on stomach (soups, idli, grilled items, steamed)
- heavy = rich, oily, creamy, very filling (butter chicken, biryani, fried items)
- Tandoori and grilled non-veg = high-protein + light (NOT heavy)
- Khichdi = light
- high-protein ONLY for: meat, eggs, dal, paneer, soya, fish, prawns. NOT for plain vegetable dishes

time_affinity rules:
- Valid values are ONLY: breakfast, lunch, snack, dinner, late-night
- NEVER use "dessert", "appetizer", or any other value
- Desserts → snack or dinner only
- Starters → snack or dinner only

occasion_tags rules:
- comfort is NOT a valid occasion_tag — do not use it
- Only assign when dish is STRONGLY associated with that occasion
- Most dishes should have empty occasion_tags []

dietary_tags rules:
- jain = no onion, no garlic, no root vegetables
- vrat-friendly = no onion, no garlic, no regular grains
- no-onion-garlic = contains neither onion nor garlic
- Only assign if confident — default to empty []"""


def tag_dish(dish, restaurant_name, cuisine_list):
    dish_info = (
        f"Restaurant: {restaurant_name}\n"
        f"Restaurant cuisines: {', '.join(cuisine_list)}\n"
        f"Dish: {dish['name']}\n"
        f"Category: {dish['category']}\n"
        f"Price: ₹{dish['price']}\n"
        f"Description: {dish.get('description', 'N/A')}\n"
        f"Existing tags: {', '.join(dish.get('tags', []))}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": TAGGING_PROMPT},
            {"role": "user", "content": dish_info}
        ],
        max_tokens=300,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def run_batch(start_idx, end_idx):
    with open(INPUT_PATH) as f:
        data = json.load(f)

    try:
        with open(OUTPUT_PATH) as f:
            enriched_data = json.load(f)
        print(f"Resuming from existing enriched file.")
    except FileNotFoundError:
        enriched_data = json.loads(json.dumps(data))

    all_dishes = []
    for r_idx, restaurant in enumerate(data):
        for d_idx, dish in enumerate(restaurant["dishes"]):
            all_dishes.append((r_idx, d_idx, dish, restaurant["name"], restaurant["cuisine"]))

    total = len(all_dishes)
    end_idx = min(end_idx, total)

    print(f"\nTagging dishes {start_idx+1} to {end_idx} of {total}...")

    batch = all_dishes[start_idx:end_idx]
    errors = []

    for i, (r_idx, d_idx, dish, r_name, r_cuisine) in enumerate(batch):
        global_idx = start_idx + i
        print(f"  [{global_idx+1}/{total}] {dish['name']} @ {r_name}")

        try:
            new_tags = tag_dish(dish, r_name, r_cuisine)
            enriched_data[r_idx]["dishes"][d_idx].update(new_tags)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append(f"[{global_idx+1}] {dish['name']}: {e}")

        if (i + 1) % 10 == 0:
            time.sleep(1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"\nBatch complete. Saved to {OUTPUT_PATH}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print("No errors.")

    sample_dish = enriched_data[all_dishes[start_idx][0]]["dishes"][all_dishes[start_idx][1]]
    print(f"\nSample: {sample_dish['name']}")
    for field in ["cuisine_type", "cuisine_region", "cooking_method", "health_tags",
                  "calorie_band", "serving_format", "holds_well", "portability",
                  "time_affinity", "occasion_tags", "dietary_tags"]:
        print(f"  {field}: {sample_dish.get(field, 'MISSING')}")

    return end_idx, total


if __name__ == "__main__":
    import sys
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else BATCH_SIZE

    end_idx, total = run_batch(start, end)

    if end_idx < total:
        print(f"\nNext batch command:")
        print(f"  python enrich_metadata.py {end_idx} {min(end_idx + BATCH_SIZE, total)}")
    else:
        print(f"\nAll {total} dishes tagged. Run chunker → embedder → upsert_pinecone to deploy.")
