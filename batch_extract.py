import os
import sys
import json
import time
import re
import glob
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EXTRACTION_PROMPT = """You are a structured data extraction assistant. I will paste raw menu data from a restaurant (copy-pasted from Zomato or similar). Extract it into this exact JSON format:

{
  "restaurant_id": "R__",
  "name": "Restaurant Name",
  "location": "Area, Delhi NCR",
  "cuisine": ["Cuisine1", "Cuisine2"],
  "avg_cost_for_two": 400,
  "veg_options": true,
  "dishes": [
    {
      "id": "R__001",
      "name": "Dish Name",
      "category": "Menu Section Name",
      "price": 000,
      "description": "One sentence description that reads like a real Zomato listing. Not a food blog.",
      "tags": ["veg/non-veg", "spice-level", "cuisine-type", "meal-type", "texture", "occasion"],
      "category_type": "one of: dessert/snack/main/breakfast/beverage/complete_meal/soup/pizza/burger/pasta/salad/biryani",
      "cuisine_type": "one of: north-indian/south-indian/indo-chinese/thai/tibetan/chinese/continental/lebanese/mughlai/fast-food/bakery/kerala/middle-eastern",
      "cuisine_region": "one of: punjabi/marathi/bengali/tamil/kerala/awadhi/hyderabadi/chettinad/kashmiri/karnataka/null",
      "cooking_method": "one of: fried/grilled/steamed/boiled/simmered/baked/tandoor/raw/slow-cooked/tawa/wok-tossed/mixed/dum",
      "health_tags": ["from: low-cal/high-protein/gut-friendly/light/heavy"],
      "calorie_band": "one of: low/medium/high",
      "serving_format": "one of: finger-food/handheld/bowl/platter/sit-down/sharing",
      "holds_well": true,
      "portability": true,
      "time_affinity": ["from: breakfast/lunch/snack/dinner/late-night"],
      "occasion_tags": [],
      "dietary_tags": []
    }
  ]
}

EXTRACTION RULES:
- Extract EVERY individual dish listed. Do not skip, group, or summarize.
- Each dish is a separate entry — "Chicken Tikka" and "Paneer Tikka" are two entries.
- Skip combo deals, bundle offers, meal deals, and promotional packs (e.g. "2 Medium Pizza + Coke", "Super Saver Deal", anything with "Meal" + price that bundles multiple items). Only extract individual orderable dishes.
- If a dish has size variants (Small/Medium/Large), create a separate entry for each variant.
- Process the complete menu before responding. Do not truncate or stop early.

TAGS FIELD RULES:
- Diet: always include veg or non-veg
- Spice: mild, medium-spicy, spicy, very-spicy
- Taste: add tangy if primarily sour/tangy (chaat, achari, rasam, tamarind-based)
- Satiety: add filling if rice/biryani/noodles/pasta/thali/dal. Add light if low oil, easy on stomach.
- Health: add healthy if grilled/steamed/low-oil/vegetable-forward
- Texture: add creamy, rich, crispy, smoky, indulgent where applicable
- Occasion: add budget if under ₹150, premium if over ₹500, bestseller/popular only if marked on menu

CATEGORY_TYPE RULES:
- Pizza restaurants: individual pizzas = pizza, sides/fries/garlic bread = snack, desserts = dessert, bundled meal deals = complete_meal
- Never use "meal" as category_type — use complete_meal instead

CUISINE_TYPE RULES:
- Momos (any) = tibetan
- Manchurian, Chilli Chicken, Fried Rice, Hakka Noodles = indo-chinese
- Andhra, Chettinad = south-indian (NOT indo-chinese)
- Thai curry, Tomyum = thai
- Clear/Sweet Corn/Hot Sour soup = chinese
- Biryani at Kerala restaurant = south-indian
- Fast food chains (McDonald's, KFC, Domino's) = fast-food
- Pizza, pasta, garlic bread = continental

CUISINE_REGION RULES:
- Rogan Josh = kashmiri. Hyderabadi biryani = hyderabadi. Bisibele Bhaat = karnataka.
- Kerala restaurant biryani = kerala. Otherwise null if not clearly regional.

COOKING_METHOD RULES:
- raw = uncooked only. Soups = simmered. Biryani = dum or slow-cooked.
- Fried rice/noodles = wok-tossed. Combo/unclear = mixed. Never null.

HEALTH_TAGS RULES:
- "medium" is NEVER valid. Never assign both light and heavy to same dish.
- Always assign at least one health_tag.
- Fried/cheesy/creamy/rich dishes = heavy. Grilled/steamed/salad/soup = light.
- Tandoori/grilled non-veg = high-protein + light. Khichdi = light.
- high-protein ONLY for meat, eggs, dal, paneer, soya, fish, prawns.
- When uncertain: mains default to heavy, soups/salads default to light.

TIME_AFFINITY: ONLY breakfast/lunch/snack/dinner/late-night. Never "dessert" or "appetizer".
OCCASION_TAGS: comfort is NOT valid. Most dishes = [].
DIETARY_TAGS: Only assign jain/vrat-friendly/no-onion-garlic/gluten-free if confident.

DESCRIPTION: One sentence, real Zomato listing style, no invented details.

OUTPUT: Return only valid JSON. No explanation, no markdown fences, no preamble."""


def extract_restaurant(menu_text, restaurant_id):
    user_message = f"Restaurant ID: {restaurant_id}\n\n{menu_text}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_message}
        ],
        max_tokens=16000,
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Fix common LLM JSON issues
    raw = re.sub(r':\s*000\b', ': 0', raw)  # fix invalid 000
    raw = re.sub(r':\s*0+(\d)', r': \1', raw)  # fix leading zeros like 0400
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("  Warning: JSON truncated — attempting repair...")
        open_braces = raw.count('{') - raw.count('}')
        open_brackets = raw.count('[') - raw.count(']')
        last_complete = raw.rfind('},')
        if last_complete > 0:
            raw = raw[:last_complete+1]
            raw += ']' * open_brackets + '}' * open_braces
        data = json.loads(raw)
        print(f"  Repair successful.")

    usage = response.usage
    cost = (usage.prompt_tokens / 1_000_000) * 0.15 + \
           (usage.completion_tokens / 1_000_000) * 0.60

    return data, len(data.get("dishes", [])), cost


def merge_parts(output_folder):
    """Merge R18_PART1.json + R18_PART2.json → R18.json"""
    part_files = (
        glob.glob(os.path.join(output_folder, "*_part*.json")) +
        glob.glob(os.path.join(output_folder, "*_PART*.json"))
    )

    if not part_files:
        return

    # Group by base restaurant ID
    groups = {}
    for f in sorted(part_files):
        base = os.path.splitext(os.path.basename(f))[0]
        rid = re.split("_part|_PART", base)[0]
        if rid not in groups:
            groups[rid] = []
        groups[rid].append(f)

    for rid, parts in groups.items():
        print(f"  Merging {len(parts)} parts for {rid}...")
        merged_dishes = []
        base_data = None

        for part_file in sorted(parts):
            with open(part_file, encoding="utf-8") as f:
                data = json.load(f)
            if base_data is None:
                base_data = data
                merged_dishes = data.get("dishes", [])
            else:
                merged_dishes.extend(data.get("dishes", []))

        if base_data:
            # Deduplicate by dish name — keep first occurrence
            seen_names = set()
            unique_dishes = []
            for dish in merged_dishes:
                name = dish.get("name", "").strip().lower()
                if name and name not in seen_names:
                    seen_names.add(name)
                    unique_dishes.append(dish)
            print(f"  Deduped: {len(merged_dishes)} → {len(unique_dishes)} dishes")

            # Renumber dish IDs
            for i, dish in enumerate(unique_dishes, 1):
                dish["id"] = f"{rid}{i:03d}"

            base_data["dishes"] = unique_dishes
            base_data["restaurant_id"] = rid

            out_path = os.path.join(output_folder, f"{rid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(base_data, f, indent=2, ensure_ascii=False)
            print(f"  → {rid}.json ({len(unique_dishes)} dishes)")


def batch_extract(menus_folder="menus", output_folder="extracted"):
    os.makedirs(output_folder, exist_ok=True)

    all_files = os.listdir(menus_folder)

    # Find which base IDs have parts — skip original if parts exist
    has_parts = set()
    for f in all_files:
        if re.search(r'_part\d+', f, re.IGNORECASE) and f.endswith('.txt'):
            base = re.split(r'_part', f, flags=re.IGNORECASE)[0].upper()
            has_parts.add(base)

    txt_files = sorted([
        f for f in all_files
        if f.endswith('.txt')
        and '_clean' not in f.lower()
        and not (
            not re.search(r'_part\d+', f, re.IGNORECASE)
            and os.path.splitext(f)[0].upper() in has_parts
        )
    ])

    if not txt_files:
        print(f"No .txt files found in '{menus_folder}' folder.")
        return

    print(f"Found {len(txt_files)} menu files: {', '.join(txt_files)}\n")

    total_cost = 0
    results = []

    for txt_file in txt_files:
        restaurant_id = txt_file.replace(".txt", "").upper()
        input_path = os.path.join(menus_folder, txt_file)
        output_path = os.path.join(output_folder, f"{restaurant_id}.json")

        if os.path.exists(output_path):
            print(f"[SKIP] {restaurant_id} — already extracted ({output_path})")
            continue

        print(f"[{restaurant_id}] Extracting from {txt_file}...")

        with open(input_path, encoding="utf-8") as f:
            menu_text = f.read()

        try:
            data, dish_count, cost = extract_restaurant(menu_text, restaurant_id)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            total_cost += cost
            results.append({
                "id": restaurant_id,
                "name": data.get("name", "Unknown"),
                "dishes": dish_count,
                "cost": cost,
                "status": "OK"
            })

            print(f"  ✓ {data.get('name')} | {dish_count} dishes | ${cost:.4f}")

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {e}")
            results.append({"id": restaurant_id, "status": "ERROR", "error": str(e)})

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"id": restaurant_id, "status": "ERROR", "error": str(e)})

        time.sleep(2)

    # Auto-merge parts after extraction
    merge_parts(output_folder)

    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE")
    print(f"{'='*50}")
    successful = [r for r in results if r.get("status") == "OK"]
    failed = [r for r in results if r.get("status") == "ERROR"]
    skipped = len(txt_files) - len(results)

    print(f"Processed: {len(successful)} | Failed: {len(failed)} | Skipped: {skipped}")
    print(f"Total dishes extracted: {sum(r['dishes'] for r in successful)}")
    print(f"Total API cost: ${total_cost:.4f}")
    print(f"\nOutputs saved to: {output_folder}/")

    if failed:
        print(f"\nFailed restaurants:")
        for r in failed:
            print(f"  {r['id']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "menus"
    out = sys.argv[2] if len(sys.argv) > 2 else "extracted"
    batch_extract(folder, out)
