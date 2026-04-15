import os
import sys
import json
import time
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
  "avg_cost_for_two": 000,
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
- Extract EVERY dish listed. Do not skip any dish, do not group dishes, do not summarize sections.
- Each individual dish is a separate entry. "Chicken Tikka" and "Paneer Tikka" are two entries.
- If a dish has size/quantity variants (1pc/2pc, Small/Large), create a separate entry for each variant.
- Process the complete menu before responding. Do not truncate or stop early.

TAGS FIELD RULES:
- Diet: always include veg or non-veg
- Spice: mild, medium-spicy, spicy, very-spicy
- Taste: add tangy if primarily sour/tangy (chaat, achari, rasam, tamarind-based)
- Satiety: add filling if rice/biryani/noodles/pasta/thali/dal. Add light if low oil, easy on stomach.
- Health: add healthy if grilled/steamed/low-oil/vegetable-forward
- Texture: add creamy, rich, crispy, smoky, indulgent where applicable
- Occasion: add budget if under ₹150, premium if over ₹500, bestseller/popular only if marked on menu

CUISINE_TYPE RULES:
- Momos (any) = tibetan
- Manchurian, Chilli Chicken, Fried Rice, Hakka Noodles = indo-chinese
- Andhra, Chettinad = south-indian (NOT indo-chinese)
- Thai curry, Tomyum = thai
- Clear/Sweet Corn/Hot Sour soup = chinese
- Biryani at Kerala restaurant = south-indian
- Fast food chains = fast-food

CUISINE_REGION RULES:
- Rogan Josh = kashmiri. Hyderabadi biryani = hyderabadi. Bisibele Bhaat = karnataka.
- Kerala restaurant biryani = kerala. Otherwise null if not clearly regional.

COOKING_METHOD RULES:
- raw = uncooked only. Soups = simmered. Biryani = dum or slow-cooked.
- Fried rice/noodles = wok-tossed. Combo/unclear = mixed. Never null.

HEALTH_TAGS RULES:
- "medium" is NEVER valid. Never assign both light and heavy.
- Tandoori/grilled non-veg = high-protein + light. Khichdi = light.
- high-protein ONLY for meat, eggs, dal, paneer, soya, fish, prawns.

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
    data = json.loads(raw)

    usage = response.usage
    cost = (usage.prompt_tokens / 1_000_000) * 0.15 + \
           (usage.completion_tokens / 1_000_000) * 0.60

    return data, len(data.get("dishes", [])), cost


def batch_extract(menus_folder="menus", output_folder="extracted"):
    """
    Process all .txt files in menus_folder.
    File naming convention: R18.txt, R19.txt etc.
    Outputs R18.json, R19.json etc. in output_folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find all txt files
    txt_files = sorted([
        f for f in os.listdir(menus_folder)
        if f.endswith(".txt")
    ])

    if not txt_files:
        print(f"No .txt files found in '{menus_folder}' folder.")
        print("Create a 'menus' folder and add your menu files named R18.txt, R19.txt etc.")
        return

    print(f"Found {len(txt_files)} menu files: {', '.join(txt_files)}\n")

    total_cost = 0
    results = []

    for txt_file in txt_files:
        restaurant_id = txt_file.replace(".txt", "").upper()
        input_path = os.path.join(menus_folder, txt_file)
        output_path = os.path.join(output_folder, f"{restaurant_id}.json")

        # Skip if already extracted
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
            # Save raw output for debugging
            error_path = os.path.join(output_folder, f"{restaurant_id}_ERROR.txt")
            results.append({"id": restaurant_id, "status": "ERROR", "error": str(e)})

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({"id": restaurant_id, "status": "ERROR", "error": str(e)})

        # Rate limit buffer between restaurants
        time.sleep(2)

    # Summary
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
