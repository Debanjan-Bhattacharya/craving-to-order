import os
import sys
import json
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
    """Extract full restaurant JSON from raw menu text."""

    user_message = f"Restaurant ID: {restaurant_id}\n\n{menu_text}"

    print(f"Sending to GPT-4o-mini... (this may take 30-60 seconds for large menus)")

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

    # Parse and validate
    data = json.loads(raw)

    # Basic validation
    dish_count = len(data.get("dishes", []))
    print(f"Extracted {dish_count} dishes.")
    print(f"Restaurant: {data.get('name')}")
    print(f"Cuisine: {data.get('cuisine')}")
    print(f"Avg cost for two: ₹{data.get('avg_cost_for_two')}")

    # Token usage
    usage = response.usage
    input_cost = (usage.prompt_tokens / 1_000_000) * 0.15
    output_cost = (usage.completion_tokens / 1_000_000) * 0.60
    total_cost = input_cost + output_cost
    print(f"\nTokens: {usage.prompt_tokens} input + {usage.completion_tokens} output")
    print(f"Cost: ${total_cost:.4f}")

    return data


def save_output(data, restaurant_id, output_dir="."):
    filename = f"{restaurant_id}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {filepath}")
    return filepath


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_restaurant.py <restaurant_id> <menu_file.txt>")
        print("Example: python extract_restaurant.py R18 menu_r18.txt")
        sys.exit(1)

    restaurant_id = sys.argv[1].upper()
    menu_file = sys.argv[2]

    if not os.path.exists(menu_file):
        print(f"Error: Menu file '{menu_file}' not found.")
        sys.exit(1)

    with open(menu_file, encoding="utf-8") as f:
        menu_text = f.read()

    print(f"\nExtracting {restaurant_id} from {menu_file}...")
    print(f"Menu text length: {len(menu_text)} characters\n")

    data = extract_restaurant(menu_text, restaurant_id)
    save_output(data, restaurant_id)


if __name__ == "__main__":
    main()
