import os
import json
from openai import OpenAI
from retrieval import retrieve, get_similar_dishes_for_all
from reranker import rerank

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"
PROMPT_VERSION = "v1.2"

COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60
COST_PER_1M_EMBED = 0.02

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a warm, intuitive food recommendation assistant for Delhi restaurants.
Your job is to recommend up to 5 dishes from the options provided, like a mother who knows exactly
what you want before you do.

Rules:
- Recommend exactly as many unique dishes as are available in the options — never repeat a dish to pad to 5.
- Prioritise variety — different dishes, different restaurants where possible.
- Never recommend the same dish twice even if it appears at multiple restaurants, unless no alternatives exist.
- For each dish give: name, restaurant, price, and one sentence explaining WHY this fits the craving.
- Tone: warm, direct, confident. Not salesy. Not listy.
- Never recommend a dish outside the provided options.
- Always mention price."""


class CostTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.expansion_input_tokens = 0
        self.expansion_output_tokens = 0
        self.embed_tokens = 0
        self.generation_input_tokens = 0
        self.generation_output_tokens = 0

    def add_expansion(self, usage):
        self.expansion_input_tokens += usage.prompt_tokens
        self.expansion_output_tokens += usage.completion_tokens

    def add_embed(self, token_count):
        self.embed_tokens += token_count

    def add_generation(self, usage):
        self.generation_input_tokens += usage.prompt_tokens
        self.generation_output_tokens += usage.completion_tokens

    def total_cost(self):
        expansion_cost = (
            (self.expansion_input_tokens / 1_000_000) * COST_PER_1M_INPUT +
            (self.expansion_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        )
        embed_cost = (self.embed_tokens / 1_000_000) * COST_PER_1M_EMBED
        generation_cost = (
            (self.generation_input_tokens / 1_000_000) * COST_PER_1M_INPUT +
            (self.generation_output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        )
        return expansion_cost + embed_cost + generation_cost

    def total_tokens(self):
        return (
            self.expansion_input_tokens + self.expansion_output_tokens +
            self.embed_tokens +
            self.generation_input_tokens + self.generation_output_tokens
        )

    def summary(self):
        return {
            "total_tokens": self.total_tokens(),
            "estimated_cost_usd": round(self.total_cost(), 6),
            "breakdown": {
                "expansion_tokens": self.expansion_input_tokens + self.expansion_output_tokens,
                "embed_tokens": self.embed_tokens,
                "generation_tokens": self.generation_input_tokens + self.generation_output_tokens
            }
        }


def extract_dish_names(response_text):
    prompt = (
        "Extract only the dish names from this food recommendation response. "
        "Return a JSON array of strings only. No explanation, no markdown.\n\n"
        f"{response_text}"
    )
    result = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    raw = result.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return []


def hallucination_guard(response_text, hits):
    hit_names = [h["dish"].lower() for h in hits]
    recommended = extract_dish_names(response_text)
    flagged = []
    for dish in recommended:
        dish_lower = dish.lower()
        if not any(dish_lower in hit or hit in dish_lower for hit in hit_names):
            flagged.append(dish)
    if flagged:
        note = (
            f"\n\n⚠️ Note: {len(flagged)} recommendation(s) flagged for review "
            f"— dish not found in verified menu data: {', '.join(flagged)}"
        )
        return response_text + note, True, flagged
    return response_text, False, []


def generate_response(raw_query, hits, tracker):
    options_text = ""
    for i, h in enumerate(hits, 1):
        taste    = h.get("taste_tags", "")
        texture  = h.get("texture_tags", "")
        cuisine  = h.get("cuisine_type", "")
        category = h.get("category_type", "")
        health   = h.get("health_tags", "")
        options_text += (
            f"{i}. {h['dish']} @ {h['restaurant']} | ₹{h['price']}\n"
            f"   Cuisine: {cuisine} | Category: {category}\n"
            f"   Taste: {taste} | Texture: {texture}\n"
            f"   Health: {health}\n"
        )

    user_prompt = (
        f"User craving: \"{raw_query}\"\n\n"
        f"Available options (already filtered and ranked for relevance):\n{options_text}\n"
        f"Recommend up to 5 dishes from these options.\n"
        f"Rules:\n"
        f"- Prioritise variety — different cuisines, categories, restaurants\n"
        f"- Never recommend the same dish twice\n"
        f"- For each dish give: name, restaurant, price, one sentence explaining why it fits the craving\n"
        f"- If fewer than 5 options are available, recommend only what's listed\n"
        f"- Tone: warm, direct, confident. Not salesy."
    )

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    tracker.add_generation(response.usage)
    return response.choices[0].message.content.strip()


def recommend(raw_query):
    tracker = CostTracker()
    hits = retrieve(raw_query, top_k=10, tracker=tracker)
    hits = rerank(hits, raw_query, top_n=5, debug=True)

    if not hits:
        return {
            "response": "Sorry, I couldn't find anything matching that craving in our restaurants.",
            "hits": [],
            "hallucination_flagged": False,
            "flagged_dishes": [],
            "cost": tracker.summary()
        }

    hits = get_similar_dishes_for_all(hits, top_n=3)

    response_text = generate_response(raw_query, hits, tracker)
    response_text, hallucination_flagged, flagged_dishes = hallucination_guard(response_text, hits)

    return {
        "response": response_text,
        "hits": hits,
        "hallucination_flagged": hallucination_flagged,
        "flagged_dishes": flagged_dishes,
        "cost": tracker.summary()
    }
