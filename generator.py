import os
import json
from openai import OpenAI
from retrieval import retrieve

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EXPANSION_MODEL = "gpt-4o-mini"

PROMPT_VERSION = "v1.2"
PROMPT_NOTES = "v1.1. Up to 5 recommendations with reason. Variety enforced — duplicate dishes only when alternatives not available. Ma jaisi tone."

# GPT-4o-mini pricing per 1M tokens
COST_PER_1M_INPUT = 0.15
COST_PER_1M_OUTPUT = 0.60
COST_PER_1M_EMBED = 0.02

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a warm, intuitive food recommendation assistant for Delhi restaurants.
Your job is to recommend up to 5 dishes from the options provided, like a mother who knows exactly
what you want before you do.

Rules:
- You MUST recommend 5 dishes. Only drop below 5 if the same dish appears more than twice with no alternatives.
- Prioritise variety — different dishes, different restaurants where possible.
- Never recommend the same dish twice even if it appears at multiple restaurants, unless no alternatives exist.
- For each dish give: name, restaurant, price, and one sentence explaining WHY this fits the craving.
- Tone: warm, direct, confident. Not salesy. Not listy.
- Never recommend a dish outside the provided options.
- Always mention price."""


# --- COMPONENT 8: Cost Tracker ---
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
            self.expansion_input_tokens +
            self.expansion_output_tokens +
            self.embed_tokens +
            self.generation_input_tokens +
            self.generation_output_tokens
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


# --- COMPONENT 6: Hallucination Guard ---
def extract_dish_names(response_text):
    """Extract dish names from response using LLM."""
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
    """
    Check recommended dishes exist in retrieval hits.
    Returns: (response_text, hallucination_flagged, flagged_dishes)
    """
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
    """Convert retrieval hits into natural recommendation response."""
    options_text = ""
    for i, h in enumerate(hits, 1):
        options_text += (
            f"{i}. {h['dish']} @ {h['restaurant']} | ₹{h['price']}\n"
            f"   Tags: {', '.join(h['tags'])}\n"
        )

    user_prompt = (
        f"User craving: \"{raw_query}\"\n\n"
        f"Available options (already filtered for relevance):\n{options_text}\n"
        f"Recommend exactly 5 dishes from these options. "
        f"Prioritise variety across restaurants and dish types."
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


def recommend(raw_query, conversation_history=None):
    """Full pipeline: retrieve → generate → hallucination check → cost track."""
    tracker = CostTracker()

    hits = retrieve(raw_query, top_k=10, tracker=tracker, conversation_history=conversation_history)

    if not hits:
        return {
            "response": "Sorry, I couldn't find anything matching that craving in our restaurants.",
            "hallucination_flagged": False,
            "flagged_dishes": [],
            "cost": tracker.summary()
        }

    response_text = generate_response(raw_query, hits, tracker)
    response_text, hallucination_flagged, flagged_dishes = hallucination_guard(response_text, hits)

    return {
        "response": response_text,
        "hallucination_flagged": hallucination_flagged,
        "flagged_dishes": flagged_dishes,
        "cost": tracker.summary()
    }


if __name__ == "__main__":
    test_queries = [
        "something creamy and mild under 300",
        "spicy street food snack",
        "light South Indian breakfast",
        "something sour and filling",
        "dessert under 200"
    ]

    print(f"=== Generator {PROMPT_VERSION} ===\n")
    total_cost = 0

    for query in test_queries:
        print(f"Query: '{query}'")
        print("-" * 40)
        result = recommend(query)
        print(result["response"])
        if result["hallucination_flagged"]:
            print(f"FLAGGED: {result['flagged_dishes']}")
        cost = result["cost"]
        total_cost += cost["estimated_cost_usd"]
        print(f"\nTokens: {cost['total_tokens']} | Cost: ${cost['estimated_cost_usd']:.6f}")
        print("=" * 60 + "\n")

    print(f"Total cost for {len(test_queries)} queries: ${total_cost:.6f}")
