import os
import json
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_HOST = "https://craving-to-order-9udw9bu.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL = "text-embedding-3-small"
EXPANSION_MODEL = "gpt-4o-mini"
TOP_K = 5

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

expansion_cache = {}

# --- STEP 5: Hindi intent mapping ---
HINDI_MAP = {
    "pyaz": "onion", "lehsun": "garlic", "adrak": "ginger",
    "vrat": "fasting", "khatta": "tangy sour", "meetha": "sweet",
    "teekha": "spicy", "halka": "light", "bhari": "filling",
    "maida": "refined flour", "dahi": "yogurt", "paneer": "cottage cheese",
    "no pyaz": "no onion", "no lehsun": "no garlic",
    "bina pyaz": "without onion", "bina lehsun": "without garlic",
    "nashta": "breakfast snack", "khana": "meal food",
    "mitha": "sweet dessert", "namkeen": "salty savory snack"
}

def translate_hindi(raw_query):
    """Translate Hindi food terms to English before parsing."""
    q = raw_query.lower()
    for hindi, english in HINDI_MAP.items():
        q = q.replace(hindi, english)
    return q


# --- STEP 2: Fixed expansion prompt — cuisine-aware, no Indian bias ---
def expand_query(raw_query, tracker=None, conversation_history=None):
    """Use LLM to rewrite user query into dish/cuisine concepts.
    If conversation_history provided, resolves relative references like
    'something spicier', 'veg only', 'different restaurant'.
    """
    # Only use cache for queries without conversation history
    cache_key = raw_query if not conversation_history else None
    if cache_key and cache_key in expansion_cache:
        print(f"  [cache hit] Reusing expansion for: '{raw_query}'")
        return expansion_cache[cache_key]

   # Build context string from conversation history
    context_str = ""
    if conversation_history and len(conversation_history) > 0:
        context_str = "\n\nConversation history (most recent last):\n"
        for turn in conversation_history[-3:]:  # last 3 turns only
            context_str += f"User asked: {turn['query']}\n"
            context_str += f"Recommended: {', '.join(turn['dishes'][:3])}\n"
        context_str += "\nThe new query may refer to the above context."
        context_str += "\nTry to stay in a similar cuisine and category unless the user explicitly asks to change it."
        # Add cuisine context from last turn if available
        last_turn = conversation_history[-1]
        if last_turn.get("cuisines"):
            context_str += f"\nPrevious recommendations were {', '.join(last_turn['cuisines'])} cuisine. Prefer the same unless asked to change."

    prompt = (
        "You are a food search assistant for Delhi restaurants. "
        "The menu includes North Indian, Mughlai, South Indian, Kerala, "
        "Indo-Chinese, Lebanese, Continental, and fast food dishes.\n\n"
        "Convert the user's craving into specific dish names, ingredients, "
        "and textures that match their intent — preserving the cuisine type "
        "they asked for. If they ask for western or continental food, use "
        "western dish names. If they ask for Indian food, use Indian terminology. "
        "If no cuisine is specified, use the most relevant terminology."
        f"{context_str}\n\n"
        f"User craving: \"{raw_query}\"\n\n"
        "Respond with a single search string of 15-25 words. No explanation."
    )

    response = client.chat.completions.create(
        model=EXPANSION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0
    )
    if tracker:
        tracker.add_expansion(response.usage)

    expanded = response.choices[0].message.content.strip()
    if cache_key:
        expansion_cache[cache_key] = expanded
    return expanded


def embed_query(text, tracker=None):
    """Embed a single query string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    if tracker:
        tracker.add_embed(response.usage.total_tokens)
    return response.data[0].embedding


def parse_budget(raw_query):
    """Extract budget constraint. Returns int or None."""
    import re
    patterns = [
        r'under\s*(?:rs\.?|₹)?\s*(\d+)',
        r'below\s*(?:rs\.?|₹)?\s*(\d+)',
        r'less\s*than\s*(?:rs\.?|₹)?\s*(\d+)',
        r'(?:rs\.?|₹)\s*(\d+)\s*(?:only|max|maximum)?',
        r'within\s*(?:rs\.?|₹)?\s*(\d+)',
        r'budget.*?(\d+)',
    ]
    raw_lower = raw_query.lower()
    for pattern in patterns:
        match = re.search(pattern, raw_lower)
        if match:
            return int(match.group(1))
    return None


def parse_category(raw_query):
    """Detect category_type from query."""
    q = raw_query.lower()
    if any(w in q for w in ["dessert", "sweet", "cake", "ice cream", "cheesecake", "meetha", "mitha"]):
        return "dessert"
    if any(w in q for w in ["pizza"]):
        return "pizza"
    if any(w in q for w in ["burger"]):
        return "burger"
    if any(w in q for w in ["pasta", "noodles"]):
        return "pasta"
    if any(w in q for w in ["soup"]):
        return "soup"
    if any(w in q for w in ["biryani"]):
        return "biryani"
    if any(w in q for w in ["breakfast", "morning", "nashta"]):
        return "breakfast"
    if any(w in q for w in ["salad"]):
        return "salad"
    return None


def parse_taste(raw_query):
    """Detect taste descriptors."""
    q = raw_query.lower()
    if any(w in q for w in ["sour", "tangy", "achari", "khatta"]):
        return "tangy"
    return None


def parse_ingredient(raw_query):
    """Detect primary ingredient signals that map to existing tags."""
    q = raw_query.lower()
    if any(w in q for w in ["seafood", "fish", "prawn", "prawns", "crab", "lobster", "machli", "jhinga"]):
        return "seafood"
    if any(w in q for w in ["paneer", "cottage cheese"]):
        return "paneer"
    if any(w in q for w in ["mushroom", "mushrooms"]):
        return "mushroom"
    if any(w in q for w in ["egg", "eggs", "anda"]):
        return "egg"
    return None


# --- STEP 3: Cuisine filter ---
def parse_cuisine(raw_query):
    """Detect cuisine include/exclude signals."""
    q = raw_query.lower()

    # Exclusion signals
    if any(w in q for w in ["not indian", "non indian", "western only", "no indian"]):
        return {"exclude": ["North Indian", "South Indian", "Mughlai"]}

    # Inclusion signals
    if any(w in q for w in ["european", "continental", "western"]):
        return {"include": ["Continental", "Italian", "Western", "American"]}
    if any(w in q for w in ["chinese", "indo chinese", "indo-chinese"]):
        return {"include": ["Chinese"]}
    if any(w in q for w in ["south indian"]):
        return {"include": ["South Indian", "Tamil", "Kerala"]}
    if any(w in q for w in ["lebanese", "middle eastern", "mediterranean"]):
        return {"include": ["Lebanese", "Middle Eastern", "Mediterranean"]}
    if any(w in q for w in ["mughlai", "awadhi"]):
        return {"include": ["Mughlai"]}
    if any(w in q for w in ["punjabi"]):
        return {"include": ["Punjabi", "North Indian"]}
    if any(w in q for w in ["kerala"]):
        return {"include": ["Kerala"]}
    if any(w in q for w in ["italian"]):
        return {"include": ["Italian"]}
    if any(w in q for w in ["fast food", "junk food"]):
        return {"include": ["Fast Food", "Burgers", "Pizza"]}

    return None


# --- STEP 4: Diet filter ---
def parse_diet(raw_query):
    """Detect veg/non-veg preference."""
    q = raw_query.lower()
    if any(w in q for w in ["veg only", "vegetarian", "no meat", "veg ", "only veg"]):
        return "veg"
    if any(w in q for w in ["non veg", "nonveg", "noveg", "chicken", "mutton", "fish", "meat", "egg"]):
        return "non-veg"
    return None


def retrieve(raw_query, top_k=TOP_K, tracker=None, conversation_history=None):
    """Full retrieval pipeline: translate → expand → embed → filter → search."""

    # Step 0: Translate Hindi terms
    translated_query = translate_hindi(raw_query)
    if translated_query != raw_query.lower():
        print(f"Translated: '{translated_query}'")

    print(f"\nQuery: '{raw_query}'")

    # Step 1: Expand
    expanded = expand_query(translated_query, tracker=tracker, conversation_history=conversation_history)
    print(f"Expanded: '{expanded[:80]}'")

    # Step 2: Embed
    query_vector = embed_query(expanded, tracker=tracker)

    # Step 3: Parse all filters
    budget = parse_budget(raw_query)
    category = parse_category(raw_query)
    taste = parse_taste(raw_query)
    cuisine = parse_cuisine(raw_query)
    diet = parse_diet(raw_query)
    ingredient = parse_ingredient(raw_query)

    if budget: print(f"Budget filter: ₹{budget}")
    if category: print(f"Category filter: {category}")
    if taste: print(f"Taste filter: {taste}")
    if cuisine: print(f"Cuisine filter: {cuisine}")
    if diet: print(f"Diet filter: {diet}")
    if ingredient: print(f"Ingredient filter: {ingredient}")

    # Step 4: Build conditions — FIXED to handle any number
    conditions = []

    if budget:
        conditions.append({"price": {"$lte": budget}})

    if category:
        conditions.append({"category_type": {"$eq": category}})

    if taste:
        conditions.append({"tags": {"$in": [taste]}})

    if diet:
        conditions.append({"tags": {"$in": [diet]}})

    if ingredient:
        conditions.append({"tags": {"$in": [ingredient]}})

    if cuisine:
        if "include" in cuisine:
            conditions.append({"cuisine": {"$in": cuisine["include"]}})
        elif "exclude" in cuisine:
            # Pinecone doesn't support $nin directly — use $and with $ne per value
            for excluded in cuisine["exclude"]:
                conditions.append({"cuisine": {"$ne": excluded}})

    # --- STEP 1 FIX: conditions block handles any number ---
    if len(conditions) == 0:
        pinecone_filter = {}
    elif len(conditions) == 1:
        pinecone_filter = conditions[0]
    else:
        pinecone_filter = {"$and": conditions}

    # Step 5: Query Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter=pinecone_filter if pinecone_filter else None,
        include_metadata=True
    )

    hits = []
    for match in results.matches:
        hits.append({
            "score": round(match.score, 4),
            "dish": match.metadata["dish_name"],
            "restaurant": match.metadata["restaurant_name"],
            "price": match.metadata["price"],
            "tags": match.metadata["tags"],
            "cuisine": match.metadata["cuisine"]
        })

    return hits


if __name__ == "__main__":
    test_queries = [
        "something creamy western not indian",
        "veg only spicy snack",
        "no pyaz lehsun dish",
        "something continental under 500",
        "non veg biryani",
        "Lebanese food",
        "south indian breakfast",
        "something sour and filling",
    ]

    for query in test_queries:
        hits = retrieve(query)
        print(f"\nTop {len(hits)} results:")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h['dish']} @ {h['restaurant']} | ₹{h['price']} | score: {h['score']}")
        print("-" * 60)
