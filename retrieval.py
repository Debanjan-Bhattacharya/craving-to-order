"""
retrieval.py
LLM signal extractor → metadata-first Pinecone query → diversity cap → reranker candidates
"""

import os
import json
import numpy as np
from collections import Counter
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_HOST       = "https://craving-to-order-9udw9bu.svc.aped-4627-b74a.pinecone.io"
EMBEDDING_MODEL  = "text-embedding-3-small"
SIGNAL_MODEL     = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(host=INDEX_HOST)

# ── Load dish name embeddings ─────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(_BASE, "dish_name_embeddings.json"), encoding="utf-8") as _f:
        DISH_NAME_EMBEDDINGS = json.load(_f)
    print(f"[retrieval] Loaded {len(DISH_NAME_EMBEDDINGS)} dish name embeddings")
except FileNotFoundError:
    DISH_NAME_EMBEDDINGS = {}
    print("[retrieval] WARNING: dish_name_embeddings.json not found")

# ── Load restaurant names ─────────────────────────────────────────────────────
try:
    with open(os.path.join(_BASE, "dish_lookup.json"), encoding="utf-8") as _f:
        _dl = json.load(_f)
    RESTAURANT_NAMES = list({
        v["all_restaurants"][0]
        for v in _dl.values()
        if v.get("all_restaurants")
    })[:100]
    print(f"[retrieval] Loaded {len(RESTAURANT_NAMES)} restaurant names")
except Exception:
    RESTAURANT_NAMES = []

# ── Signal extractor prompt ───────────────────────────────────────────────────

SIGNAL_EXTRACTOR_PROMPT = """You are a food query signal extractor for a Delhi restaurant recommendation system.

Extract structured signals from the user's food query. Return ONLY a JSON object with these fields:

{
  "cuisine_type": ["list of cuisines if specified, else empty — e.g. thai, north-indian, japanese, korean, italian, mexican, bengali, south-indian, chinese, lebanese, continental, mughlai, street-food"],
  "cuisine_region": ["specific region if mentioned — e.g. bengali, awadhi, chettinad, punjabi, kerala, hyderabadi, tibetan"],
  "category": ["dish category if specified — e.g. biryani, pizza, burger, dessert, soup, snack, breakfast, momo, noodles, roll, wrap, thali"],
  "diet": "veg or non-veg or jain or vrat or null — only if explicitly stated or strongly implied",
  "texture": ["texture signals — e.g. creamy, crispy, crunchy, soft, juicy, rich, light, velvety"],
  "taste": ["taste signals — e.g. spicy, mild, tangy, sweet, smoky, buttery, sour, umami"],
  "occasion": ["occasion signals — e.g. party, date-night, office-lunch, quick-bite, post-workout, hangover-food"],
  "time": ["time signals — e.g. breakfast, lunch, dinner, late-night, snack"],
  "festival": ["festival signals — e.g. eid, diwali, holi, navratri, durga-puja, christmas, iftar, sehri"],
  "health": ["health signals — e.g. low-cal, high-protein, light, gut-friendly, low-fat, high-fibre"],
  "cooking_method": ["e.g. grilled, fried, steamed, tandoor, baked — only if mentioned"],
  "serving_format": ["e.g. bowl, roll, finger-food, platter, handheld — only if mentioned"],
  "holds_well": true or false or null,
  "portability": true or false or null,
  "protein_band": "high or low or null — only for explicit fitness/protein queries",
  "calorie_band": "low or high or null — only for explicit diet queries",
  "budget": integer max price in INR or null,
  "restaurant_include": ["restaurant names to include — only if explicitly named in query"],
  "restaurant_exclude": ["restaurant names to exclude — normalize to full name e.g. Domino's Pizza not dominos"],
  "is_vague": true or false,
  "embedding_query": "short 5-10 word neutral description for vector search — focus on food concepts not dish names"
}

Rules:
- Only populate fields clearly present in the query — leave others null or empty
- is_vague = true when query has no parseable food signals
- For Hindi/Urdu: teekha=spicy, halka=light, meetha=sweet, khatta=tangy, nashta=breakfast
- restaurant_include: ONLY if user explicitly types the restaurant name. NEVER infer from context.
- For restaurant names, use full official names: Domino's Pizza, Pizza Hut, McDonald's, Burger King, KFC
- Return ONLY the JSON object. No explanation. No markdown."""

VALID_CATEGORIES = {
    "biryani", "pizza", "burger", "dessert", "soup", "salad",
    "breakfast", "snack", "starter", "main-course", "bread",
    "rice-dish", "momo", "noodles", "wrap", "roll", "thali",
    "bowl", "pasta", "sandwich", "beverage", "street-food",
    "combo-meal", "side-dish"
}


def extract_signals(raw_query: str, tracker=None) -> dict:
    restaurant_context = (
        f"\n\nKnown restaurants: {', '.join(RESTAURANT_NAMES[:80])}"
        if RESTAURANT_NAMES else ""
    )
    user_content = f"Query: \"{raw_query}\"{restaurant_context}"

    try:
        response = client.chat.completions.create(
            model=SIGNAL_MODEL,
            messages=[
                {"role": "system", "content": SIGNAL_EXTRACTOR_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=400,
            temperature=0,
        )
        if tracker:
            tracker.add_expansion(response.usage)

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        signals = json.loads(raw)
        signals["_raw_query"] = raw_query
        return signals

    except Exception as e:
        print(f"  [signal extractor error] {e}")
        return {
            "is_vague": True,
            "embedding_query": raw_query,
            "_raw_query": raw_query,
            "cuisine_type": [], "cuisine_region": [], "category": [],
            "diet": None, "texture": [], "taste": [], "occasion": [],
            "time": [], "festival": [], "health": [], "cooking_method": [],
            "serving_format": [], "holds_well": None, "portability": None,
            "protein_band": None, "calorie_band": None, "budget": None,
            "restaurant_include": [], "restaurant_exclude": [],
        }


def embed_query(text: str, tracker=None) -> list:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    if tracker:
        tracker.add_embed(response.usage.total_tokens)
    return response.data[0].embedding


def build_pinecone_filter(signals: dict) -> dict:
    conditions = []

    diet = signals.get("diet")
    if diet == "veg":
        conditions.append({"is_veg": {"$eq": "veg"}})
    elif diet == "non-veg":
        conditions.append({"is_veg": {"$eq": "non-veg"}})
    elif diet in ("jain", "vrat"):
        conditions.append({"dietary_tags": {"$in": [diet]}})

    budget = signals.get("budget")
    if budget:
        conditions.append({"price": {"$lte": budget}})

    # Restaurant include — only if explicitly in raw query text
    restaurant_include = signals.get("restaurant_include") or []
    if restaurant_include:
        query_lower = signals.get("_raw_query", "").lower()
        explicit = [r for r in restaurant_include if r.lower() in query_lower]
        if explicit:
            conditions.append({"restaurant_name": {"$in": explicit}})

    for r in (signals.get("restaurant_exclude") or []):
        conditions.append({"restaurant_name": {"$ne": r}})

    cuisine_types = signals.get("cuisine_type") or []
    if cuisine_types:
        conditions.append({"cuisine_type": {"$in": cuisine_types}})

    cuisine_regions = signals.get("cuisine_region") or []
    if cuisine_regions:
        conditions.append({"cuisine_region": {"$in": cuisine_regions}})

    categories = [c for c in (signals.get("category") or []) if c in VALID_CATEGORIES]
    if categories:
        conditions.append({"category_type": {"$in": categories}})

    if len(conditions) == 0:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def apply_diversity_cap(
    hits: list,
    signals: dict,
    preferred_per_restaurant: int = 1,
    preferred_per_cuisine: int = 1,
    hard_cap_restaurant: int = 2,
    hard_cap_cuisine: int = 3,
    target: int = 15,
) -> list:
    specified_cuisines    = set(signals.get("cuisine_type") or [])
    specified_restaurants = set(signals.get("restaurant_include") or [])

    restaurant_count = {}
    cuisine_count    = {}
    selected         = []

    def get_cuisine(hit):
        return (hit.get("cuisine_type") or "unknown").lower()

    def caps_ok(hit, use_preferred=True):
        r = hit.get("restaurant", "")
        c = get_cuisine(hit)
        cap_r = 999 if r in specified_restaurants else (preferred_per_restaurant if use_preferred else hard_cap_restaurant)
        cap_c = 999 if c in specified_cuisines    else (preferred_per_cuisine    if use_preferred else hard_cap_cuisine)
        return restaurant_count.get(r, 0) < cap_r and cuisine_count.get(c, 0) < cap_c

    def add_hit(hit):
        r = hit.get("restaurant", "")
        c = get_cuisine(hit)
        restaurant_count[r] = restaurant_count.get(r, 0) + 1
        cuisine_count[c]    = cuisine_count.get(c, 0) + 1
        selected.append(hit)

    for hit in hits:
        if len(selected) >= target:
            break
        if caps_ok(hit, use_preferred=True):
            add_hit(hit)

    for hit in hits:
        if len(selected) >= target:
            break
        if hit in selected:
            continue
        if caps_ok(hit, use_preferred=False):
            add_hit(hit)

    return selected


def retrieve(raw_query: str, top_k: int = 20, tracker=None) -> list:
    print(f"\nQuery: '{raw_query}'")

    signals = extract_signals(raw_query, tracker)
    is_vague        = signals.get("is_vague", True)
    embedding_query = signals.get("embedding_query") or raw_query

    active = {k: v for k, v in signals.items()
              if v and k not in ("is_vague", "embedding_query", "_raw_query")}
    print(f"  Signals: {active}")
    print(f"  Vague: {is_vague} | Embedding query: '{embedding_query}'")

    pinecone_filter = build_pinecone_filter(signals)
    fetch_k = 20 if not is_vague else 30

    if pinecone_filter:
        print(f"  Pinecone filter: {pinecone_filter}")

    query_vector = embed_query(embedding_query, tracker)

    diet = signals.get("diet")

    if not diet and not pinecone_filter.get("is_veg"):
        veg_filter    = {**pinecone_filter, "is_veg": {"$eq": "veg"}}    if pinecone_filter else {"is_veg": {"$eq": "veg"}}
        nonveg_filter = {**pinecone_filter, "is_veg": {"$eq": "non-veg"}} if pinecone_filter else {"is_veg": {"$eq": "non-veg"}}

        if pinecone_filter and "$and" in pinecone_filter:
            veg_filter    = {"$and": pinecone_filter["$and"] + [{"is_veg": {"$eq": "veg"}}]}
            nonveg_filter = {"$and": pinecone_filter["$and"] + [{"is_veg": {"$eq": "non-veg"}}]}

        half_k = fetch_k // 2
        veg_r    = index.query(vector=query_vector, top_k=half_k, filter=veg_filter,    include_metadata=True)
        nonveg_r = index.query(vector=query_vector, top_k=half_k, filter=nonveg_filter, include_metadata=True)

        all_matches = veg_r.matches + nonveg_r.matches
        all_matches = sorted(all_matches, key=lambda x: x.score, reverse=True)

        rest_counts = {}
        capped = []
        for m in all_matches:
            r = m.metadata.get("restaurant_name", "")
            if rest_counts.get(r, 0) < 2:
                capped.append(m)
                rest_counts[r] = rest_counts.get(r, 0) + 1
        all_matches = capped
    else:
        res         = index.query(vector=query_vector, top_k=fetch_k,
                                  filter=pinecone_filter if pinecone_filter else None,
                                  include_metadata=True)
        all_matches = res.matches

    def match_to_hit(match):
        m = match.metadata
        return {
            "score":          round(match.score, 4),
            "dish":           m.get("dish_name", ""),
            "restaurant":     m.get("restaurant_name", ""),
            "price":          m.get("price", 0),
            "is_veg":         m.get("is_veg", ""),
            "cuisine_type":   m.get("cuisine_type", ""),
            "cuisine_region": m.get("cuisine_region", ""),
            "category_type":  m.get("category_type", ""),
            "taste_tags":     m.get("taste_tags", ""),
            "texture_tags":   m.get("texture_tags", ""),
            "dietary_tags":   m.get("dietary_tags", ""),
            "health_tags":    m.get("health_tags", ""),
            "cooking_method": m.get("cooking_method", ""),
            "serving_format": m.get("serving_format", ""),
            "holds_well":     m.get("holds_well", ""),
            "portability":    m.get("portability", ""),
            "time_affinity":  m.get("time_affinity", ""),
            "occasion_tags":  m.get("occasion_tags", ""),
            "festival_tags":  m.get("festival_tags", ""),
            "calorie_band":   m.get("calorie_band", ""),
            "protein_band":   m.get("protein_band", ""),
        }

    hits = [match_to_hit(m) for m in all_matches]
    print(f"  Pinecone returned: {len(hits)} hits")

    # Retry without category filter if zero results
    if len(hits) == 0 and pinecone_filter:
        print("  [retry] Zero hits — retrying without category filter")
        if "$and" in pinecone_filter:
            relaxed = [c for c in pinecone_filter["$and"] if "category_type" not in c]
            rf = {"$and": relaxed} if len(relaxed) > 1 else relaxed[0] if relaxed else {}
        else:
            rf = {}
        retry = index.query(vector=query_vector, top_k=fetch_k,
                            filter=rf if rf else None, include_metadata=True)
        hits = [match_to_hit(m) for m in retry.matches]

    diverse_hits = apply_diversity_cap(hits, signals, target=15)
    print(f"  After diversity cap: {len(diverse_hits)} candidates")

    for h in diverse_hits:
        h["_signals"] = signals

    return diverse_hits


# ── Fuzzy dedup ───────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def is_duplicate_dish(dish_name: str, accepted_dishes: list, threshold: float = 0.82) -> bool:
    name_lower = dish_name.lower().strip()

    if name_lower in {d.lower().strip() for d in accepted_dishes}:
        return True

    for accepted in accepted_dishes:
        a = accepted.lower().strip()
        if a and (a in name_lower or name_lower in a):
            return True

    if not DISH_NAME_EMBEDDINGS:
        return False

    vec_a = DISH_NAME_EMBEDDINGS.get(name_lower)
    if not vec_a:
        return False

    for accepted in accepted_dishes:
        vec_b = DISH_NAME_EMBEDDINGS.get(accepted.lower().strip())
        if vec_b and cosine_similarity(vec_a, vec_b) >= threshold:
            return True

    return False


# ── Similar dishes ────────────────────────────────────────────────────────────

def get_similar_dishes(hit: dict, exclude_dish_names: list, top_n: int = 3) -> list:
    cuisine_type = hit.get("cuisine_type", "")
    taste        = hit.get("taste_tags", "")
    texture      = hit.get("texture_tags", "")
    category     = hit.get("category_type", "")

    search_parts = [p for p in [
        cuisine_type,
        taste.replace("|", " ").strip() if taste else "",
        texture.replace("|", " ").strip() if texture else "",
        category,
    ] if p]
    search_text  = " ".join(search_parts) if search_parts else hit.get("dish", "")
    query_vector = embed_query(search_text)

    conditions = []
    if cuisine_type:
        conditions.append({"cuisine_type": {"$eq": cuisine_type}})
    if category:
        conditions.append({"category_type": {"$eq": category}})

    if len(conditions) == 0:
        pf = {}
    elif len(conditions) == 1:
        pf = conditions[0]
    else:
        pf = {"$and": conditions}

    results = index.query(vector=query_vector, top_k=20,
                          filter=pf if pf else None, include_metadata=True)

    accepted_names = list(exclude_dish_names)
    similar        = []

    for match in results.matches:
        m          = match.metadata
        match_dish = m.get("dish_name", "")

        if is_duplicate_dish(match_dish, accepted_names):
            continue

        accepted_names.append(match_dish)
        similar.append({
            "dish":          match_dish,
            "restaurant":    m.get("restaurant_name", ""),
            "price":         m.get("price", 0),
            "is_veg":        m.get("is_veg", ""),
            "cuisine_type":  m.get("cuisine_type", ""),
            "category_type": m.get("category_type", ""),
            "taste_tags":    m.get("taste_tags", ""),
            "texture_tags":  m.get("texture_tags", ""),
            "score":         round(match.score, 4),
        })

        if len(similar) >= top_n:
            break

    return similar


def get_similar_dishes_for_all(hits: list, top_n: int = 3) -> list:
    """Sequential for cross-card deduplication."""
    exclude_names = [h["dish"] for h in hits]
    globally_used = set(d.lower() for d in exclude_names)
    enriched      = [dict(h) for h in hits]

    for i, hit in enumerate(enriched):
        similar = get_similar_dishes(hit, list(globally_used), top_n)
        enriched[i]["similar_dishes"] = similar
        for s in similar:
            globally_used.add(s["dish"].lower())

    return enriched
