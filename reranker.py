"""
reranker.py — Second-pass scoring layer between Pinecone retrieval and generation.

Architecture:
  Stage 1: Pinecone returns top 10 by cosine similarity (recall-focused)
  Stage 2: Reranker scores each on metadata signal match (precision-focused)
  Stage 3: Returns top 5 by rerank_score to generator

Boost signals used:
  - Spice intensity match
  - Texture match (crispy, creamy, smoky, cheesy, soft)
  - Cooking method match (grilled, tandoor, fried, steamed)
  - Health signal match (low-cal, high-protein, gut-friendly)
  - Time affinity match (breakfast, lunch, snack, dinner, late-night)
  - Serving format match (finger-food, handheld, bowl, platter)
  - Occasion match (party, cheat-day, date-night etc.)
  - Dietary match (vrat-friendly, no-onion-garlic, jain)
  - Cuisine exact match
  - Portability match
"""


# --- Signal detection maps ---

SPICE_SIGNALS = {
    "very spicy": "very-spicy",
    "extra spicy": "very-spicy",
    "very-spicy": "very-spicy",
    "spicy": "spicy",
    "medium spicy": "medium-spicy",
    "mild": "mild",
    "not spicy": "mild",
    "no spice": "mild",
}

TEXTURE_SIGNALS = [
    "crispy", "crunchy", "creamy", "smoky", "cheesy",
    "soft", "gooey", "juicy", "puffy", "rich", "indulgent"
]

COOKING_SIGNALS = {
    "grilled": "grilled",
    "tandoor": "tandoor",
    "tandoori": "tandoor",
    "fried": "fried",
    "steamed": "steamed",
    "baked": "baked",
    "slow cooked": "slow-cooked",
    "dum": "dum",
    "wok": "wok-tossed",
}

HEALTH_SIGNALS = {
    "low cal": "low-cal",
    "low calorie": "low-cal",
    "diet": "low-cal",
    "weight": "low-cal",
    "non fattening": "low-cal",
    "not fattening": "low-cal",
    "light": "light",
    "healthy": "light",
    "gym": "high-protein",
    "protein": "high-protein",
    "high protein": "high-protein",
    "muscle": "high-protein",
    "post workout": "high-protein",
    "stomach": "gut-friendly",
    "acidity": "gut-friendly",
    "sick": "gut-friendly",
    "bloating": "gut-friendly",
}

TIME_SIGNALS = {
    "breakfast": "breakfast",
    "morning": "breakfast",
    "nashta": "breakfast",
    "lunch": "lunch",
    "afternoon": "lunch",
    "dinner": "dinner",
    "evening": "dinner",
    "night": "dinner",
    "late night": "late-night",
    "midnight": "late-night",
    "snack": "snack",
    "munchies": "snack",
}

SERVING_SIGNALS = {
    "finger food": "finger-food",
    "party": "finger-food",
    "guests": "finger-food",
    "drinks": "finger-food",
    "handheld": "handheld",
    "on the go": "handheld",
    "travel": "handheld",
    "bowl": "bowl",
    "platter": "platter",
    "sharing": "sharing",
}

OCCASION_SIGNALS = {
    "party": "party",
    "cheat day": "cheat-day",
    "cheat": "cheat-day",
    "date": "date-night",
    "romantic": "date-night",
    "holi": "holi",
    "eid": "eid",
    "iftar": "iftar",
    "sehri": "sehri",
    "vrat": "vrat",
    "fasting": "vrat",
    "diwali": "diwali",
}

DIETARY_SIGNALS = {
    "jain": "jain",
    "vrat": "vrat-friendly",
    "fasting": "vrat-friendly",
    "no onion": "no-onion-garlic",
    "no garlic": "no-onion-garlic",
    "no pyaz": "no-onion-garlic",
    "no lehsun": "no-onion-garlic",
    "gluten free": "gluten-free",
}

PORTABILITY_SIGNALS = [
    "travel", "picnic", "on the go", "takeaway", "packed lunch", "tiffin"
]


def detect_signals(raw_query):
    """Extract all reranking signals from raw query text."""
    q = raw_query.lower()
    signals = {}

    # Spice
    for phrase, tag in SPICE_SIGNALS.items():
        if phrase in q:
            signals["spice"] = tag
            break

    # Textures — can be multiple
    signals["textures"] = [t for t in TEXTURE_SIGNALS if t in q]

    # Cooking method
    for phrase, method in COOKING_SIGNALS.items():
        if phrase in q:
            signals["cooking"] = method
            break

    # Health
    for phrase, tag in HEALTH_SIGNALS.items():
        if phrase in q:
            signals["health"] = tag
            break

    # Time
    for phrase, time in TIME_SIGNALS.items():
        if phrase in q:
            signals["time"] = time
            break

    # Serving format
    for phrase, fmt in SERVING_SIGNALS.items():
        if phrase in q:
            signals["serving"] = fmt
            break

    # Occasion
    for phrase, occ in OCCASION_SIGNALS.items():
        if phrase in q:
            signals["occasion"] = occ
            break

    # Dietary
    for phrase, tag in DIETARY_SIGNALS.items():
        if phrase in q:
            signals["dietary"] = tag
            break

    # Portability
    if any(p in q for p in PORTABILITY_SIGNALS):
        signals["portable"] = True

    return signals


def score_hit(hit, signals, raw_query):
    """Score a single retrieval hit against detected signals."""
    q = raw_query.lower()
    boost = 0.0
    boost_reasons = []

    tags = hit.get("tags", [])
    health_tags = hit.get("health_tags", []) or []
    time_affinity = hit.get("time_affinity", []) or []
    occasion_tags = hit.get("occasion_tags", []) or []
    dietary_tags = hit.get("dietary_tags", []) or []
    cooking_method = hit.get("cooking_method", "") or ""
    serving_format = hit.get("serving_format", "") or ""
    cuisine_type = hit.get("cuisine_type", "") or ""
    holds_well = hit.get("holds_well", False)
    portability = hit.get("portability", False)

    # Spice match
    spice_signal = signals.get("spice")
    if spice_signal:
        if spice_signal in tags:
            boost += 0.15
            boost_reasons.append(f"spice:{spice_signal}+0.15")

    # Texture match
    for texture in signals.get("textures", []):
        if texture in tags or texture in q:
            boost += 0.10
            boost_reasons.append(f"texture:{texture}+0.10")

    # Cooking method match
    cooking_signal = signals.get("cooking")
    if cooking_signal and cooking_method == cooking_signal:
        boost += 0.12
        boost_reasons.append(f"cooking:{cooking_signal}+0.12")

    # Health signal match
    health_signal = signals.get("health")
    if health_signal:
        if health_signal in health_tags:
            boost += 0.15
            boost_reasons.append(f"health:{health_signal}+0.15")
        # Calorie band bonus for low-cal queries
        if health_signal == "low-cal" and hit.get("calorie_band") == "low":
            boost += 0.08
            boost_reasons.append("calorie_band:low+0.08")

    # Time affinity match
    time_signal = signals.get("time")
    if time_signal and time_signal in time_affinity:
        boost += 0.12
        boost_reasons.append(f"time:{time_signal}+0.12")

    # Serving format match
    serving_signal = signals.get("serving")
    if serving_signal:
        if serving_format == serving_signal:
            boost += 0.10
            boost_reasons.append(f"serving:{serving_signal}+0.10")
        # Party/drinks queries also boost holds_well
        if serving_signal == "finger-food" and holds_well:
            boost += 0.08
            boost_reasons.append("holds_well+0.08")

    # Occasion match
    occasion_signal = signals.get("occasion")
    if occasion_signal and occasion_signal in occasion_tags:
        boost += 0.12
        boost_reasons.append(f"occasion:{occasion_signal}+0.12")

    # Dietary match
    dietary_signal = signals.get("dietary")
    if dietary_signal and dietary_signal in dietary_tags:
        boost += 0.15
        boost_reasons.append(f"dietary:{dietary_signal}+0.15")

    # Portability match
    if signals.get("portable") and portability:
        boost += 0.10
        boost_reasons.append("portable+0.10")

    return round(boost, 4), boost_reasons


def deduplicate_and_diversify(ranked, top_n=5, debug=True):
    """
    Post-rerank deduplication and diversity enforcement.

    Rules:
    1. Deduplicate by dish name — keep highest scoring instance,
       track all restaurants for UI "available at X restaurants" cue
    2. Category diversity — max 2 dishes of same category_type in top_n
    3. Restaurant diversity — max 2 dishes from same restaurant in top_n
    4. Cuisine diversity — max 3 dishes of same cuisine_type in top_n

    Returns top_n diverse unique dishes + full restaurant list per dish.
    """
    # Step 1: Deduplicate by dish name
    seen_names = {}
    for hit in ranked:
        # Normalise spelling variations (e.g. "Vada Paav" vs "Vada Pav")
        name = hit["dish"].lower().strip()
        name = name.replace("paav", "pav").replace("aloo", "alu").replace("alu ", "aloo ")
        name = ' '.join(name.split())  # normalise whitespace
        if name not in seen_names:
            seen_names[name] = hit.copy()
            seen_names[name]["all_restaurants"] = [hit["restaurant"]]
        else:
            seen_names[name]["all_restaurants"].append(hit["restaurant"])

    deduped = list(seen_names.values())

    if debug:
        removed_dupes = len(ranked) - len(deduped)
        if removed_dupes > 0:
            print(f"[DEDUP] Removed {removed_dupes} duplicate dish names")

    # Step 2: Diversity enforcement
    final = []
    category_counts = {}
    restaurant_counts = {}
    cuisine_counts = {}

    MAX_SAME_CATEGORY = 2
    MAX_SAME_RESTAURANT = 2
    MAX_SAME_CUISINE = 3

    for hit in deduped:
        if len(final) >= top_n:
            break

        cat = hit.get("category_type", "")
        restaurant = hit.get("restaurant", "")
        cuisine = hit.get("cuisine_type", "")

        # Check diversity constraints
        if category_counts.get(cat, 0) >= MAX_SAME_CATEGORY and cat:
            if debug:
                print(f"[DIVERSITY] Skipping {hit['dish']} — too many {cat} dishes")
            continue

        if restaurant_counts.get(restaurant, 0) >= MAX_SAME_RESTAURANT and restaurant:
            if debug:
                print(f"[DIVERSITY] Skipping {hit['dish']} — too many dishes from {restaurant}")
            continue

        if cuisine_counts.get(cuisine, 0) >= MAX_SAME_CUISINE and cuisine:
            if debug:
                print(f"[DIVERSITY] Skipping {hit['dish']} — too many {cuisine} dishes")
            continue

        # Add to final list
        final.append(hit)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        restaurant_counts[restaurant] = restaurant_counts.get(restaurant, 0) + 1
        cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1

    # If diversity was too strict and we have fewer than top_n, fill remaining slots
    if len(final) < top_n:
        final_names = {h['dish'].lower().strip() for h in final}
        for hit in deduped:
            if hit['dish'].lower().strip() not in final_names:
                final.append(hit)
                final_names.add(hit['dish'].lower().strip())
            if len(final) >= top_n:
                break

    if debug:
        print(f"\n[FINAL] Top {len(final)} after dedup + diversity:")
        for i, h in enumerate(final, 1):
            r_count = len(h.get("all_restaurants", [h["restaurant"]]))
            print(f"  {i}. {h['dish']} @ {h['restaurant']} "
                  f"(+{r_count-1} more restaurants) | "
                  f"score: {h['rerank_score']}")

    return final


def rerank(hits, raw_query, top_n=5, debug=True):
    """
    Full reranking pipeline:
    1. Score all hits on metadata signals
    2. Sort by rerank_score
    3. Deduplicate by dish name
    4. Enforce category/restaurant/cuisine diversity
    5. Return top_n

    Args:
        hits: list of dicts from Pinecone retrieval (top 10)
        raw_query: original user query string
        top_n: number of results to return (default 5)
        debug: if True, prints scoring breakdown for all hits

    Returns:
        top_n diverse unique hits with rerank_score and all_restaurants added
    """
    signals = detect_signals(raw_query)

    if debug:
        print(f"\n[RERANKER] Signals detected: {signals}")
        print(f"[RERANKER] Scoring {len(hits)} candidates:\n")

    for hit in hits:
        boost, reasons = score_hit(hit, signals, raw_query)
        hit["rerank_score"] = round(hit["score"] + boost, 4)
        hit["boost"] = boost
        hit["boost_reasons"] = reasons

    # Sort by rerank_score descending
    ranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)

    if debug:
        for i, h in enumerate(ranked):
            reasons_str = ", ".join(h["boost_reasons"]) if h["boost_reasons"] else "no boost"
            print(
                f"  {i+1}. {h['dish']} @ {h['restaurant']} | "
                f"cosine: {h['score']} | boost: +{h['boost']} ({reasons_str}) | "
                f"final: {h['rerank_score']}"
            )

    # Deduplicate and enforce diversity
    return deduplicate_and_diversify(ranked, top_n=top_n, debug=debug)
