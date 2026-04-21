"""
reranker.py
Scores candidates on full signal alignment + diversity bonus.
Greedy selection: maximises both score AND variety.
"""

from retrieval import is_duplicate_dish

SIGNAL_WEIGHTS = {
    "diet":           0.20,
    "budget":         0.15,
    "festival":       0.15,
    "cuisine_type":   0.12,
    "cuisine_region": 0.10,
    "category_type":  0.10,
    "texture":        0.08,
    "taste":          0.08,
    "occasion":       0.06,
    "time":           0.05,
    "health":         0.05,
    "protein_band":   0.04,
    "calorie_band":   0.04,
    "cooking_method": 0.03,
    "serving_format": 0.03,
    "holds_well":     0.03,
    "portability":    0.03,
    "cosine":         0.10,
}

DIVERSITY_BONUS = {
    "cuisine_type":  0.15,
    "category_type": 0.10,
    "restaurant":    0.05,
}


def to_set(value) -> set:
    if not value:
        return set()
    if isinstance(value, list):
        return set(v.lower().strip() for v in value if v)
    return set(v.lower().strip() for v in str(value).split("|") if v.strip())


def overlap_score(query_vals: list, hit_val) -> float:
    if not query_vals:
        return 0.0
    query_set = set(v.lower().strip() for v in query_vals if v)
    hit_set   = to_set(hit_val)
    if not query_set or not hit_set:
        return 0.0
    return len(query_set & hit_set) / len(query_set)


def hard_match(query_val, hit_val) -> float:
    if not query_val:
        return 0.0
    if isinstance(hit_val, str):
        return 1.0 if query_val.lower() == hit_val.lower() else 0.0
    return 0.0


def score_hit(hit: dict, signals: dict, selected: list) -> float:
    score = 0.0
    score += hit.get("score", 0.0) * SIGNAL_WEIGHTS["cosine"]

    diet = signals.get("diet")
    if diet:
        if diet == "veg" and hit.get("is_veg") != "veg":
            return 0.0
        elif diet == "non-veg" and hit.get("is_veg") == "veg":
            return 0.0
        elif diet in ("jain", "vrat"):
            if diet not in to_set(hit.get("dietary_tags", "")):
                return 0.0
        score += SIGNAL_WEIGHTS["diet"]

    budget = signals.get("budget")
    if budget:
        try:
            if float(hit.get("price", 0)) <= float(budget):
                score += SIGNAL_WEIGHTS["budget"]
            else:
                return 0.0
        except (ValueError, TypeError):
            pass

    festival_signals = signals.get("festival") or []
    if festival_signals:
        score += overlap_score(festival_signals, hit.get("festival_tags", "")) * SIGNAL_WEIGHTS["festival"]

    cuisine_signals = signals.get("cuisine_type") or []
    if cuisine_signals:
        score += overlap_score(cuisine_signals, hit.get("cuisine_type", "")) * SIGNAL_WEIGHTS["cuisine_type"]

    region_signals = signals.get("cuisine_region") or []
    if region_signals:
        score += overlap_score(region_signals, hit.get("cuisine_region", "")) * SIGNAL_WEIGHTS["cuisine_region"]

    category_signals = signals.get("category") or []
    if category_signals:
        score += overlap_score(category_signals, hit.get("category_type", "")) * SIGNAL_WEIGHTS["category_type"]

    texture_signals = signals.get("texture") or []
    if texture_signals:
        score += overlap_score(texture_signals, hit.get("texture_tags", "")) * SIGNAL_WEIGHTS["texture"]

    taste_signals = signals.get("taste") or []
    if taste_signals:
        score += overlap_score(taste_signals, hit.get("taste_tags", "")) * SIGNAL_WEIGHTS["taste"]

    occasion_signals = signals.get("occasion") or []
    if occasion_signals:
        score += overlap_score(occasion_signals, hit.get("occasion_tags", "")) * SIGNAL_WEIGHTS["occasion"]

    time_signals = signals.get("time") or []
    if time_signals:
        score += overlap_score(time_signals, hit.get("time_affinity", "")) * SIGNAL_WEIGHTS["time"]

    health_signals = signals.get("health") or []
    if health_signals:
        score += overlap_score(health_signals, hit.get("health_tags", "")) * SIGNAL_WEIGHTS["health"]

    protein_signal = signals.get("protein_band")
    if protein_signal:
        score += hard_match(protein_signal, hit.get("protein_band", "")) * SIGNAL_WEIGHTS["protein_band"]

    calorie_signal = signals.get("calorie_band")
    if calorie_signal:
        score += hard_match(calorie_signal, hit.get("calorie_band", "")) * SIGNAL_WEIGHTS["calorie_band"]

    cooking_signals = signals.get("cooking_method") or []
    if cooking_signals:
        score += overlap_score(cooking_signals, hit.get("cooking_method", "")) * SIGNAL_WEIGHTS["cooking_method"]

    format_signals = signals.get("serving_format") or []
    if format_signals:
        score += overlap_score(format_signals, hit.get("serving_format", "")) * SIGNAL_WEIGHTS["serving_format"]

    if signals.get("holds_well") is True:
        score += (1.0 if hit.get("holds_well", "").lower() == "yes" else 0.0) * SIGNAL_WEIGHTS["holds_well"]

    if signals.get("portability") is True:
        score += (1.0 if hit.get("portability", "").lower() == "yes" else 0.0) * SIGNAL_WEIGHTS["portability"]

    # Penalty for standalone bread/beverage/side-dish/dessert when no category signal
    if not (signals.get("category") or []):
        if hit.get("category_type", "") in ("bread", "side-dish", "beverage", "dessert"):
            score *= 0.3

    # Diversity bonus
    specified_cuisines    = set(signals.get("cuisine_type") or [])
    specified_restaurants = set(signals.get("restaurant_include") or [])
    specified_categories  = set(signals.get("category") or [])

    selected_cuisines    = set(h.get("cuisine_type", "").lower() for h in selected)
    selected_categories  = set(h.get("category_type", "").lower() for h in selected)
    selected_restaurants = set(h.get("restaurant", "").lower() for h in selected)

    if not specified_cuisines and hit.get("cuisine_type", "").lower() not in selected_cuisines:
        score += DIVERSITY_BONUS["cuisine_type"]

    if not specified_categories and hit.get("category_type", "").lower() not in selected_categories:
        score += DIVERSITY_BONUS["category_type"]

    if not specified_restaurants and hit.get("restaurant", "").lower() not in selected_restaurants:
        score += DIVERSITY_BONUS["restaurant"]

    return round(score, 4)


def rerank(hits: list, raw_query: str, top_n: int = 5, debug: bool = False) -> list:
    if not hits:
        return []

    signals = hits[0].get("_signals", {}) if hits else {}

    if debug:
        active = {k: v for k, v in signals.items()
                  if v and k not in ("embedding_query", "dish_exclude", "_raw_query")}
        print(f"\n[RERANKER] Active signals: {active}")
        print(f"[RERANKER] Scoring {len(hits)} candidates:")

    selected  = []
    remaining = list(hits)

    while len(selected) < top_n and remaining:
        scored = [(score_hit(h, signals, selected), h) for h in remaining]
        scored.sort(key=lambda x: x[0], reverse=True)

        if debug:
            for s, h in scored[:5]:
                print(f"  {h['dish'][:40]:40s} | score: {s:.4f} | "
                      f"cuisine: {h.get('cuisine_type','')[:15]} | "
                      f"cat: {h.get('category_type','')[:12]}")

        best_score, best_hit = scored[0]

        if best_score <= 0.0:
            break

        if is_duplicate_dish(best_hit["dish"], [h["dish"] for h in selected], threshold=0.82):
            remaining.remove(best_hit)
            continue

        selected.append(best_hit)
        remaining.remove(best_hit)

        if debug:
            print(f"  → Selected: {best_hit['dish']} (score: {best_score:.4f})\n")

    if debug:
        print(f"\n[RERANKER] Final {len(selected)} selections:")
        for i, h in enumerate(selected, 1):
            print(f"  {i}. {h['dish']} @ {h['restaurant']} | "
                  f"cuisine: {h.get('cuisine_type','')} | "
                  f"cat: {h.get('category_type','')} | "
                  f"₹{h.get('price','')}")

    for h in selected:
        h.pop("_signals", None)

    return selected
