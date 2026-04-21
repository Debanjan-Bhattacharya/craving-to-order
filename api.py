from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from generator import recommend
from intent_classifier import classify_intent, OUT_OF_SCOPE_RESPONSE
from retrieval import is_duplicate_dish, get_similar_dishes, DISH_NAME_EMBEDDINGS

import json
with open("dish_lookup.json", encoding="utf-8") as f:
    DISH_LOOKUP = json.load(f)

app = FastAPI(title="Craving-to-Order API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "Craving-to-Order API is running"}


@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    query = request.query.strip()

    # ── Step 1: Intent classification ─────────────────────────────────────────
    intent_result = classify_intent(query)
    intent        = intent_result.get("intent", "food_recommendation")

    # ── Step 2: Route by intent ────────────────────────────────────────────────

    # Out of scope
    if intent == "out_of_scope":
        return {
            "query":                query,
            "response":             OUT_OF_SCOPE_RESPONSE,
            "hits":                 [],
            "cost":                 {"estimated_cost_usd": 0},
            "hallucination_flagged": False,
        }

    # Surprise me
    if intent == "surprise_me":
        query = "varied food selection across different cuisines and categories"

    # Exact dish lookup — show exact match first + recommendation pipeline
    if intent == "exact_dish_lookup" and len(query.strip().split()) >= 2:
        dish_name = query
        exact = DISH_LOOKUP.get(dish_name.strip().lower())
        if exact:
            # Get similar dishes for exact hit first
            exact_similar = get_similar_dishes(
                {
                    "dish":          exact["dish_name"],
                    "cuisine_type":  exact["cuisine_type"],
                    "category_type": exact["category_type"],
                    "taste_tags":    exact["taste_tags"],
                    "texture_tags":  exact["texture_tags"],
                },
                [exact["dish_name"]],
                top_n=5
            )

            reco_result = recommend(query)
            reco_hits   = reco_result.get("hits", [])

            for hit in reco_hits:
                key = hit["dish"].lower()
                agg = DISH_LOOKUP.get(key, {})
                hit["restaurant_count"] = agg.get("restaurant_count", 1)
                hit["price_min"]        = agg.get("price_min", hit.get("price", 0))
                hit["price_max"]        = agg.get("price_max", hit.get("price", 0))
                hit["all_restaurants"]  = agg.get("all_restaurants", [hit.get("restaurant", "")])

            # Exclude exact similar from reco similar dishes
            exact_similar_names = [s["dish"].lower() for s in exact_similar]
            for hit in reco_hits:
                hit["similar_dishes"] = [
                    s for s in hit.get("similar_dishes", [])
                    if s["dish"].lower() not in exact_similar_names
                ]

            exact_hit = {
                "dish":             exact["dish_name"],
                "restaurant":       exact["all_restaurants"][0] if exact["all_restaurants"] else "",
                "price":            exact["price_min"],
                "is_veg":           exact["is_veg"],
                "cuisine_type":     exact["cuisine_type"],
                "category_type":    exact["category_type"],
                "taste_tags":       exact["taste_tags"],
                "texture_tags":     exact["texture_tags"],
                "restaurant_count": exact["restaurant_count"],
                "price_min":        exact["price_min"],
                "price_max":        exact["price_max"],
                "all_restaurants":  exact["all_restaurants"],
                "similar_dishes":   exact_similar,
                "score":            1.0,
            }

            merged = [exact_hit] + [
                h for h in reco_hits
                if not is_duplicate_dish(h["dish"], [exact["dish_name"]])
            ][:4]

            return {
                "query":                query,
                "response":             reco_result.get("response", ""),
                "hits":                 merged,
                "cost":                 reco_result.get("cost", {"estimated_cost_usd": 0}),
                "hallucination_flagged": reco_result.get("hallucination_flagged", False),
            }

    # Restaurant menu
    if intent == "restaurant_menu":
        restaurant = intent_result.get("restaurant_name")
        if restaurant:
            query = f"{query} only from {restaurant}"

    # ── Step 3: Full recommendation pipeline ──────────────────────────────────
    result = recommend(query)

    # Extract budget for restaurant filtering
    from retrieval import extract_signals
    _signals = extract_signals(query)
    _budget  = _signals.get("budget")

    for hit in result.get("hits", []):
        key = hit["dish"].lower()
        agg = DISH_LOOKUP.get(key, {})
        hit["restaurant_count"] = agg.get("restaurant_count", 1)
        hit["price_min"]        = agg.get("price_min", hit.get("price", 0))
        hit["price_max"]        = agg.get("price_max", hit.get("price", 0))
        hit["all_restaurants"]  = agg.get("all_restaurants", [hit.get("restaurant", "")])

        if _budget:
            hit["all_restaurants"]  = [hit.get("restaurant", "")]
            hit["restaurant_count"] = 1
            hit["price_min"]        = hit.get("price", 0)
            hit["price_max"]        = hit.get("price", 0)

    return {
        "query":                query,
        "response":             result["response"],
        "hits":                 result["hits"],
        "cost":                 result["cost"],
        "hallucination_flagged": result["hallucination_flagged"],
    }
