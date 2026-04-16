from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from generator import recommend

app = FastAPI(title="Craving-to-Order API")

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    conversation_history: Optional[list] = []

class DishResult(BaseModel):
    dish: str
    restaurant: str
    price: int
    score: float
    rerank_score: float
    tags: list
    cuisine_type: str = ""
    cooking_method: str = ""
    health_tags: list = []
    time_affinity: list = []
    serving_format: str = ""

@app.get("/")
def root():
    return {"status": "Craving-to-Order API is running"}

@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    result = recommend(
        request.query,
        conversation_history=request.conversation_history
    )
    return {
        "query": request.query,
        "response": result["response"],
        "hits": result["hits"],
        "cost": result["cost"],
        "hallucination_flagged": result["hallucination_flagged"]
    }