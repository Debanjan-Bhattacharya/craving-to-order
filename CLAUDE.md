# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Craving-to-Order** is a RAG-based food recommendation chatbot for Delhi restaurants. Users describe a food craving in natural language (including Hindi terms), and the app retrieves matching dishes from a Pinecone vector index and generates conversational recommendations using GPT.

## Setup

**Install dependencies:**
```bash
pip install streamlit openai pinecone python-dotenv
```

**Environment (`.env` file required):**
```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
```

## Running the App

```bash
streamlit run app.py
```

Streamlit config (theme, colors) is in `.streamlit/config.toml`.

## Running Evaluation

```bash
python eval.py
```

Runs 56 test queries across 5 categories (response quality, retrieval, constraint adherence, hallucination, adversarial). Outputs a CSV (`eval_results_v*.csv`) with 4-point Likert scores.

Each module also has a `__main__` block for manual testing:
```bash
python retrieval.py   # Test retrieval pipeline
python generator.py   # Test generation pipeline
```

## Architecture

### Query Pipeline (production path)

```
User query → retrieval.py → generator.py → app.py (Streamlit UI)
```

1. **`retrieval.py`** — Takes a user query + conversation history, translates Hindi terms, expands the query via GPT, embeds it with `text-embedding-3-small`, applies metadata filters (budget, diet, cuisine, taste, ingredient, category), and returns top-k Pinecone hits. Caches results for repeated queries (without conversation context).

2. **`generator.py`** — Takes retrieval hits + conversation history, generates a conversational recommendation (up to 5 dishes) via `gpt-4o-mini`, then validates each dish against actual retrieval results (hallucination guard). Returns the response with per-query token cost breakdown.

3. **`app.py`** — Streamlit UI with session state for conversation history (last 3 turns retained), query history (last 10), and running cost/query stats.

### Data Pipeline (for regenerating the dataset)

Run in order to rebuild the dataset from raw menu text files:

```bash
python extract_restaurant.py <menu_file>  # Extract structured data from raw menu text
python batch_extract.py                   # Batch version of above
python enrich_metadata.py                 # Add semantic tags (cuisine, cooking method, diet, occasion)
python merge_restaurants.py               # Combine all extracted/enriched data
python chunker.py [dataset_enriched.json] # Convert dishes to natural language chunks
python embedder.py                        # Embed chunks with OpenAI (writes embedded_chunks.json)
python upsert_pinecone.py                 # Upload embeddings + metadata to Pinecone
```

### Key Data Files

| File | Description |
|------|-------------|
| `dataset_enriched.json` | 17 restaurants, 674 dishes with semantic metadata |
| `chunks.json` | Natural language chunks ready for embedding |
| `embedded_chunks.json` | 1536-dim OpenAI embeddings (21 MB, not re-embed unless necessary) |
| `menus/` | Raw restaurant menu text files (source of truth) |

### Models Used

- **`gpt-4o-mini`** — Query expansion, response generation, hallucination validation
- **`text-embedding-3-small`** — Semantic search embeddings
- **Pinecone index** — Hosted at `craving-to-order-9udw9bu.svc.aped-4627-b74a.pinecone.io`

### Cost Tracking

Rates hardcoded in `generator.py`: $0.15/1M input tokens, $0.60/1M output tokens, $0.02/1M embedding tokens. The UI shows per-query and cumulative costs.
