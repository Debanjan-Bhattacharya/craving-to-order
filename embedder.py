import json
import os
import time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

# --- CONFIG ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNKS_PATH = "chunks.json"
OUTPUT_PATH = "embedded_chunks.json"
BATCH_SIZE = 100  # Ada-002 supports up to 2048 inputs per call; 100 is safe

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_batch(texts):
    """Embed a batch of texts. Returns list of vectors."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    # Response contains embeddings in same order as input
    return [item.embedding for item in response.data]

def embed_all_chunks(chunks_path, output_path):
    with open(chunks_path) as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"Embedding {total} chunks in batches of {BATCH_SIZE}...")

    embedded = []
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = embed_batch(texts)

        for chunk, vector in zip(batch, vectors):
            embedded.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "embedding": vector,       # 1536-dim float list
                "metadata": chunk["metadata"]
            })

        print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total}")
        time.sleep(0.5)  # Avoid rate limit on free tier

    # Save locally before Pinecone upsert
    with open(output_path, "w") as f:
        json.dump(embedded, f)

    print(f"\nDone. Saved to {output_path}")
    print(f"Sample vector (first 5 dims): {embedded[0]['embedding'][:5]}")
    print(f"Vector length: {len(embedded[0]['embedding'])}")
    return embedded

if __name__ == "__main__":
    embed_all_chunks(CHUNKS_PATH, OUTPUT_PATH)
