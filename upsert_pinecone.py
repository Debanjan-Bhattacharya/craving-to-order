import json
import os
from pinecone import Pinecone

# --- CONFIG ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_HOST = "https://craving-to-order-9udw9bu.svc.aped-4627-b74a.pinecone.io"
EMBEDDED_CHUNKS_PATH = "embedded_chunks.json"
BATCH_SIZE = 100

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=INDEX_HOST)

def upsert_all(embedded_chunks_path):
    with open(embedded_chunks_path) as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"Upserting {total} vectors to Pinecone...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # Pinecone expects: list of (id, vector, metadata)
        vectors = []
        for c in batch:
            # Pinecone doesn't accept null — convert None to empty string
            clean_metadata = {
                k: ("" if v is None else v)
                for k, v in c["metadata"].items()
            }
            vectors.append({
                "id": c["id"],
                "values": c["embedding"],
                "metadata": clean_metadata
            })

        index.upsert(vectors=vectors)
        print(f"  Upserted {min(i + BATCH_SIZE, total)}/{total}")

    print("\nDone. Verifying index stats...")
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")

if __name__ == "__main__":
    upsert_all(EMBEDDED_CHUNKS_PATH)
