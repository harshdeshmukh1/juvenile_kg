from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ====================== CONFIG ======================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

CHUNK_SIZE_WORDS = 250
OVERLAP_WORDS = 40

# ====================== MODEL ======================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ====================== NEO4J ======================
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def chunk_text(text, chunk_size=250, overlap=40):
    words = text.split()
    if not words:
        return

    step = max(1, chunk_size - overlap)
    chunk_no = 0

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        yield chunk_no, " ".join(chunk_words)
        chunk_no += 1

def main():
    print("Starting chunking and embedding...")

    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)-[:HAS_FULL_TEXT]->(t:JudgmentText)
            RETURN c.doc_id AS doc_id, t.text AS text
        """)

        count_cases = 0
        total_chunks = 0

        for record in result:
            doc_id = record["doc_id"]
            full_text = record["text"]

            if not full_text or not full_text.strip():
                print(f"Skipping {doc_id}: empty text")
                continue

            count_cases += 1
            chunks = list(chunk_text(full_text, CHUNK_SIZE_WORDS, OVERLAP_WORDS))

            if not chunks:
                print(f"Skipping {doc_id}: no chunks created")
                continue

            chunk_texts = [chunk for _, chunk in chunks]
            embeddings = embedder.encode(chunk_texts, batch_size=32, show_progress_bar=False)

            print(f"{doc_id}: {len(chunks)} chunks")

            for (chunk_no, chunk), emb in zip(chunks, embeddings):
                chunk_id = f"{doc_id}_{chunk_no}"

                session.run("""
                    MATCH (c:Case {doc_id: $doc_id})
                    MERGE (tc:TextChunk {chunk_id: $chunk_id})
                    SET tc.doc_id = $doc_id,
                        tc.chunk_no = $chunk_no,
                        tc.text = $text,
                        tc.embedding = $embedding
                    MERGE (c)-[:HAS_CHUNK]->(tc)
                """, {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_no": chunk_no,
                    "text": chunk,
                    "embedding": emb.tolist()
                }).consume()

                total_chunks += 1

    driver.close()
    print(f"Done. Cases processed: {count_cases}, chunks created: {total_chunks}")

if __name__ == "__main__":
    main()