from neo4j import GraphDatabase
import json

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

data = []

with driver.session(database="neo4j") as session:
    result = session.run("""
        MATCH (c:Case)-[:HAS_FULL_TEXT]->(t:JudgmentText)
        OPTIONAL MATCH (c)-[:MENTIONS]->(l:LegalReference)
        OPTIONAL MATCH (c)-[:CITES]->(cc:CitedCase)
        RETURN 
            c.doc_id AS doc_id,
            t.text AS text,
            collect(DISTINCT l.name) AS legal_refs,
            collect(DISTINCT cc.title) AS citations
    """)

    for record in result:
        data.append({
            "doc_id": record["doc_id"],
            "text": record["text"] or "",
            "legal_refs": record["legal_refs"],
            "citations": record["citations"]
        })

driver.close()

# Save dataset
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Saved dataset")
print("Total samples:", len(data))
print(data[0])