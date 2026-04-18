import json
from collections import Counter
from statistics import mean
from sklearn.model_selection import train_test_split
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"
TOP_K = 5
RANDOM_STATE = 42

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

def fetch_cases():
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)
            OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
            RETURN
                c.doc_id AS doc_id,
                collect(DISTINCT lr.name) AS legal_refs,
                collect(DISTINCT t.name) AS topics
        """)

        cases = []
        for r in result:
            legal_refs = [x for x in (r["legal_refs"] or []) if x]
            topics = [x for x in (r["topics"] or []) if x]

            if not legal_refs:
                continue

            cases.append({
                "doc_id": r["doc_id"],
                "legal_refs": legal_refs,
                "topics": topics
            })

    return cases

def predict_legal_refs(doc_id, train_ids, fallback_labels):
    """
    KG-only prediction:
    - find cases sharing topics with the target case
    - keep only training cases
    - collect the legal refs they mention
    - rank by frequency
    """
    with driver.session(database="neo4j") as session:
        result = session.run("""
            MATCH (target:Case {doc_id: $doc_id})-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(other:Case)
            WHERE other.doc_id IN $train_ids AND other.doc_id <> $doc_id
            MATCH (other)-[:MENTIONS]->(lr:LegalReference)
            RETURN lr.name AS label, count(*) AS score
            ORDER BY score DESC, label ASC
            LIMIT 5
        """, {
            "doc_id": doc_id,
            "train_ids": list(train_ids)
        })

        preds = [r["label"] for r in result if r["label"]]

    if not preds:
        preds = fallback_labels[:TOP_K]

    return preds[:TOP_K]

def precision_recall_f1(pred, actual, k=5):
    pred_k = pred[:k]
    actual_set = set(actual)

    if not pred_k or not actual_set:
        return None

    hits = len(set(pred_k) & actual_set)
    precision = hits / len(pred_k)
    recall = hits / len(actual_set)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    hit_rate = 1 if hits > 0 else 0

    return precision, recall, f1, hit_rate

def main():
    cases = fetch_cases()
    print("Total usable cases:", len(cases))

    doc_ids = [c["doc_id"] for c in cases]
    train_ids, test_ids = train_test_split(
        doc_ids,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_set = set(train_ids)
    test_set = set(test_ids)

    # Baseline = most frequent statutes in the training split
    label_counter = Counter()
    for c in cases:
        if c["doc_id"] in train_set:
            label_counter.update(c["legal_refs"])

    baseline_topk = [label for label, _ in label_counter.most_common(TOP_K)]
    print("Baseline top labels:", baseline_topk)

    graph_precisions = []
    graph_recalls = []
    graph_f1s = []
    graph_hits = []

    base_precisions = []
    base_recalls = []
    base_f1s = []
    base_hits = []

    example_shown = 0

    for case in cases:
        if case["doc_id"] not in test_set:
            continue

        actual = case["legal_refs"]
        if not actual:
            continue

        pred_graph = predict_legal_refs(case["doc_id"], train_set, baseline_topk)
        pred_base = baseline_topk[:TOP_K]

        g = precision_recall_f1(pred_graph, actual, TOP_K)
        b = precision_recall_f1(pred_base, actual, TOP_K)

        if g is not None:
            p, r, f1, hit = g
            graph_precisions.append(p)
            graph_recalls.append(r)
            graph_f1s.append(f1)
            graph_hits.append(hit)

        if b is not None:
            p, r, f1, hit = b
            base_precisions.append(p)
            base_recalls.append(r)
            base_f1s.append(f1)
            base_hits.append(hit)

        if example_shown < 5:
            print("\n--- Example ---")
            print("Case:", case["doc_id"])
            print("Actual :", actual[:TOP_K])
            print("KG pred:", pred_graph)
            print("Base   :", pred_base)
            example_shown += 1

    print("\n=== KG Recommendation Results ===")
    print("Cases evaluated:", len(graph_precisions))
    print("Precision@5 :", round(mean(graph_precisions), 4) if graph_precisions else 0)
    print("Recall@5    :", round(mean(graph_recalls), 4) if graph_recalls else 0)
    print("F1@5        :", round(mean(graph_f1s), 4) if graph_f1s else 0)
    print("HitRate@5   :", round(mean(graph_hits), 4) if graph_hits else 0)

    print("\n=== Baseline Results ===")
    print("Precision@5 :", round(mean(base_precisions), 4) if base_precisions else 0)
    print("Recall@5    :", round(mean(base_recalls), 4) if base_recalls else 0)
    print("F1@5        :", round(mean(base_f1s), 4) if base_f1s else 0)
    print("HitRate@5   :", round(mean(base_hits), 4) if base_hits else 0)

    driver.close()

if __name__ == "__main__":
    main()