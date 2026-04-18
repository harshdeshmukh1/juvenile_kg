import time
from collections import Counter
from statistics import mean
from sklearn.model_selection import train_test_split
from neo4j import GraphDatabase

# ================= CONFIG =================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

TOP_K = 5
RANDOM_STATE = 42

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================================================
# STEP 1: FETCH DATA FROM KNOWLEDGE GRAPH
# =========================================================
def fetch_cases():
    """
    We extract:
    - Case ID
    - Legal references (labels to predict)
    - Topics (used for similarity)
    """

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

            # Ignore cases with no labels
            if not legal_refs:
                continue

            cases.append({
                "doc_id": r["doc_id"],
                "legal_refs": legal_refs,
                "topics": topics
            })

    return cases


# =========================================================
# STEP 2: KG-BASED PREDICTION (CORE IDEA)
# =========================================================
def predict_legal_refs(doc_id, train_ids, fallback_labels):
    """
    CORE IDEA (IMPORTANT):

    We do NOT use weak signals independently.

    Instead:
    1. Find cases that share TOPICS
    2. AND share CITATIONS (>=2) → strong legal similarity
    3. Optionally boost if same court
    4. Then collect their legal references

    This ensures:
    - Less noise
    - Higher precision
    """

    with driver.session(database="neo4j") as session:

        result = session.run("""
            // =============================
            // STEP 1: Target case
            // =============================
            MATCH (target:Case {doc_id: $doc_id})

            // =============================
            // STEP 2: Topic similarity
            // Find cases with SAME topic
            // =============================
            MATCH (target)-[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-(c:Case)

            // =============================
            // STEP 3: Citation similarity
            // Find cases sharing cited cases
            // =============================
            MATCH (target)-[:CITES]->(cc:CitedCase)<-[:CITES]-(c)

            // =============================
            // STEP 4: Only training cases
            // =============================
            WHERE c.doc_id IN $train_ids AND c.doc_id <> $doc_id

            // =============================
            // STEP 5: Count shared citations
            // This removes weak overlaps
            // =============================
            WITH c, count(DISTINCT cc) AS common_cites

            // =============================
            // STEP 6: Strong filtering
            // Only keep strong matches
            // =============================
            WHERE common_cites >= 2

            // =============================
            // STEP 7: Optional court bonus
            // Court is weak → only bonus
            // =============================
            OPTIONAL MATCH (target)-[:HEARD_IN]->(court)<-[:HEARD_IN]-(c)

            WITH c, common_cites,
                 CASE WHEN court IS NOT NULL THEN 0.5 ELSE 0 END AS court_bonus

            // =============================
            // STEP 8: Get legal references
            // =============================
            MATCH (c)-[:MENTIONS]->(lr:LegalReference)

            // =============================
            // STEP 9: Final scoring
            // More shared citations = stronger similarity
            // =============================
            RETURN 
                lr.name AS label,
                (2.0 * common_cites + court_bonus) AS score
        """, {
            "doc_id": doc_id,
            "train_ids": list(train_ids)
        })

        # Aggregate scores
        scores = {}
        for r in result:
            label = r["label"]
            score = r["score"]
            if label:
                scores[label] = scores.get(label, 0) + score

        # Sort and take top K
        sorted_labels = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        preds = [label for label, _ in sorted_labels[:TOP_K]]

    # Fallback if no prediction
    if not preds:
        preds = fallback_labels[:TOP_K]

    return preds


# =========================================================
# STEP 3: EVALUATION METRICS
# =========================================================
def precision_recall_f1(pred, actual, k=5):
    pred_k = pred[:k]
    actual_set = set(actual)

    if not pred_k or not actual_set:
        return None

    hits = len(set(pred_k) & actual_set)

    precision = hits / len(pred_k)
    recall = hits / len(actual_set)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    hit_rate = 1 if hits > 0 else 0

    return precision, recall, f1, hit_rate


# =========================================================
# STEP 4: MAIN PIPELINE
# =========================================================
def main():
    print("Fetching cases...")
    cases = fetch_cases()
    print("Total cases:", len(cases))

    doc_ids = [c["doc_id"] for c in cases]

    # Train-test split
    train_ids, test_ids = train_test_split(
        doc_ids,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_set = set(train_ids)
    test_set = set(test_ids)

    # ================= BASELINE =================
    # Most frequent legal refs
    counter = Counter()
    for c in cases:
        if c["doc_id"] in train_set:
            counter.update(c["legal_refs"])

    baseline = [x for x, _ in counter.most_common(TOP_K)]

    # Metrics storage
    g_p, g_r, g_f, g_h = [], [], [], []
    b_p, b_r, b_f, b_h = [], [], [], []

    print("\nStarting evaluation...\n")

    start_time = time.time()
    processed = 0
    total = len(test_set)

    # ================= EVALUATION LOOP =================
    for case in cases:
        if case["doc_id"] not in test_set:
            continue

        case_start = time.time()

        actual = case["legal_refs"]

        pred_g = predict_legal_refs(case["doc_id"], train_set, baseline)
        pred_b = baseline

        g = precision_recall_f1(pred_g, actual)
        b = precision_recall_f1(pred_b, actual)

        if g:
            p, r, f1, h = g
            g_p.append(p)
            g_r.append(r)
            g_f.append(f1)
            g_h.append(h)

        if b:
            p, r, f1, h = b
            b_p.append(p)
            b_r.append(r)
            b_f.append(f1)
            b_h.append(h)

        # ================= PROGRESS =================
        processed += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / processed
        remaining = avg_time * (total - processed)

        print(f"[{processed}/{total}] "
              f"{case['doc_id']} | "
              f"{round(time.time()-case_start,2)}s | "
              f"ETA: {round(remaining,1)}s")

        # Interim results
        if processed % 20 == 0:
            print("\n--- Interim F1:", round(mean(g_f), 4), "---\n")

    # ================= FINAL RESULTS =================
    print("\n=== KG (Strong Multi-Signal) ===")
    print("Precision:", round(mean(g_p), 4))
    print("Recall   :", round(mean(g_r), 4))
    print("F1       :", round(mean(g_f), 4))
    print("HitRate  :", round(mean(g_h), 4))

    print("\n=== Baseline ===")
    print("Precision:", round(mean(b_p), 4))
    print("Recall   :", round(mean(b_r), 4))
    print("F1       :", round(mean(b_f), 4))
    print("HitRate  :", round(mean(b_h), 4))

    print("\nTotal time:", round(time.time() - start_time, 2), "seconds")

    driver.close()


if __name__ == "__main__":
    main()