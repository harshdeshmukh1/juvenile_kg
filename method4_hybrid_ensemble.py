"""
================================================================================
METHOD 4: HYBRID GRAPH + ML ENSEMBLE (GRAPH RULE × XGBoost FUSION)
================================================================================
Title  : Knowledge Graph + ML-based Legal Case Retrieval and Legal Reference
         Prediction System
Method : Hybrid Ensemble — Linearly fuses Graph Rule score and ML score

Description:
    This method combines the strengths of Method 2 (Graph Rule) and Method 3
    (XGBoost ML) via a score-level fusion strategy.

    Motivation:
        - Graph Rule is interpretable and precise for strongly connected cases.
        - XGBoost captures non-linear feature interactions but can over-score
          popular labels for weakly connected queries.
        - Their fusion produces more robust, balanced predictions.

    Fusion formula (for each training case b):
        hybrid_score(b) = α × graph_score(b) + (1 - α) × ml_score(b)

    where:
        graph_score(b)  is the weighted-signal score from Method 2
        ml_score(b)     is P(similar=1 | features) from Method 3
        α               is a mixing coefficient in [0, 1] (default 0.5)

    Label ranking:
        hybrid_score is aggregated across training cases per label, then
        top-K labels are returned.

    Purpose in the paper:
        Demonstrates that ensemble / hybrid fusion further improves over either
        standalone method, validating the complementary nature of rule-based
        graph signals and data-driven ML ranking.

Evaluation Metrics:
    - Precision@K  : fraction of predicted labels that are correct
    - Recall@K     : fraction of true labels that are predicted
    - F1@K         : harmonic mean of Precision and Recall
    - Hit Rate@K   : 1 if at least one predicted label is correct, else 0

Usage:
    python method4_hybrid_ensemble.py

Requirements:
    pip install neo4j scikit-learn xgboost numpy
================================================================================
"""

import numpy as np
from collections import defaultdict, Counter
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NEO4J_URI       = "bolt://localhost:7687"
NEO4J_USER      = "neo4j"
NEO4J_PASSWORD  = "password123"

TOP_K           = 5     # Predictions per query case
TEST_SIZE       = 0.2   # Held-out test fraction
RANDOM_STATE    = 42    # Reproducibility

ALPHA           = 0.5   # Fusion coefficient: 0=pure ML, 1=pure Graph Rule

# Graph rule weights (from Method 2)
WEIGHT_TOPIC    = 3.0
WEIGHT_CITATION = 2.0
WEIGHT_COURT    = 0.5
MIN_CITATIONS   = 2

# XGBoost training parameters (from Method 3)
MAX_POSITIVES   = 5
NEG_PER_CASE    = 3
XGB_N_ESTIMATORS   = 300
XGB_MAX_DEPTH      = 6
XGB_LEARNING_RATE  = 0.05
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD CASES
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(driver: GraphDatabase.driver) -> list[dict]:
    """
    Load all cases from the Neo4j knowledge graph.

    Returns list of dicts:
        id        (str)  : document identifier
        topics    (set)  : topic names
        labels    (list) : legal reference names
        citations (set)  : cited case identifiers
        court     (str)  : court name
    """
    query = """
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)
        OPTIONAL MATCH (c)-[:CITES]->(cc:CitedCase)
        OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
        RETURN
            c.doc_id                   AS id,
            collect(DISTINCT t.name)   AS topics,
            collect(DISTINCT lr.name)  AS labels,
            collect(DISTINCT cc.name)  AS citations,
            court.name                 AS court
    """
    cases = []
    with driver.session() as session:
        for record in session.run(query):
            if not record["labels"]:
                continue
            cases.append({
                "id"       : record["id"],
                "topics"   : set(record["topics"]    or []),
                "labels"   : record["labels"],
                "citations": set(record["citations"] or []),
                "court"    : record["court"] or "",
            })

    print(f"[Data]  Loaded {len(cases)} cases.")
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD INDEXES
# ──────────────────────────────────────────────────────────────────────────────

def build_indexes(cases: list[dict]):
    """Build fast lookup structures (see Methods 2 & 3 for details)."""
    case_topics, case_labels, case_citations, case_court = {}, {}, {}, {}
    topic_index = defaultdict(set)

    for c in cases:
        cid = c["id"]
        case_topics[cid]    = c["topics"]
        case_labels[cid]    = c["labels"]
        case_citations[cid] = c["citations"]
        case_court[cid]     = c["court"]
        for t in c["topics"]:
            topic_index[t].add(cid)

    return case_topics, case_labels, case_citations, case_court, topic_index


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3a: GRAPH RULE SCORING (from Method 2)
# ──────────────────────────────────────────────────────────────────────────────

def graph_rule_score(
    query_id      : str,
    candidate_id  : str,
    case_topics   : dict,
    case_labels   : dict,
    case_citations: dict,
    case_court    : dict,
) -> float:
    """
    Compute the weighted graph similarity score between a query and a candidate.

    Score = WEIGHT_TOPIC * topic_overlap
          + WEIGHT_CITATION * citation_overlap (if >= MIN_CITATIONS)
          + WEIGHT_COURT * court_match
    """
    t1 = case_topics.get(query_id, set())
    t2 = case_topics.get(candidate_id, set())

    c1 = case_citations.get(query_id, set())
    c2 = case_citations.get(candidate_id, set())

    topic_overlap    = len(t1 & t2)
    citation_overlap = len(c1 & c2)
    citation_signal  = citation_overlap if citation_overlap >= MIN_CITATIONS else 0
    court_match      = 1 if (
        case_court.get(query_id)
        and case_court.get(query_id) == case_court.get(candidate_id)
    ) else 0

    return (
        WEIGHT_TOPIC    * topic_overlap
        + WEIGHT_CITATION * citation_signal
        + WEIGHT_COURT    * court_match
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3b: ML FEATURE EXTRACTION (from Method 3)
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    id_a: str,
    id_b: str,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
) -> list[float]:
    """
    Extract 7-dimensional KG feature vector for an (A, B) case pair.
    (Identical to Method 3.)
    """
    t_a = case_topics.get(id_a, set());    t_b = case_topics.get(id_b, set())
    c_a = case_citations.get(id_a, set()); c_b = case_citations.get(id_b, set())
    l_a = set(case_labels.get(id_a, [])); l_b = set(case_labels.get(id_b, []))

    t_inter = len(t_a & t_b); t_union = len(t_a | t_b)
    c_inter = len(c_a & c_b); c_union = len(c_a | c_b)

    return [
        float(t_inter),
        float(c_inter),
        1.0 if (case_court.get(id_a) and case_court.get(id_a) == case_court.get(id_b)) else 0.0,
        float(len(l_a & l_b)),
        float(len(l_b)),
        t_inter / t_union if t_union > 0 else 0.0,
        c_inter / c_union if c_union > 0 else 0.0,
    ]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: TRAIN ML MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_and_train(
    train_cases    : list[dict],
    train_set      : set,
    topic_index    : dict,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
) -> XGBClassifier:
    """
    Build pairwise training dataset and fit XGBoost model.
    (Identical logic to Method 3.)

    Returns:
        Trained XGBClassifier.
    """
    rng        = np.random.default_rng(RANDOM_STATE)
    train_list = list(train_set)
    X, y       = [], []

    print("[Train] Building pairwise dataset...")

    for case in train_cases:
        qid = case["id"]

        # Positive pairs: topic neighbors
        neighbors = set()
        for t in case_topics.get(qid, set()):
            neighbors |= topic_index[t]
        neighbors = list((neighbors & train_set) - {qid})

        for pos_id in neighbors[:MAX_POSITIVES]:
            X.append(extract_features(qid, pos_id, case_topics, case_labels, case_citations, case_court))
            y.append(1)

        # Negative pairs: no topic overlap
        neg_count = 0
        attempts  = 0
        while neg_count < NEG_PER_CASE and attempts < 50:
            neg_id = rng.choice(train_list)
            attempts += 1
            if neg_id == qid:
                continue
            if len(case_topics.get(qid, set()) & case_topics.get(neg_id, set())) == 0:
                X.append(extract_features(qid, neg_id, case_topics, case_labels, case_citations, case_court))
                y.append(0)
                neg_count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[Train] {X.shape[0]} pairs — Pos: {y.sum()} | Neg: {(y==0).sum()}")

    model = XGBClassifier(
        n_estimators      = XGB_N_ESTIMATORS,
        max_depth         = XGB_MAX_DEPTH,
        learning_rate     = XGB_LEARNING_RATE,
        subsample         = XGB_SUBSAMPLE,
        colsample_bytree  = XGB_COLSAMPLE,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
    )
    model.fit(X, y)
    print("[Model] XGBoost trained.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: HYBRID PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_hybrid(
    query_id       : str,
    train_set      : set,
    model          : XGBClassifier,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
    fallback       : list[str],
    top_k          : int,
    alpha          : float,
) -> list[str]:
    """
    Predict top-K legal references using the hybrid fusion of graph and ML scores.

    For each training case b:
        graph_s = graph_rule_score(query, b)   — normalised to [0, 1]
        ml_s    = P(similar=1 | features)       — already in [0, 1]
        hybrid  = alpha * graph_s + (1-alpha) * ml_s

    Label scores are accumulated over all training cases and the top-K
    labels are returned.

    Args:
        query_id  : ID of the query case
        train_set : set of training case IDs
        model     : trained XGBClassifier
        alpha     : mixing weight (0=pure ML, 1=pure graph rule)
        fallback  : global frequency labels for cold-start

    Returns:
        List of top-K predicted legal reference strings.
    """
    train_ids = list(train_set - {query_id})
    if not train_ids:
        return fallback[:top_k]

    # ── Compute raw graph rule scores ─────────────────────────────────────────
    g_scores_raw = np.array([
        graph_rule_score(
            query_id, cid,
            case_topics, case_labels, case_citations, case_court,
        )
        for cid in train_ids
    ], dtype=np.float64)

    # Normalise graph scores to [0, 1] so they are on same scale as ML proba
    g_max = g_scores_raw.max()
    g_scores = g_scores_raw / g_max if g_max > 0 else g_scores_raw

    # ── Compute ML similarity probabilities ───────────────────────────────────
    feature_matrix = np.array([
        extract_features(
            query_id, cid,
            case_topics, case_labels, case_citations, case_court,
        )
        for cid in train_ids
    ], dtype=np.float32)
    ml_scores = model.predict_proba(feature_matrix)[:, 1]

    # ── Fuse scores ───────────────────────────────────────────────────────────
    hybrid_scores = alpha * g_scores + (1 - alpha) * ml_scores

    # ── Aggregate label ranking ───────────────────────────────────────────────
    label_scores = defaultdict(float)
    for cid, h_score in zip(train_ids, hybrid_scores):
        for label in case_labels.get(cid, []):
            label_scores[label] += h_score

    if not label_scores:
        return fallback[:top_k]

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    return [label for label, _ in ranked[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    test_cases     : list[dict],
    train_set      : set,
    model          : XGBClassifier,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
    fallback       : list[str],
    top_k          : int,
    alpha          : float,
) -> dict:
    """
    Evaluate hybrid model on the test set. Returns mean metric dictionary.
    """
    precision_list, recall_list, f1_list, hit_list = [], [], [], []
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        qid    = case["id"]
        actual = set(case["labels"])

        predicted = predict_hybrid(
            qid, train_set, model,
            case_topics, case_labels, case_citations, case_court,
            fallback, top_k, alpha,
        )

        hits      = len(set(predicted) & actual)
        precision = hits / len(predicted) if predicted else 0.0
        recall    = hits / len(actual)    if actual    else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        hit       = 1 if hits > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        hit_list.append(hit)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] evaluated...")

    return {
        "Precision@K" : np.mean(precision_list),
        "Recall@K"    : np.mean(recall_list),
        "F1@K"        : np.mean(f1_list),
        "HitRate@K"   : np.mean(hit_list),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print(f"  METHOD 4 — HYBRID ENSEMBLE (alpha={ALPHA})")
    print("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    cases = load_cases(driver)
    train_cases, test_cases = train_test_split(
        cases, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Train: {len(train_cases)} | Test: {len(test_cases)}")

    train_set = {c["id"] for c in train_cases}

    case_topics, case_labels, case_citations, case_court, topic_index = \
        build_indexes(cases)

    counter = Counter()
    for c in train_cases:
        counter.update(c["labels"])
    fallback = [label for label, _ in counter.most_common(TOP_K)]

    model = build_dataset_and_train(
        train_cases, train_set, topic_index,
        case_topics, case_labels, case_citations, case_court,
    )

    print("\n[Eval]  Running evaluation on test cases...")
    results = evaluate(
        test_cases, train_set, model,
        case_topics, case_labels, case_citations, case_court,
        fallback, TOP_K, ALPHA,
    )

    print("\n" + "=" * 70)
    print(f"  RESULTS  (K = {TOP_K}, alpha = {ALPHA})")
    print("=" * 70)
    for metric, value in results.items():
        print(f"  {metric:<15} : {value:.4f}")
    print("=" * 70)

    driver.close()
    return results


if __name__ == "__main__":
    main()
