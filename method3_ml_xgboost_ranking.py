"""
================================================================================
METHOD 3: ML RANKING MODEL (XGBoost) WITH KNOWLEDGE GRAPH FEATURES
================================================================================
Title  : Knowledge Graph + ML-based Legal Case Retrieval and Legal Reference
         Prediction System
Method : ML Ranking Model — XGBoost trained on KG-derived pairwise features

Description:
    This method replaces the hand-crafted rule weights of Method 2 with a
    trained XGBoost binary classifier. The model learns — from data — how
    important each graph signal is for determining whether two cases are similar.

    Training paradigm (Learning-to-Rank via pairwise binary classification):
        - POSITIVE pair  : (query, candidate) where they share ≥1 topic → label 1
        - NEGATIVE pair  : (query, random case with no shared topics)   → label 0
        - The classifier predicts P(pair is similar), used as a relevance score.

    Feature vector for a (case_A, case_B) pair:
        [0] topic_overlap      — |topics_A ∩ topics_B|
        [1] citation_overlap   — |citations_A ∩ citations_B|
        [2] court_match        — 1 if same court, else 0
        [3] label_overlap      — |labels_A ∩ labels_B| (weak supervision)
        [4] label_popularity_B — number of labels case_B has (size signal)
        [5] jaccard_topics     — |topics_A ∩ topics_B| / |topics_A ∪ topics_B|
        [6] jaccard_citations  — Jaccard over citation sets

    Prediction:
        For a query case, score each training case using predict_proba, then
        aggregate scores across training cases to rank all legal references.

    Purpose in the paper:
        Demonstrates that learned feature weights outperform fixed rule weights,
        validating the research contribution of combining KG structure with ML.

Evaluation Metrics:
    - Precision@K  : fraction of predicted labels that are correct
    - Recall@K     : fraction of true labels that are predicted
    - F1@K         : harmonic mean of Precision and Recall
    - Hit Rate@K   : 1 if at least one predicted label is correct, else 0

Usage:
    python method3_ml_xgboost_ranking.py

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

TOP_K           = 5     # Number of labels to predict per query case
TEST_SIZE       = 0.2   # Fraction of data held out for evaluation
RANDOM_STATE    = 42    # Reproducibility

MAX_POSITIVES   = 5     # Max positive samples drawn per query case
NEG_PER_CASE    = 3     # Number of negative samples per query case

# XGBoost hyperparameters
XGB_N_ESTIMATORS    = 300
XGB_MAX_DEPTH       = 6
XGB_LEARNING_RATE   = 0.05
XGB_SUBSAMPLE       = 0.8
XGB_COLSAMPLE       = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD CASES FROM NEO4J
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(driver: GraphDatabase.driver) -> list[dict]:
    """
    Query Neo4j knowledge graph and return structured case data.

    Returns list of dicts with keys:
        id        (str)  : unique document identifier
        topics    (set)  : topic names linked via HAS_TOPIC edges
        labels    (list) : LegalReference names (prediction targets)
        citations (set)  : cited case identifiers via CITES edges
        court     (str)  : court name via HEARD_IN edge
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
# STEP 2: BUILD LOOKUP INDEXES
# ──────────────────────────────────────────────────────────────────────────────

def build_indexes(cases: list[dict]):
    """
    Construct fast in-memory lookups from case list.

    Returns:
        case_topics    : {id -> set[str]}
        case_labels    : {id -> list[str]}
        case_citations : {id -> set[str]}
        case_court     : {id -> str}
        topic_index    : {topic -> set[case_id]}
    """
    case_topics, case_labels, case_citations, case_court = {}, {}, {}, {}
    topic_index = defaultdict(set)

    for c in cases:
        cid = c["id"]
        case_topics[cid]    = c["topics"]
        case_labels[cid]    = c["labels"]
        case_citations[cid] = c["citations"]
        case_court[cid]     = c["court"]

        for topic in c["topics"]:
            topic_index[topic].add(cid)

    return case_topics, case_labels, case_citations, case_court, topic_index


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
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
    Compute a 7-dimensional feature vector for a (case_A, case_B) pair.

    Features:
        [0] topic_overlap      : raw count of shared topics
        [1] citation_overlap   : raw count of shared cited cases
        [2] court_match        : 1.0 if heard in same court else 0.0
        [3] label_overlap      : raw count of shared legal references
        [4] label_popularity_B : number of labels case_B has (global importance)
        [5] jaccard_topics     : Jaccard similarity over topic sets
        [6] jaccard_citations  : Jaccard similarity over citation sets

    Args:
        id_a, id_b : case identifiers
        (dicts)    : prebuilt lookup structures

    Returns:
        List of 7 float feature values.
    """
    t_a = case_topics.get(id_a, set())
    t_b = case_topics.get(id_b, set())

    c_a = case_citations.get(id_a, set())
    c_b = case_citations.get(id_b, set())

    l_a = set(case_labels.get(id_a, []))
    l_b = set(case_labels.get(id_b, []))

    topic_inter     = len(t_a & t_b)
    topic_union     = len(t_a | t_b)
    citation_inter  = len(c_a & c_b)
    citation_union  = len(c_a | c_b)

    f0_topic_overlap      = float(topic_inter)
    f1_citation_overlap   = float(citation_inter)
    f2_court_match        = 1.0 if (
        case_court.get(id_a) and case_court.get(id_a) == case_court.get(id_b)
    ) else 0.0
    f3_label_overlap      = float(len(l_a & l_b))
    f4_label_popularity_b = float(len(l_b))
    f5_jaccard_topics     = topic_inter    / topic_union    if topic_union    > 0 else 0.0
    f6_jaccard_citations  = citation_inter / citation_union if citation_union > 0 else 0.0

    return [
        f0_topic_overlap,
        f1_citation_overlap,
        f2_court_match,
        f3_label_overlap,
        f4_label_popularity_b,
        f5_jaccard_topics,
        f6_jaccard_citations,
    ]

FEATURE_NAMES = [
    "topic_overlap",
    "citation_overlap",
    "court_match",
    "label_overlap",
    "label_popularity_B",
    "jaccard_topics",
    "jaccard_citations",
]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: BUILD PAIRWISE TRAINING DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_training_dataset(
    train_cases    : list[dict],
    train_set      : set,
    topic_index    : dict,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairwise (case_A, case_B) training examples with binary labels.

    Positive examples (label=1):
        case_B shares at least one topic with case_A.
        Up to MAX_POSITIVES per query.

    Negative examples (label=0):
        case_B is a random training case with NO shared topics.
        NEG_PER_CASE negatives per query.

    Returns:
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,) with 0/1 labels
    """
    rng     = np.random.default_rng(RANDOM_STATE)
    train_list = list(train_set)

    X, y = [], []
    print("[Train] Building pairwise dataset...")

    for case in train_cases:
        qid = case["id"]

        # ── Positive sampling: cases sharing a topic ──────────────────────────
        topic_neighbors = set()
        for topic in case_topics.get(qid, set()):
            topic_neighbors |= topic_index[topic]
        topic_neighbors = list((topic_neighbors & train_set) - {qid})

        for pos_id in topic_neighbors[:MAX_POSITIVES]:
            feats = extract_features(
                qid, pos_id,
                case_topics, case_labels, case_citations, case_court,
            )
            X.append(feats)
            y.append(1)

        # ── Negative sampling: random cases ───────────────────────────────────
        neg_count = 0
        attempts  = 0
        while neg_count < NEG_PER_CASE and attempts < 50:
            neg_id = rng.choice(train_list)
            attempts += 1
            if neg_id == qid:
                continue
            # Prefer negatives with zero topic overlap for clean signal
            if len(case_topics.get(qid, set()) & case_topics.get(neg_id, set())) == 0:
                feats = extract_features(
                    qid, neg_id,
                    case_topics, case_labels, case_citations, case_court,
                )
                X.append(feats)
                y.append(0)
                neg_count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[Train] Dataset: {X.shape[0]} pairs | Positives: {y.sum()} | Negatives: {(y==0).sum()}")
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: TRAIN XGBOOST MODEL
# ──────────────────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost binary classifier to predict pairwise case similarity.

    The model outputs P(similar=1 | features), used as a relevance score.

    Args:
        X : feature matrix (n_pairs × 7)
        y : binary labels (1=similar, 0=dissimilar)

    Returns:
        Trained XGBClassifier instance.
    """
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

    # Print feature importance for interpretability
    importances = model.feature_importances_
    print("\n[Model] Feature Importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"         {name:<25} : {imp:.4f}")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6: PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_ml(
    query_id       : str,
    train_set      : set,
    model          : XGBClassifier,
    case_topics    : dict,
    case_labels    : dict,
    case_citations : dict,
    case_court     : dict,
    fallback       : list[str],
    top_k          : int,
) -> list[str]:
    """
    Predict the top-K legal references for a query case using the ML model.

    Algorithm:
        1. For each training case, extract the (query, train_case) feature vector.
        2. Use model.predict_proba to get a similarity score in [0, 1].
        3. Accumulate scores across training cases to build a label ranking.
        4. Return top-K labels by accumulated score.
        5. Fallback to global frequency baseline if nothing scored.

    Args:
        query_id    : ID of the query case
        train_set   : set of training case IDs
        model       : trained XGBClassifier
        fallback    : global frequency baseline labels

    Returns:
        List of top-K predicted legal reference strings.
    """
    # Build feature matrix for (query, every_train_case)
    train_ids = list(train_set - {query_id})
    if not train_ids:
        return fallback[:top_k]

    feature_matrix = np.array([
        extract_features(
            query_id, cid,
            case_topics, case_labels, case_citations, case_court,
        )
        for cid in train_ids
    ], dtype=np.float32)

    # P(similar=1) for each training case
    similarity_scores = model.predict_proba(feature_matrix)[:, 1]

    # Aggregate label scores across training cases
    label_scores = defaultdict(float)
    for cid, score in zip(train_ids, similarity_scores):
        for label in case_labels.get(cid, []):
            label_scores[label] += score

    if not label_scores:
        return fallback[:top_k]

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    return [label for label, _ in ranked[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7: EVALUATION
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
) -> dict:
    """
    Evaluate the ML model on the held-out test set.

    Returns:
        Dictionary with mean Precision@K, Recall@K, F1@K, HitRate@K.
    """
    precision_list, recall_list, f1_list, hit_list = [], [], [], []
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        qid    = case["id"]
        actual = set(case["labels"])

        predicted = predict_ml(
            qid, train_set, model,
            case_topics, case_labels, case_citations, case_court,
            fallback, top_k,
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

        # Progress indicator for large test sets
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
    print("  METHOD 3 — ML RANKING: XGBoost on KG-derived Pairwise Features")
    print("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Load data
    cases = load_cases(driver)

    # Train / test split
    train_cases, test_cases = train_test_split(
        cases, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Train: {len(train_cases)} | Test: {len(test_cases)}")

    train_set = {c["id"] for c in train_cases}

    # Build indexes
    case_topics, case_labels, case_citations, case_court, topic_index = \
        build_indexes(cases)

    # Global frequency fallback (from training set only)
    counter = Counter()
    for c in train_cases:
        counter.update(c["labels"])
    fallback = [label for label, _ in counter.most_common(TOP_K)]

    # Build pairwise training dataset
    X, y = build_training_dataset(
        train_cases, train_set, topic_index,
        case_topics, case_labels, case_citations, case_court,
    )

    # Train XGBoost classifier
    model = train_model(X, y)

    # Evaluate on test set
    print("\n[Eval]  Running evaluation on test cases...")
    results = evaluate(
        test_cases, train_set, model,
        case_topics, case_labels, case_citations, case_court,
        fallback, TOP_K,
    )

    # Print results
    print("\n" + "=" * 70)
    print(f"  RESULTS  (K = {TOP_K})")
    print("=" * 70)
    for metric, value in results.items():
        print(f"  {metric:<15} : {value:.4f}")
    print("=" * 70)

    driver.close()
    return results


if __name__ == "__main__":
    main()
