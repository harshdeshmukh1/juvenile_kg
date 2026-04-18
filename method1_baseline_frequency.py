"""
================================================================================
METHOD 1: BASELINE FREQUENCY-BASED LEGAL REFERENCE PREDICTION
================================================================================
Title  : Knowledge Graph + ML-based Legal Case Retrieval and Legal Reference
         Prediction System
Method : Baseline — Global Label Frequency (No Graph, No ML)

Description:
    This is the simplest possible baseline. It predicts the same top-K most
    frequently occurring legal references (sections / statutes) across ALL
    training cases for EVERY query case. It does NOT use any case-specific
    features, graph structure, or similarity computation.

    Purpose in the paper:
        Establishes a lower-bound benchmark. If our Graph/ML methods cannot
        beat this, they are not useful.

Evaluation Metrics:
    - Precision@K  : fraction of predicted labels that are correct
    - Recall@K     : fraction of true labels that are predicted
    - F1@K         : harmonic mean of Precision and Recall
    - Hit Rate@K   : 1 if at least one predicted label is correct, else 0

Usage:
    python method1_baseline_frequency.py

Requirements:
    pip install neo4j scikit-learn numpy
================================================================================
"""

import numpy as np
from collections import Counter
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

TOP_K          = 5    # Number of labels to predict per case
TEST_SIZE      = 0.2  # Fraction of cases used for evaluation
RANDOM_STATE   = 42   # For reproducibility


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING FROM NEO4J KNOWLEDGE GRAPH
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(driver: GraphDatabase.driver) -> list[dict]:
    """
    Query the Neo4j knowledge graph and return a list of case dictionaries.

    Each dictionary has:
        - id     (str)  : unique document identifier
        - topics (set)  : set of topic names associated with the case
        - labels (list) : list of legal reference names (sections / statutes)

    Only cases that have at least one legal reference label are kept,
    because label-less cases cannot contribute to training or evaluation.
    """
    query = """
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)
        RETURN
            c.doc_id                   AS id,
            collect(DISTINCT t.name)   AS topics,
            collect(DISTINCT lr.name)  AS labels
    """
    cases = []
    with driver.session() as session:
        for record in session.run(query):
            if not record["labels"]:
                continue
            cases.append({
                "id"    : record["id"],
                "topics": set(record["topics"] or []),
                "labels": record["labels"],
            })

    print(f"[Data]  Loaded {len(cases)} cases with legal references.")
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def split_cases(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Perform a reproducible stratified split of cases into training and test sets.
    Returns (train_cases, test_cases).
    """
    train_cases, test_cases = train_test_split(
        cases,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
    )
    print(f"[Split] Train: {len(train_cases)} | Test: {len(test_cases)}")
    return train_cases, test_cases


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: BUILD BASELINE — GLOBAL FREQUENCY RANKING
# ──────────────────────────────────────────────────────────────────────────────

def build_frequency_baseline(train_cases: list[dict], top_k: int) -> list[str]:
    """
    Count how often each legal reference label appears across all training cases
    and return the top-K most frequent labels.

    This list is then predicted identically for every test case.

    Args:
        train_cases : list of training case dictionaries
        top_k       : number of labels to return

    Returns:
        List of the top-K most common legal reference strings.
    """
    counter = Counter()
    for case in train_cases:
        counter.update(case["labels"])

    baseline_labels = [label for label, _ in counter.most_common(top_k)]
    print(f"[Baseline] Top-{top_k} global labels: {baseline_labels}")
    return baseline_labels


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(predictions: list[str], test_cases: list[dict]) -> dict:
    """
    Evaluate the prediction system on the test set.

    For each test case, we compare the predicted label list against the
    ground-truth labels and compute four metrics.

    Metrics:
        Precision@K  = |predicted ∩ actual| / K
        Recall@K     = |predicted ∩ actual| / |actual|
        F1@K         = 2 * P * R / (P + R)
        Hit Rate@K   = 1 if |predicted ∩ actual| >= 1 else 0

    Args:
        predictions : list of predicted label strings (same for all cases here)
        test_cases  : list of test case dictionaries

    Returns:
        Dictionary with mean values of all four metrics.
    """
    pred_set = set(predictions)
    precision_list, recall_list, f1_list, hit_list = [], [], [], []

    for case in test_cases:
        actual = set(case["labels"])
        hits   = len(pred_set & actual)

        precision = hits / len(predictions) if predictions else 0.0
        recall    = hits / len(actual)      if actual      else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        hit       = 1 if hits > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        hit_list.append(hit)

    results = {
        "Precision@K" : np.mean(precision_list),
        "Recall@K"    : np.mean(recall_list),
        "F1@K"        : np.mean(f1_list),
        "HitRate@K"   : np.mean(hit_list),
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  METHOD 1 — BASELINE: Global Label Frequency")
    print("=" * 70)

    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Load cases from the knowledge graph
    cases = load_cases(driver)

    # Split into train / test
    train_cases, test_cases = split_cases(cases)

    # Build frequency baseline using training set only
    baseline_predictions = build_frequency_baseline(train_cases, top_k=TOP_K)

    # Evaluate
    results = evaluate(baseline_predictions, test_cases)

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
