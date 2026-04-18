"""
================================================================================
METHOD 2: GRAPH-BASED RULE RANKING FOR LEGAL REFERENCE PREDICTION
================================================================================
Title  : Knowledge Graph + ML-based Legal Case Retrieval and Legal Reference
         Prediction System
Method : Graph Rule Model — Weighted Signal Aggregation over Knowledge Graph

Description:
    This method uses the Neo4j knowledge graph structure directly.
    For a given query case, it finds similar cases using multiple graph-based
    signals, scores them using a hand-crafted weighted formula, aggregates
    their legal references, and returns the top-K ranked labels.

    Graph signals used:
        1. Topic Overlap     — cases sharing the same legal topics
        2. Citation Overlap  — cases sharing cited case references (>=2 required)
        3. Court Similarity  — weak boost when heard in the same court

    Compared to the Baseline (Method 1), this method IS case-specific:
    different query cases produce different ranked predictions.

    Purpose in the paper:
        Shows that graph structure alone (without ML) already outperforms the
        purely frequency-based baseline.

Evaluation Metrics:
    - Precision@K  : fraction of predicted labels that are correct
    - Recall@K     : fraction of true labels that are predicted
    - F1@K         : harmonic mean of Precision and Recall
    - Hit Rate@K   : 1 if at least one predicted label is correct, else 0

Usage:
    python method2_graph_rule_ranking.py

Requirements:
    pip install neo4j scikit-learn numpy
================================================================================
"""

import numpy as np
from collections import defaultdict, Counter
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

TOP_K          = 5     # Number of labels to predict per case
TEST_SIZE      = 0.2   # Fraction of cases held out for evaluation
RANDOM_STATE   = 42    # Reproducibility seed

# Weights for the scoring formula (tunable hyperparameters)
WEIGHT_TOPIC    = 3.0  # Topic overlap carries the most discriminating power
WEIGHT_CITATION = 2.0  # Shared citations indicate strong legal reasoning overlap
WEIGHT_COURT    = 0.5  # Same court is a weak but useful signal
MIN_CITATIONS   = 2    # Minimum shared citations to be treated as "related"


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING FROM NEO4J
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(driver: GraphDatabase.driver) -> list[dict]:
    """
    Load cases from Neo4j, including topics, labels, citations, and court.

    Returns a list of dictionaries:
        - id        (str)  : document identifier
        - topics    (set)  : set of associated topic names
        - labels    (list) : list of LegalReference names (ground truth)
        - citations (set)  : set of cited case identifiers
        - court     (str)  : court name (or empty string)
    """
    query = """
        MATCH (c:Case)
        OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
        OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)
        OPTIONAL MATCH (c)-[:CITES]->(cc:CitedCase)
        OPTIONAL MATCH (c)-[:HEARD_IN]->(court:Court)
        RETURN
            c.doc_id                    AS id,
            collect(DISTINCT t.name)    AS topics,
            collect(DISTINCT lr.name)   AS labels,
            collect(DISTINCT cc.name)   AS citations,
            court.name                  AS court
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

    print(f"[Data]  Loaded {len(cases)} cases with legal references.")
    return cases


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: BUILD INVERTED INDEXES FOR FAST LOOKUP
# ──────────────────────────────────────────────────────────────────────────────

def build_indexes(cases: list[dict]) -> tuple[dict, dict, dict, dict, dict]:
    """
    Build in-memory lookup structures from the case list for O(1) access.

    Returns:
        case_topics    : {case_id -> set of topic names}
        case_labels    : {case_id -> list of label strings}
        case_citations : {case_id -> set of cited case names}
        case_court     : {case_id -> court name string}
        topic_index    : {topic_name -> set of case_ids sharing that topic}
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
# STEP 3: SIMILARITY SCORING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def score_pair(
    query_id   : str,
    candidate_id: str,
    case_topics : dict,
    case_labels : dict,
    case_citations: dict,
    case_court  : dict,
) -> float:
    """
    Compute a weighted graph-based similarity score between two cases.

    Formula:
        score = w_topic    * |topics_A ∩ topics_B|
              + w_citation * |citations_A ∩ citations_B|  (only if >= MIN_CITATIONS)
              + w_court    * (1 if same court else 0)

    Args:
        query_id      : ID of the query case
        candidate_id  : ID of the candidate case
        (lookup dicts): prebuilt index structures

    Returns:
        Float score (higher = more similar / more relevant).
    """
    # Topic overlap
    t1 = case_topics.get(query_id, set())
    t2 = case_topics.get(candidate_id, set())
    topic_overlap = len(t1 & t2)

    # Citation overlap — only counted if strong enough (>=MIN_CITATIONS)
    c1 = case_citations.get(query_id, set())
    c2 = case_citations.get(candidate_id, set())
    citation_overlap = len(c1 & c2)
    citation_signal  = citation_overlap if citation_overlap >= MIN_CITATIONS else 0

    # Court match — binary weak boost
    court_match = 1 if (
        case_court.get(query_id)
        and case_court.get(query_id) == case_court.get(candidate_id)
    ) else 0

    score = (
        WEIGHT_TOPIC    * topic_overlap
        + WEIGHT_CITATION * citation_signal
        + WEIGHT_COURT    * court_match
    )
    return score


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: PREDICTION USING GRAPH RULE MODEL
# ──────────────────────────────────────────────────────────────────────────────

def predict_graph_rule(
    query_id    : str,
    train_set   : set,
    topic_index : dict,
    case_topics : dict,
    case_labels : dict,
    case_citations: dict,
    case_court  : dict,
    fallback    : list[str],
    top_k       : int,
) -> list[str]:
    """
    Predict top-K legal references for a given query case using graph signals.

    Algorithm:
        1. Use topic index to retrieve candidate cases (shared at least 1 topic).
        2. Score each candidate using the weighted formula.
        3. Aggregate legal references from candidates, weighted by their scores.
        4. Return the top-K labels by aggregated score.
        5. If no candidates found, fall back to global frequency baseline.

    Args:
        query_id    : ID of the query case
        train_set   : set of training case IDs (candidates must be in train)
        fallback    : list of globally frequent labels used when no graph neighbors found

    Returns:
        List of top-K predicted legal reference strings.
    """
    # Retrieve candidates that share at least one topic with query
    candidates = set()
    for topic in case_topics.get(query_id, set()):
        candidates |= topic_index[topic]

    # Keep only training cases; exclude the query case itself
    candidates = (candidates & train_set) - {query_id}

    if not candidates:
        return fallback[:top_k]

    # Score each candidate
    label_scores = defaultdict(float)
    for cid in candidates:
        score = score_pair(
            query_id, cid,
            case_topics, case_labels, case_citations, case_court,
        )
        if score <= 0:
            continue
        # Distribute score across that candidate's legal references
        for label in case_labels.get(cid, []):
            label_scores[label] += score

    if not label_scores:
        return fallback[:top_k]

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    return [label for label, _ in ranked[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5: EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    test_cases  : list[dict],
    train_set   : set,
    topic_index : dict,
    case_topics : dict,
    case_labels : dict,
    case_citations: dict,
    case_court  : dict,
    fallback    : list[str],
    top_k       : int,
) -> dict:
    """
    Evaluate the Graph Rule model on the test set.

    For each test case, generate a prediction and compare to ground truth.

    Returns:
        Dictionary of mean metric values.
    """
    precision_list, recall_list, f1_list, hit_list = [], [], [], []

    for case in test_cases:
        qid    = case["id"]
        actual = set(case["labels"])

        predicted = predict_graph_rule(
            qid, train_set, topic_index,
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
    print("  METHOD 2 — GRAPH RULE MODEL: Weighted KG Signal Aggregation")
    print("=" * 70)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Load and split
    cases = load_cases(driver)
    train_cases, test_cases = train_test_split(
        cases, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Train: {len(train_cases)} | Test: {len(test_cases)}")

    train_set = {c["id"] for c in train_cases}

    # Build indexes
    case_topics, case_labels, case_citations, case_court, topic_index = \
        build_indexes(cases)

    # Build frequency fallback from training set only
    counter = Counter()
    for c in train_cases:
        counter.update(c["labels"])
    fallback = [label for label, _ in counter.most_common(TOP_K)]

    # Evaluate
    results = evaluate(
        test_cases, train_set, topic_index,
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
