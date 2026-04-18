"""
================================================================================
MASTER COMPARISON RUNNER — ALL METHODS
================================================================================
Title  : Knowledge Graph + ML-based Legal Case Retrieval and Legal Reference
         Prediction System

Description:
    This script runs all four methods back-to-back on the SAME train/test split
    and prints a side-by-side comparison table suitable for a research paper.

    Methods compared:
        1. Baseline       — Global Label Frequency (no graph, no ML)
        2. Graph Rule     — Weighted KG Signal Aggregation (no ML)
        3. XGBoost ML     — Pairwise ML ranking with KG features
        4. Hybrid         — Linear fusion of Graph Rule + XGBoost scores

    All methods use the same:
        - Neo4j knowledge graph
        - Train/test split (80/20, seed=42)
        - TOP_K = 5

Usage:
    python run_all_comparison.py

Requirements:
    pip install neo4j scikit-learn xgboost numpy tabulate
================================================================================
"""

import numpy as np
from collections import defaultdict, Counter
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ──────────────────────────────────────────────────────────────────────────────
# SHARED CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

TOP_K          = 5
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# Graph rule weights
WEIGHT_TOPIC    = 3.0
WEIGHT_CITATION = 2.0
WEIGHT_COURT    = 0.5
MIN_CITATIONS   = 2
ALPHA           = 0.5   # Hybrid fusion coefficient

# XGBoost params
MAX_POSITIVES        = 5
NEG_PER_CASE         = 3
XGB_N_ESTIMATORS     = 300
XGB_MAX_DEPTH        = 6
XGB_LEARNING_RATE    = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE        = 0.8


# ──────────────────────────────────────────────────────────────────────────────
# SHARED: DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_cases(driver):
    """Load all labelled cases from Neo4j KG (shared across all methods)."""
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
    print(f"[Data]  Loaded {len(cases)} cases from Neo4j.")
    return cases


def build_indexes(cases):
    """Build in-memory lookup structures (shared)."""
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


def compute_metrics(predicted, actual_set):
    """Compute P, R, F1, Hit for a single prediction."""
    hits      = len(set(predicted) & actual_set)
    precision = hits / len(predicted)   if predicted   else 0.0
    recall    = hits / len(actual_set)  if actual_set  else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    hit       = 1 if hits > 0 else 0
    return precision, recall, f1, hit


def aggregate_metrics(metric_lists):
    """Aggregate per-case metrics into means."""
    p, r, f, h = metric_lists
    return {
        "Precision@K" : np.mean(p),
        "Recall@K"    : np.mean(r),
        "F1@K"        : np.mean(f),
        "HitRate@K"   : np.mean(h),
    }


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 1: BASELINE FREQUENCY
# ──────────────────────────────────────────────────────────────────────────────

def run_method1(train_cases, test_cases, case_labels):
    """Global frequency baseline: predict the same top-K labels for all queries."""
    print("\n[Method 1] Running Baseline Frequency...")

    counter = Counter()
    for c in train_cases:
        counter.update(c["labels"])
    baseline = [label for label, _ in counter.most_common(TOP_K)]

    p_l, r_l, f_l, h_l = [], [], [], []
    for case in test_cases:
        p, r, f, h = compute_metrics(baseline, set(case["labels"]))
        p_l.append(p); r_l.append(r); f_l.append(f); h_l.append(h)

    return aggregate_metrics((p_l, r_l, f_l, h_l)), baseline


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 2: GRAPH RULE RANKING
# ──────────────────────────────────────────────────────────────────────────────

def _graph_score(qid, cid, case_topics, case_labels, case_citations, case_court):
    """Weighted graph similarity between two cases."""
    t1 = case_topics.get(qid, set());    t2 = case_topics.get(cid, set())
    c1 = case_citations.get(qid, set()); c2 = case_citations.get(cid, set())
    topic_ovl   = len(t1 & t2)
    cite_ovl    = len(c1 & c2)
    cite_signal = cite_ovl if cite_ovl >= MIN_CITATIONS else 0
    court_match = 1 if (case_court.get(qid) and case_court.get(qid) == case_court.get(cid)) else 0
    return WEIGHT_TOPIC * topic_ovl + WEIGHT_CITATION * cite_signal + WEIGHT_COURT * court_match


def _predict_graph(qid, train_set, topic_index, case_topics, case_labels,
                   case_citations, case_court, fallback):
    """Graph rule prediction for one query case."""
    candidates = set()
    for t in case_topics.get(qid, set()):
        candidates |= topic_index[t]
    candidates = (candidates & train_set) - {qid}
    if not candidates:
        return fallback[:TOP_K]

    label_scores = defaultdict(float)
    for cid in candidates:
        score = _graph_score(qid, cid, case_topics, case_labels, case_citations, case_court)
        if score > 0:
            for label in case_labels.get(cid, []):
                label_scores[label] += score

    if not label_scores:
        return fallback[:TOP_K]
    return [l for l, _ in sorted(label_scores.items(), key=lambda x: -x[1])[:TOP_K]]


def run_method2(train_cases, test_cases, train_set, topic_index,
                case_topics, case_labels, case_citations, case_court, fallback):
    """Graph rule model evaluation."""
    print("[Method 2] Running Graph Rule Ranking...")
    p_l, r_l, f_l, h_l = [], [], [], []
    for case in test_cases:
        pred = _predict_graph(
            case["id"], train_set, topic_index,
            case_topics, case_labels, case_citations, case_court, fallback,
        )
        p, r, f, h = compute_metrics(pred, set(case["labels"]))
        p_l.append(p); r_l.append(r); f_l.append(f); h_l.append(h)
    return aggregate_metrics((p_l, r_l, f_l, h_l))


# ──────────────────────────────────────────────────────────────────────────────
# ML FEATURE EXTRACTOR (shared by Methods 3 & 4)
# ──────────────────────────────────────────────────────────────────────────────

def _features(id_a, id_b, case_topics, case_labels, case_citations, case_court):
    t_a = case_topics.get(id_a, set());    t_b = case_topics.get(id_b, set())
    c_a = case_citations.get(id_a, set()); c_b = case_citations.get(id_b, set())
    l_a = set(case_labels.get(id_a, [])); l_b = set(case_labels.get(id_b, []))
    t_i = len(t_a & t_b); t_u = len(t_a | t_b)
    c_i = len(c_a & c_b); c_u = len(c_a | c_b)
    return [
        float(t_i),
        float(c_i),
        1.0 if (case_court.get(id_a) and case_court.get(id_a) == case_court.get(id_b)) else 0.0,
        float(len(l_a & l_b)),
        float(len(l_b)),
        t_i / t_u if t_u > 0 else 0.0,
        c_i / c_u if c_u > 0 else 0.0,
    ]


def train_xgboost(train_cases, train_set, topic_index,
                  case_topics, case_labels, case_citations, case_court):
    """Build training data and fit XGBoost."""
    rng = np.random.default_rng(RANDOM_STATE)
    train_list = list(train_set)
    X, y = [], []

    for case in train_cases:
        qid = case["id"]
        neighbors = set()
        for t in case_topics.get(qid, set()):
            neighbors |= topic_index[t]
        neighbors = list((neighbors & train_set) - {qid})
        for pos_id in neighbors[:MAX_POSITIVES]:
            X.append(_features(qid, pos_id, case_topics, case_labels, case_citations, case_court))
            y.append(1)
        neg_count = attempts = 0
        while neg_count < NEG_PER_CASE and attempts < 50:
            neg_id = rng.choice(train_list); attempts += 1
            if neg_id == qid: continue
            if len(case_topics.get(qid, set()) & case_topics.get(neg_id, set())) == 0:
                X.append(_features(qid, neg_id, case_topics, case_labels, case_citations, case_court))
                y.append(0); neg_count += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  Training XGBoost on {X.shape[0]} pairs...")

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE, use_label_encoder=False,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    print("  XGBoost trained.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 3: XGBOOST ML RANKING
# ──────────────────────────────────────────────────────────────────────────────

def _predict_ml(qid, train_set, model, case_topics, case_labels,
                case_citations, case_court, fallback):
    """XGBoost prediction for one query case."""
    train_ids = list(train_set - {qid})
    if not train_ids:
        return fallback[:TOP_K]
    F = np.array([_features(qid, cid, case_topics, case_labels, case_citations, case_court)
                  for cid in train_ids], dtype=np.float32)
    ml_scores = model.predict_proba(F)[:, 1]
    label_scores = defaultdict(float)
    for cid, sc in zip(train_ids, ml_scores):
        for label in case_labels.get(cid, []):
            label_scores[label] += sc
    if not label_scores:
        return fallback[:TOP_K]
    return [l for l, _ in sorted(label_scores.items(), key=lambda x: -x[1])[:TOP_K]]


def run_method3(test_cases, train_set, model, case_topics, case_labels,
                case_citations, case_court, fallback):
    """XGBoost ML ranking evaluation."""
    print("[Method 3] Running XGBoost ML Ranking...")
    p_l, r_l, f_l, h_l = [], [], [], []
    total = len(test_cases)
    for i, case in enumerate(test_cases):
        pred = _predict_ml(case["id"], train_set, model,
                           case_topics, case_labels, case_citations, case_court, fallback)
        p, r, f, h = compute_metrics(pred, set(case["labels"]))
        p_l.append(p); r_l.append(r); f_l.append(f); h_l.append(h)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}]")
    return aggregate_metrics((p_l, r_l, f_l, h_l))


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 4: HYBRID ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────────

def _predict_hybrid(qid, train_set, model, case_topics, case_labels,
                    case_citations, case_court, fallback, alpha):
    """Hybrid fusion prediction for one query case."""
    train_ids = list(train_set - {qid})
    if not train_ids:
        return fallback[:TOP_K]

    g_raw = np.array([_graph_score(qid, cid, case_topics, case_labels, case_citations, case_court)
                      for cid in train_ids], dtype=np.float64)
    g_max = g_raw.max()
    g_scores = g_raw / g_max if g_max > 0 else g_raw

    F = np.array([_features(qid, cid, case_topics, case_labels, case_citations, case_court)
                  for cid in train_ids], dtype=np.float32)
    ml_scores = model.predict_proba(F)[:, 1]

    hybrid = alpha * g_scores + (1 - alpha) * ml_scores

    label_scores = defaultdict(float)
    for cid, sc in zip(train_ids, hybrid):
        for label in case_labels.get(cid, []):
            label_scores[label] += sc
    if not label_scores:
        return fallback[:TOP_K]
    return [l for l, _ in sorted(label_scores.items(), key=lambda x: -x[1])[:TOP_K]]


def run_method4(test_cases, train_set, model, case_topics, case_labels,
                case_citations, case_court, fallback, alpha=ALPHA):
    """Hybrid ensemble evaluation."""
    print(f"[Method 4] Running Hybrid Ensemble (alpha={alpha})...")
    p_l, r_l, f_l, h_l = [], [], [], []
    total = len(test_cases)
    for i, case in enumerate(test_cases):
        pred = _predict_hybrid(case["id"], train_set, model, case_topics,
                               case_labels, case_citations, case_court, fallback, alpha)
        p, r, f, h = compute_metrics(pred, set(case["labels"]))
        p_l.append(p); r_l.append(r); f_l.append(f); h_l.append(h)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}]")
    return aggregate_metrics((p_l, r_l, f_l, h_l))


# ──────────────────────────────────────────────────────────────────────────────
# RESULTS PRINTING
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict):
    """
    Print a formatted comparison table of all methods' results.
    Suitable for copy-pasting into a research paper.
    """
    metrics = ["Precision@K", "Recall@K", "F1@K", "HitRate@K"]
    methods = list(all_results.keys())

    col_w = 16

    # Header
    print("\n" + "=" * (12 + col_w * len(methods)))
    print("  FINAL COMPARISON TABLE  (K = {})".format(TOP_K))
    print("=" * (12 + col_w * len(methods)))

    header = f"  {'Metric':<13}" + "".join(f"{m:<{col_w}}" for m in methods)
    print(header)
    print("-" * (12 + col_w * len(methods)))

    for metric in metrics:
        row = f"  {metric:<13}"
        for method in methods:
            val = all_results[method].get(metric, 0.0)
            row += f"{val:<{col_w}.4f}"
        print(row)

    print("=" * (12 + col_w * len(methods)))

    # Identify best value per row
    print("\n  ✓ = best value per metric")
    print("-" * (12 + col_w * len(methods)))
    for metric in metrics:
        vals = {m: all_results[m].get(metric, 0.0) for m in methods}
        best = max(vals, key=vals.get)
        row  = f"  {metric:<13}"
        for method in methods:
            marker = " ✓" if method == best else "  "
            row += f"{vals[method]:.4f}{marker:<{col_w - 6}}"
        print(row)
    print("=" * (12 + col_w * len(methods)))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LEGAL CASE RETRIEVAL & REFERENCE PREDICTION — ALL METHODS")
    print("=" * 70)

    # ── Setup ────────────────────────────────────────────────────────────────
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cases  = load_cases(driver)

    train_cases, test_cases = train_test_split(
        cases, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Train: {len(train_cases)} | Test: {len(test_cases)}")

    train_set = {c["id"] for c in train_cases}

    case_topics, case_labels, case_citations, case_court, topic_index = \
        build_indexes(cases)

    # Global frequency fallback built from training data only
    counter = Counter()
    for c in train_cases:
        counter.update(c["labels"])
    fallback = [label for label, _ in counter.most_common(TOP_K)]

    all_results = {}

    # ── Method 1: Baseline ────────────────────────────────────────────────────
    results1, _ = run_method1(train_cases, test_cases, case_labels)
    all_results["1-Baseline"] = results1

    # ── Method 2: Graph Rule ──────────────────────────────────────────────────
    results2 = run_method2(
        train_cases, test_cases, train_set, topic_index,
        case_topics, case_labels, case_citations, case_court, fallback,
    )
    all_results["2-GraphRule"] = results2

    # ── Train XGBoost (shared by Methods 3 & 4) ───────────────────────────────
    print("[ML]    Training XGBoost model...")
    model = train_xgboost(
        train_cases, train_set, topic_index,
        case_topics, case_labels, case_citations, case_court,
    )

    # ── Method 3: XGBoost ML ──────────────────────────────────────────────────
    results3 = run_method3(
        test_cases, train_set, model,
        case_topics, case_labels, case_citations, case_court, fallback,
    )
    all_results["3-XGBoostML"] = results3

    # ── Method 4: Hybrid ──────────────────────────────────────────────────────
    results4 = run_method4(
        test_cases, train_set, model,
        case_topics, case_labels, case_citations, case_court, fallback,
    )
    all_results["4-Hybrid"] = results4

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison_table(all_results)

    driver.close()
    return all_results


if __name__ == "__main__":
    main()
