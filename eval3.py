
# =========================================================
# 🚀 FINAL ML + KNOWLEDGE GRAPH RANKING SYSTEM
# =========================================================
# Combines:
# 1. Graph structure (topics, citations, courts)
# 2. ML ranking (XGBoost)
# 3. Learned similarity instead of fixed weights
# =========================================================

import numpy as np
from collections import defaultdict, Counter
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# ASSUMPTION:
# You already have:
# cases
# case_topics
# case_citations
# case_labels
# case_court
# TOP_K
# =========================================================


# =========================================================
# STEP 1: BUILD TRAIN / TEST SPLIT
# =========================================================

print("🔵 Preparing dataset split...")

all_ids = [c["id"] for c in cases]

train_ids, test_ids = train_test_split(
    all_ids,
    test_size=0.2,
    random_state=42
)

train_set = set(train_ids)
test_set = set(test_ids)

print("Train:", len(train_set), "Test:", len(test_set))


# =========================================================
# STEP 2: FEATURE ENGINEERING
# =========================================================

def extract_features(case_a, case_b):

    # -------------------------
    # Topic overlap (semantic similarity)
    # -------------------------
    topics_a = case_topics.get(case_a, set())
    topics_b = case_topics.get(case_b, set())
    topic_overlap = len(topics_a & topics_b)

    # -------------------------
    # Citation overlap (legal structure similarity)
    # -------------------------
    citation_overlap = len(
        case_citations.get(case_a, set()) &
        case_citations.get(case_b, set())
    )

    # -------------------------
    # Court similarity (weak signal)
    # -------------------------
    court_match = 1 if case_b in case_court.get(case_a, set()) else 0

    # -------------------------
    # Popularity (how frequently a case is referenced)
    # -------------------------
    popularity = len(case_labels.get(case_b, []))

    return [
        topic_overlap,
        citation_overlap,
        court_match,
        popularity
    ]


# =========================================================
# STEP 3: BUILD TRAINING DATASET
# =========================================================

def build_dataset():

    X, y = [], []

    train_cases = [c["id"] for c in cases if c["id"] in train_set]

    print("🔵 Building training data...")

    for c in cases:

        if c["id"] not in train_set:
            continue

        case_id = c["id"]

        # =====================================================
        # POSITIVE SAMPLES (real similar cases via topics)
        # =====================================================
        positive_candidates = set()

        for t in case_topics.get(case_id, set()):
            positive_candidates |= set(topic_to_cases.get(t, []))

        positive_candidates = list(positive_candidates & set(train_cases))

        for pos in positive_candidates[:5]:

            X.append(extract_features(case_id, pos))
            y.append(1)

        # =====================================================
        # NEGATIVE SAMPLES (random incorrect cases)
        # =====================================================
        for _ in range(3):

            neg = np.random.choice(train_cases)

            if neg == case_id:
                continue

            X.append(extract_features(case_id, neg))
            y.append(0)

    return np.array(X), np.array(y)


# =========================================================
# STEP 4: TRAIN MODEL
# =========================================================

X, y = build_dataset()

print("Dataset shape:", X.shape)

print("🔵 Training XGBoost Ranker...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X, y)

print("✅ Model trained")


# =========================================================
# STEP 5: ML PREDICTION
# =========================================================

def predict(case_id, fallback):

    scores = {}

    for c in train_set:

        feat = extract_features(case_id, c)
        score = model.predict_proba([feat])[0][1]

        scores[c] = score

    # aggregate to labels
    label_scores = defaultdict(float)

    for c, sc in scores.items():
        for l in case_labels.get(c, []):
            label_scores[l] += sc

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:TOP_K]] or fallback


# =========================================================
# STEP 6: BASELINE (for comparison)
# =========================================================

def build_baseline():

    counter = Counter()

    for c in cases:
        if c["id"] in train_set:
            counter.update(case_labels[c["id"]])

    return [x[0] for x in counter.most_common(TOP_K)]


baseline = build_baseline()


# =========================================================
# STEP 7: EVALUATION
# =========================================================

def evaluate():

    print("\n🚀 Evaluating FINAL ML KG MODEL...\n")

    p_list, r_list, f_list, h_list = [], [], [], []

    for c in cases:

        if c["id"] not in test_set:
            continue

        actual = c["labels"]

        pred = predict(c["id"], baseline)

        hits = len(set(pred) & set(actual))

        precision = hits / len(pred) if pred else 0
        recall = hits / len(actual) if actual else 0
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        hit = 1 if hits > 0 else 0

        p_list.append(precision)
        r_list.append(recall)
        f_list.append(f1)
        h_list.append(hit)

    print("\n================ FINAL RESULTS ================\n")

    print("Precision:", np.mean(p_list))
    print("Recall   :", np.mean(r_list))
    print("F1       :", np.mean(f_list))
    print("HitRate  :", np.mean(h_list))


# =========================================================
# RUN
# =========================================================

evaluate()