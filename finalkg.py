from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

def fetch_cases():
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)
            RETURN
                c.doc_id AS id,
                collect(DISTINCT t.name) AS topics,
                collect(DISTINCT lr.name) AS labels
        """)

        cases = []
        for r in result:
            if r["labels"]:
                cases.append({
                    "id": r["id"],
                    "topics": r["topics"] or [],
                    "labels": r["labels"] or []
                })

    return cases


# 🔥 THIS IS THE FIX LINE (YOU WERE MISSING THIS)
cases = fetch_cases()

print("Loaded cases:", len(cases))
import numpy as np
from collections import defaultdict, Counter
from neo4j import GraphDatabase
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# =========================================================
# CONNECT TO NEO4J
# =========================================================

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password123")
)

TOP_K = 5


# =========================================================
# STEP 1: LOAD CASE DATA FROM YOUR KG
# =========================================================

def fetch_cases():

    with driver.session() as session:

        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (c)-[:MENTIONS]->(lr:LegalReference)

            RETURN
                c.doc_id AS id,
                collect(DISTINCT t.name) AS topics,
                collect(DISTINCT lr.name) AS labels
        """)

        cases = []

        for r in result:

            if not r["labels"]:
                continue

            cases.append({
                "id": r["id"],
                "topics": set(r["topics"] or []),
                "labels": r["labels"]
            })

    return cases


cases = fetch_cases()
print("Cases loaded:", len(cases))


# =========================================================
# STEP 2: BUILD FAST LOOKUP STRUCTURES
# =========================================================

case_topics = {}
case_labels = {}
topic_index = defaultdict(set)

for c in cases:

    cid = c["id"]
    case_topics[cid] = c["topics"]
    case_labels[cid] = c["labels"]

    for t in c["topics"]:
        topic_index[t].add(cid)


# =========================================================
# STEP 3: TRAIN / TEST SPLIT
# =========================================================

ids = [c["id"] for c in cases]

train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=42)

train_set = set(train_ids)
test_set = set(test_ids)


# =========================================================
# STEP 4: FEATURE ENGINEERING (KG AWARE)
# =========================================================

def features(a, b):

    # topic overlap (VERY IMPORTANT)
    t1 = case_topics.get(a, set())
    t2 = case_topics.get(b, set())
    topic_overlap = len(t1 & t2)

    # label overlap (weak supervision signal)
    l1 = set(case_labels.get(a, []))
    l2 = set(case_labels.get(b, []))
    label_overlap = len(l1 & l2)

    # popularity (how often referenced)
    popularity = len(case_labels.get(b, []))

    return [
        topic_overlap,
        label_overlap,
        popularity
    ]


# =========================================================
# STEP 5: BUILD TRAINING DATASET
# =========================================================

def build_dataset():

    X, y = [], []

    train_list = list(train_set)

    print("Building dataset...")

    for c in cases:

        if c["id"] not in train_set:
            continue

        cid = c["id"]

        # --------------------------
        # POSITIVE SAMPLES
        # --------------------------
        positive = set()

        for t in case_topics[cid]:
            positive |= topic_index[t]

        positive = list(positive & train_set)

        for pos in positive[:5]:

            if pos == cid:
                continue

            X.append(features(cid, pos))
            y.append(1)

        # --------------------------
        # NEGATIVE SAMPLES
        # --------------------------
        for _ in range(3):

            neg = np.random.choice(train_list)

            if neg == cid:
                continue

            X.append(features(cid, neg))
            y.append(0)

    return np.array(X), np.array(y)


X, y = build_dataset()

print("Dataset shape:", X.shape)


# =========================================================
# STEP 6: TRAIN ML MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

model.fit(X, y)

print("Model trained")


# =========================================================
# STEP 7: PREDICTION FUNCTION
# =========================================================

def predict(case_id, fallback):

    scores = {}

    for c in train_set:

        f = features(case_id, c)
        score = model.predict_proba([f])[0][1]

        scores[c] = score

    label_scores = defaultdict(float)

    for c, sc in scores.items():
        for l in case_labels[c]:
            label_scores[l] += sc

    ranked = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked[:TOP_K]] or fallback


# =========================================================
# STEP 8: BASELINE
# =========================================================

counter = Counter()

for c in cases:
    if c["id"] in train_set:
        counter.update(case_labels[c["id"]])

baseline = [x[0] for x in counter.most_common(TOP_K)]


# =========================================================
# STEP 9: EVALUATION
# =========================================================

def evaluate():

    print("\n🚀 Evaluating FINAL SYSTEM...\n")

    p_list, r_list, f_list, h_list = [], [], [], []

    for c in cases:

        if c["id"] not in test_set:
            continue

        actual = set(c["labels"])

        pred = predict(c["id"], baseline)

        hits = len(set(pred) & actual)

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


evaluate()


# =========================================================
# CLOSE CONNECTION
# =========================================================

driver.close()