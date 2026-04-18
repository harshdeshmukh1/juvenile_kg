"""
Indian Kanoon — Juvenile Justice Case Scraper
==============================================
Scrapes juvenile justice cases using the official Indian Kanoon API.

Output files:
  juvenile_cases_raw.json     — full structured data
  juvenile_cases_clean.csv    — flat CSV for analysis
  juvenile_cases_corpus.txt   — plain text for LLM training
  juvenile_cases_qa.json      — Q&A pairs for fine-tuning

Install:
  pip install requests

Run:
  python indiankanoon_scraper.py
"""

import requests
import json
import csv
import re
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# YOUR API TOKEN
# ─────────────────────────────────────────────────────────────────────────────

API_TOKEN = "6efe1ba5f3b5abb0f4ec43a5c08d19eef9fdf60e"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL  = "https://api.indiankanoon.org"
DELAY     = 2      # seconds between requests
MAX_PAGES = 10     # pages per query (10 results/page = 100 docs/query max)

QUERIES = [
    # Supreme Court focused
    "juvenile justice act 2015 doctypes:supremecourt",
    "juvenile justice act 2000 doctypes:supremecourt",
    "child in conflict with law doctypes:supremecourt",
    "juvenility age determination doctypes:supremecourt",
    "juvenile bail section 12 doctypes:supremecourt",
    "juvenile death sentence doctypes:supremecourt",
    "juvenile heinous offence trial as adult doctypes:supremecourt",
    "POCSO juvenile doctypes:supremecourt",
    "juvenile justice board doctypes:supremecourt",
    "reformative rehabilitation juvenile doctypes:supremecourt",

    # All courts (High Courts + SC)
    "juvenile justice act 2015 doctypes:judgments",
    "child in conflict with law bail doctypes:judgments",
    "juvenility claim age determination doctypes:judgments",
    "section 9 juvenile justice act doctypes:judgments",
    "section 15 juvenile justice heinous doctypes:judgments",
    "juvenile POCSO section 6 doctypes:judgments",
    "juvenile justice board preliminary assessment doctypes:judgments",
    "children court juvenile justice act doctypes:judgments",
    "child welfare committee care protection doctypes:judgments",
    "juvenile reformative detention special home doctypes:judgments",
    "juvenile sentence three years maximum doctypes:judgments",
    "juvenile tried as adult children court doctypes:judgments",
    "sheela barse juvenile children jail doctypes:judgments",
    "sampurna behura juvenile justice implementation doctypes:judgments",
    "pratap singh juvenility date offence doctypes:judgments",
    "hari ram juvenile justice retrospective doctypes:judgments",
    "abuzar hossain juvenility final disposal doctypes:judgments",
]

# ─────────────────────────────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_headers():
    return {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json",
    }


def api_search(query, pagenum=0):
    url = f"{BASE_URL}/search/"
    params = {"formInput": query, "pagenum": pagenum}
    try:
        r = requests.post(url, params=params, headers=get_headers(), timeout=20)
        if r.status_code == 403:
            print("  X 403 - token rejected. Check API_TOKEN.")
            return []
        if r.status_code == 402:
            print("  X 402 - credits exhausted. Top up at api.indiankanoon.org")
            return []
        r.raise_for_status()
        return r.json().get("docs", [])
    except Exception as e:
        print(f"  X Search error: {e}")
        return []


def api_doc(doc_id):
    url = f"{BASE_URL}/doc/{doc_id}/"
    params = {"maxcites": 30, "maxcitedby": 15}
    try:
        r = requests.post(url, params=params, headers=get_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  X Doc fetch error ({doc_id}): {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def html_to_text(html):
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.I)
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr|td)[^>]*>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", html)
    for entity, char in [
        ("&amp;","&"),("&lt;","<"),("&gt;",">"),("&nbsp;"," "),
        ("&quot;",'"'),("&#39;","'"),("&rsquo;","'"),("&lsquo;","'"),
        ("&mdash;","--"),("&ndash;","-"),("&hellip;","..."),
    ]:
        text = text.replace(entity, char)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_statutes(text):
    patterns = [
        r"Juvenile Justice(?:\s*\(Care and Protection[^)]*\))?\s*Act[,\s]*\d{4}",
        r"JJ\s*Act(?:[,\s]*\d{4})?",
        r"POCSO\s*Act",
        r"Protection of Children from Sexual Offences Act(?:[,\s]*\d{4})?",
        r"Indian Penal Code|IPC",
        r"Code of Criminal Procedure|CrPC",
        r"Article\s*(?:14|15|21|39|40|44|45|47)",
        r"Section\s*\d+(?:\(\d+\))?(?:\s*of\s*(?:the\s*)?(?:Juvenile|JJ|POCSO)[^,\.]*)?",
        r"Beijing Rules",
        r"UNCRC|UN Convention on the Rights of the Child",
        r"Children(?:'s)?\s*Act[,\s]*\d{4}",
        r"Constitution of India",
    ]
    found = set()
    for p in patterns:
        for m in re.finditer(p, text, re.I):
            val = re.sub(r"\s+", " ", m.group()).strip()
            if len(val) > 3:
                found.add(val)
    return sorted(found)


def extract_case_citations(text):
    pattern = re.compile(
        r"[A-Z][A-Za-z\s\.]{2,50}\s+v(?:s)?\.?\s+[A-Z][A-Za-z\s\.]{2,50}"
        r"(?:\s*\(\d{4}\))?(?:\s+\d+\s+SCC\s+\d+)?",
        re.MULTILINE
    )
    refs = set()
    for m in pattern.finditer(text):
        ref = re.sub(r"\s+", " ", m.group()).strip()
        if 8 < len(ref) < 150:
            refs.add(ref)
    return sorted(refs)


def classify_topics(text):
    topic_map = {
        "Bail":                   r"\bbail\b",
        "Age Determination":      r"age\s+determination|juvenility|juvenile.*age",
        "Retrospectivity":        r"retrospect",
        "Death Sentence":         r"death\s+(sentence|penalty)",
        "Rehabilitation":         r"rehabilitat|reintegrat|reform",
        "POCSO":                  r"POCSO|sexual offence.*child|child.*sexual",
        "Heinous Offence":        r"heinous\s+offence",
        "Trial as Adult":         r"tried\s+as\s+(an\s+)?adult|children.s\s+court",
        "JJB Procedure":          r"juvenile\s+justice\s+board|JJB",
        "Child Welfare":          r"child welfare committee|CWC|care and protection",
        "Sentencing":             r"\bsentenc\w+|\bconvict\w+|\bpunish\w+",
        "Constitutional Rights":  r"article\s+(?:14|15|21)|fundamental\s+right",
        "Adoption":               r"\badopt\w+",
        "Child Trafficking":      r"traffick",
        "Detention / Remand":     r"\bdetention\b|\bremand\b|\bobservation home\b",
        "Preliminary Assessment": r"preliminary\s+assessment",
    }
    tags = []
    for label, pattern in topic_map.items():
        if re.search(pattern, text, re.I):
            tags.append(label)
    return tags


def extract_bench(text):
    m = re.search(
        r"(?:Bench of\s+)?Justices?\s+([A-Z][A-Za-z\s\.,]+?)(?:\s+and\s+[A-Z][A-Za-z\s\.,]+)?[,\.]",
        text
    )
    return m.group().strip() if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_doc_ids(queries, max_pages):
    docs_meta = {}
    for query in queries:
        print(f"\n Searching: '{query}'")
        for page in range(max_pages):
            print(f"   Page {page + 1}...", end=" ", flush=True)
            docs = api_search(query, pagenum=page)
            if not docs:
                print("end of results.")
                break
            new = 0
            for doc in docs:
                tid = str(doc.get("tid", ""))
                if tid and tid not in docs_meta:
                    docs_meta[tid] = {
                        "tid":       tid,
                        "title":     doc.get("title", ""),
                        "docsource": doc.get("docsource", ""),
                        "docdate":   doc.get("docdate", ""),
                        "docsize":   doc.get("docsize", 0),
                        "headline":  doc.get("headline", ""),
                        "query":     query,
                    }
                    new += 1
            print(f"+{new} new  (total: {len(docs_meta)})")
            if len(docs) < 10:
                break
            time.sleep(DELAY)
        time.sleep(DELAY)
    return docs_meta


def fetch_full_judgment(meta):
    tid      = meta["tid"]
    doc_data = api_doc(tid)
    if not doc_data:
        return {}

    html      = doc_data.get("doc", "")
    full_text = html_to_text(html)

    statutes  = extract_statutes(full_text)
    case_refs = extract_case_citations(full_text)
    topics    = classify_topics(full_text)
    bench     = extract_bench(full_text)
    cites     = [c.get("title","") for c in doc_data.get("citeList",    []) if c.get("title")]
    cited_by  = [c.get("title","") for c in doc_data.get("citedbyList", []) if c.get("title")]

    return {
        "doc_id":             tid,
        "url":                f"https://indiankanoon.org/doc/{tid}/",
        "title":              meta.get("title", ""),
        "court":              meta.get("docsource", ""),
        "date":               meta.get("docdate", ""),
        "doc_size_chars":     meta.get("docsize", 0),
        "word_count":         len(full_text.split()),
        "search_query_used":  meta.get("query", ""),
        "headline":           meta.get("headline", ""),
        "full_text":          full_text,
        "topics":             topics,
        "statutes_mentioned": statutes,
        "case_citations":     case_refs,
        "bench":              bench,
        "cites":              cites,
        "cited_by":           cited_by,
        "scraped_at":         datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_json(records, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON saved  -> {path}  ({len(records)} records)")


def save_csv(records, path):
    if not records:
        return
    fields = [
        "doc_id","url","title","court","date","word_count",
        "topics","statutes_mentioned","bench","headline",
        "cites","cited_by","doc_size_chars","search_query_used","scraped_at",
    ]
    def flat(v):
        if isinstance(v, list):
            return " | ".join(str(x) for x in v)
        return str(v) if v is not None else ""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({k: flat(rec.get(k,"")) for k in fields})
    print(f"  CSV saved   -> {path}")


def save_corpus(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            sep = "=" * 80
            f.write(f"{sep}\n")
            f.write(f"TITLE:    {rec.get('title','')}\n")
            f.write(f"COURT:    {rec.get('court','')}\n")
            f.write(f"DATE:     {rec.get('date','')}\n")
            f.write(f"URL:      {rec.get('url','')}\n")
            f.write(f"BENCH:    {rec.get('bench','')}\n")
            f.write(f"TOPICS:   {', '.join(rec.get('topics',[]))}\n")
            f.write(f"STATUTES: {', '.join(rec.get('statutes_mentioned',[]))}\n")
            if rec.get("cites"):
                f.write(f"CITES:    {' | '.join(rec['cites'][:10])}\n")
            if rec.get("cited_by"):
                f.write(f"CITED BY: {' | '.join(rec['cited_by'][:10])}\n")
            f.write(f"\n{rec.get('full_text','')}\n\n")
    print(f"  Corpus saved -> {path}")


def save_qa_pairs(records, path):
    pairs = []
    for rec in records:
        title    = rec.get("title","")
        topics   = rec.get("topics",[])
        statutes = rec.get("statutes_mentioned",[])
        headline = rec.get("headline","")
        court    = rec.get("court","")
        date     = rec.get("date","")
        url      = rec.get("url","")

        if not title:
            continue

        if headline:
            pairs.append({
                "question": f"What is the case '{title}' about?",
                "answer":   headline,
                "source":   url,
            })
        if topics:
            pairs.append({
                "question": f"What legal topics does '{title}' cover?",
                "answer":   f"This judgment covers: {', '.join(topics)}.",
                "source":   url,
            })
        if statutes:
            pairs.append({
                "question": f"Which laws and statutes are mentioned in '{title}'?",
                "answer":   f"The following statutes are mentioned: {', '.join(statutes[:8])}.",
                "source":   url,
            })
        if court and date:
            pairs.append({
                "question": f"Which court decided '{title}' and when?",
                "answer":   f"It was decided by {court} on {date}.",
                "source":   url,
            })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  Q&A pairs   -> {path}  ({len(pairs)} pairs)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Indian Kanoon - Juvenile Justice Scraper")
    print(f"  Token: {API_TOKEN[:8]}...{API_TOKEN[-4:]}")
    print("=" * 60)

    # Step 1: collect doc IDs
    print("\nStep 1: Running search queries...")
    docs_meta = collect_all_doc_ids(QUERIES, max_pages=MAX_PAGES)
    print(f"\nFound {len(docs_meta)} unique documents.")

    # Step 2: fetch full texts
    print(f"\nStep 2: Fetching full judgment texts...")
    records = []
    failed  = []

    for i, (tid, meta) in enumerate(docs_meta.items(), 1):
        print(f"  [{i:>4}/{len(docs_meta)}] {meta.get('title','')[:65]}...")
        enriched = fetch_full_judgment(meta)
        if enriched and enriched.get("full_text"):
            records.append(enriched)
            wc = enriched.get("word_count", 0)
            t  = ", ".join(enriched.get("topics",[])[:3]) or "none"
            print(f"           OK {wc:,} words | {t}")
        else:
            failed.append(tid)
            print(f"           SKIP (empty or failed)")
        time.sleep(DELAY)

        # Checkpoint every 50 records
        if i % 50 == 0:
            cp = f"checkpoint_{i}.json"
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            print(f"  Checkpoint saved: {cp}")

    print(f"\nFetched: {len(records)} judgments  |  Failed: {len(failed)}")

    # Step 3: save outputs
    print("\nStep 3: Saving outputs...")
    save_json(records,    "juvenile_cases_raw.json")
    save_csv(records,     "juvenile_cases_clean.csv")
    save_corpus(records,  "juvenile_cases_corpus.txt")
    save_qa_pairs(records,"juvenile_cases_qa.json")

    # Step 4: stats
    all_topics = {}
    total_words = 0
    for r in records:
        for t in r.get("topics", []):
            all_topics[t] = all_topics.get(t, 0) + 1
        total_words += r.get("word_count", 0)

    print(f"\nStats:")
    print(f"  Total judgments : {len(records)}")
    print(f"  Total words     : {total_words:,}")
    print(f"  Avg words/doc   : {total_words // max(len(records),1):,}")
    print(f"\n  Topic breakdown:")
    for topic, count in sorted(all_topics.items(), key=lambda x: -x[1]):
        bar = "#" * (count * 30 // max(all_topics.values(), default=1))
        print(f"  {topic:<24} {bar} {count}")

    print("\nDone! Files created:")
    print("  juvenile_cases_raw.json     - full data with judgment text")
    print("  juvenile_cases_clean.csv    - flat table for analysis")
    print("  juvenile_cases_corpus.txt   - plain text for LLM training")
    print("  juvenile_cases_qa.json      - Q&A pairs for fine-tuning")


if __name__ == "__main__":
    main()
