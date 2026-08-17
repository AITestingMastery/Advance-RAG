"""
Offline self-test for evaluate_rag.py -- proves the eval harness's SCORING
LOGIC (router-accuracy comparison, oracle-count comparison, per-question
report shape, and the forced-hybrid isolation trick) is correct, without
calling the OpenAI API. Mirrors verify.py's own approach: monkeypatch the
LLM-touching functions with deterministic fakes, exercise the real app code
paths (process_document, apply_filters, query_document's routing logic,
hybrid_search's BM25+vector fusion) on real data through them.

This does NOT validate whether the real LLM router is accurate -- only a
live run with OPENAI_API_KEY set can tell you that. It validates that IF the
router behaves as instructed, evaluate_rag.py measures and reports that
correctly. Run with: python3 test_evaluate_rag_offline.py

Covers, end to end against the real advanced_rag.py code:
  - structured_filter (single + AND + OR filters)
  - aggregation
  - exact_lookup (added -- evaluate_rag.py's Q5 ground truth uses this)
  - run_forced_hybrid_question()'s monkeypatch-and-restore (added -- this is
    the function evaluate_rag.py uses to isolate the Hybrid RAG path for
    RAGAS even when the router correctly picks a structured strategy; a bug
    here would leave every later question silently stuck on the forced
    classifier)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Use a scratch storage dir so this never touches a real rag.db/chroma.
tmp_storage = Path(tempfile.mkdtemp())
os.environ["OPENAI_API_KEY"] = "sk-test-fake-not-used"

import advanced_rag as rag  # noqa: E402

# Redirect storage to the scratch dir (module-level constants already ran at
# import time, so patch the derived paths + collection directly).
rag.STORAGE_DIR = tmp_storage
rag.SQLITE_PATH = tmp_storage / "rag.db"
rag.CHROMA_PATH = tmp_storage / "chroma"
import chromadb  # noqa: E402
rag._chroma = chromadb.PersistentClient(path=str(rag.CHROMA_PATH))
rag._collection = rag._chroma.get_or_create_collection(name="rag_chunks", metadata={"hnsw:space": "cosine"})
rag.init_db()

# --- Fake embeddings: no network call, deterministic, cheap -----------------
import numpy as np  # noqa: E402


def _fake_embed(texts):
    # Distinct-but-deterministic vectors so hybrid_search's math doesn't
    # divide by zero. Content matters a little more now that the forced-
    # hybrid test below actually reads hybrid_search results, but exact
    # semantic meaning still doesn't -- BM25 alone is enough to make the
    # fused ranking non-empty and deterministic.
    rng = np.random.default_rng(42)
    return np.array([rng.normal(size=16) + hash(t) % 7 for t in texts], dtype=np.float32)


rag.embed = _fake_embed

# --- Fake router: deterministic keyword-based classification ----------------
# Simulates "the LLM router behaves exactly as instructed" so this test
# isolates evaluate_rag.py's scoring logic from real LLM variance.
_original_classify_query = rag.classify_query


def _fake_classify_query(question, records):
    q = question.lower()
    fields = sorted({k for r in records for k in r.keys()})
    if "how many" in q:
        strategy = "aggregation"
    elif "who is" in q or "manager of" in q or "record for" in q:
        strategy = "exact_lookup"
    else:
        strategy = "structured_filter"
    filters = rag.infer_filters_from_question(question, records)
    return {"strategy": strategy, "operation": "count" if strategy == "aggregation" else None,
            "filters": filters, "reason": "fake router for offline test"}


rag.classify_query = _fake_classify_query
rag.generate_answer = lambda question, evidence, strategy: f"[fake answer for {strategy}] {len(evidence)} item(s)."
rag.generate_aggregation_answer = lambda question, count, filters, matched: f"There are {count} matching record(s)."

# --- Seed a small CSV dataset mirroring the real workshop data's shape ------
# emp_id added so the exact_lookup test below (mirroring evaluate_rag.py's
# real Q5: "Show the employee record for EMP057.") has something to match.
csv_bytes = (
    b"emp_id,name,department,status,location,salary_inr\n"
    b"EMP001,Asha Rao,Engineering,Active,Bengaluru,70000\n"
    b"EMP002,Ravi Kumar,Engineering,On Leave,Bengaluru,65000\n"
    b"EMP003,Meera Iyer,Engineering,Active,Bengaluru,80000\n"
    b"EMP004,Kevin Davis,Engineering,Active,Mumbai,55000\n"
    b"EMP005,Sunita Rao,Finance,Active,Bengaluru,72000\n"
    b"EMP006,Jennifer Mehta,IT,On Leave,Mumbai,50000\n"
)
info = rag.process_document(csv_bytes, "fake_employees.csv")
assert info["records"] == 6, f"expected 6 seeded records, got {info}"
print(f"OK: seeded {info['records']} fake records into scratch storage")

# --- Now import and exercise evaluate_rag.py's actual functions -------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_rag as ev  # noqa: E402

all_records = ev.load_all_records()
assert len(all_records) == 6
print(f"OK: evaluate_rag.load_all_records() -> {len(all_records)} records")

# oracle_count must match plain-Python filtering independent of apply_filters
# internals -- sanity check against a hand count.
eng_count = ev.oracle_count(all_records, [{"field": "department", "operator": "equals", "value": "Engineering"}])
assert eng_count == 4, f"oracle_count wrong: {eng_count}"
print(f"OK: oracle_count (Engineering) -> {eng_count}")

or_count = ev.oracle_count(all_records, [{"field": "department", "operator": "in", "value": ["Finance", "IT"]}])
assert or_count == 2, f"oracle_count OR-filter wrong: {or_count}"
print(f"OK: oracle_count (Finance or IT) -> {or_count}")

# --- Run one full question through query_document() with the fakes in place -
result = rag.query_document("List all employees in the Engineering department", document_id=None, top_k=5)
assert result["strategy"] == "Structured filtering", f"unexpected strategy: {result['strategy']}"
assert result["record_count"] == 4, f"unexpected record_count: {result['record_count']}"
print(f"OK: query_document() end-to-end with fakes -> strategy={result['strategy']}, "
      f"record_count={result['record_count']}")

agg_result = rag.query_document('How many employees are currently "On Leave"?', document_id=None, top_k=5)
assert agg_result["strategy"] == "Structured aggregation", f"unexpected strategy: {agg_result['strategy']}"
assert agg_result["record_count"] == 2, f"unexpected record_count: {agg_result['record_count']}"
assert "2" in agg_result["answer"], f"aggregation answer missing count: {agg_result['answer']}"
print(f"OK: aggregation path end-to-end -> record_count={agg_result['record_count']}, "
      f"answer={agg_result['answer']!r}")

# --- Exact lookup path (mirrors evaluate_rag.py's real Q5 ground truth) -----
lookup_result = rag.query_document("Show the employee record for EMP002.", document_id=None, top_k=5)
assert lookup_result["strategy"] == "Exact structured lookup", f"unexpected strategy: {lookup_result['strategy']}"
assert lookup_result["record_count"] == 1, f"unexpected record_count: {lookup_result['record_count']}"
lookup_expected = ev.oracle_count(all_records, [{"field": "emp_id", "operator": "equals", "value": "EMP002"}])
assert lookup_expected == 1, f"oracle_count for emp_id lookup wrong: {lookup_expected}"
print(f"OK: exact_lookup path end-to-end -> record_count={lookup_result['record_count']}, "
      f"matches oracle_count={lookup_expected}")

# --- Confirm evaluate_rag.py's per-question scoring row shape is correct ---
gt_row = {"id": "TEST", "expected_strategy": "structured_filter",
          "oracle_filters": [{"field": "department", "operator": "equals", "value": "Engineering"}]}
actual_strategy = ev.STRATEGY_LABEL_TO_KEY.get(result["strategy"], "unknown")
assert actual_strategy == "structured_filter"
expected_count = ev.oracle_count(all_records, gt_row["oracle_filters"])
router_correct = actual_strategy == gt_row["expected_strategy"]
count_correct = result.get("record_count", 0) == expected_count
assert router_correct and count_correct
print("OK: evaluate_rag.py's scoring logic (router_correct / count_correct) computes correctly "
      "end-to-end against real process_document + query_document code paths")

# --- run_forced_hybrid_question(): the monkeypatch-and-restore isolation ----
# This is the function evaluate_rag.py uses so RAGAS can score the Hybrid RAG
# path even on a question the real router correctly sends somewhere else.
# Deliberately reuse the aggregation question above: the FAKE router would
# normally route it to "aggregation" (as just proven), so if forcing hybrid
# actually works, this call must come back "Hybrid retrieval" instead --
# and rag.classify_query must be exactly _fake_classify_query again
# immediately afterward, proving the try/finally swap-back isn't leaky.
forced = ev.run_forced_hybrid_question('How many employees are currently "On Leave"?', top_k=5)
assert forced["strategy"] == "Hybrid retrieval", f"forcing hybrid failed: {forced['strategy']}"
assert forced.get("sources"), "forced hybrid path returned no retrieved chunks -- BM25/vector fusion broke"
assert rag.classify_query is _fake_classify_query, (
    "classify_query was not restored after run_forced_hybrid_question() -- "
    "every subsequent question in a real eval run would silently be forced "
    "to hybrid_retrieval too"
)
print(f"OK: run_forced_hybrid_question() -> forced strategy={forced['strategy']}, "
      f"{len(forced['sources'])} chunk(s) retrieved, classify_query correctly restored afterward")

# Prove the restore isn't just cosmetic: the SAME question through the normal
# (non-forced) path must go back to routing exactly as the fake router says.
recheck = rag.query_document('How many employees are currently "On Leave"?', document_id=None, top_k=5)
assert recheck["strategy"] == "Structured aggregation", (
    f"router state leaked after forced-hybrid call -- got {recheck['strategy']} "
    "instead of the normal aggregation route"
)
print("OK: normal routing behaves identically before and after a forced-hybrid call "
      "-- no leaked monkeypatch state")

# --- Cleanup -----------------------------------------------------------------
rag.classify_query = _original_classify_query
shutil.rmtree(tmp_storage, ignore_errors=True)
print("\nAll offline self-tests passed -- evaluate_rag.py's scoring logic, including "
      "exact_lookup and the forced-hybrid isolation trick, is verified against the real "
      "advanced_rag.py code paths, using fakes only for the OpenAI-touching calls.")