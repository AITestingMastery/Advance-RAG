# Advanced RAG — Implementation Notes

## Pipeline

```
Upload (PDF / DOCX / CSV / TXT / MD)
  -> format-specific extraction (text + structured records where available)
  -> chunking (word-based, with overlap)
  -> structured record extraction (LLM for PDF/TXT/MD/DOCX-prose,
     programmatic parsing for CSV and DOCX tables)
  -> SQLite (documents + structured records)
  -> persistent ChromaDB vector store (chunk embeddings)
  -> query router
```

## File format handling

| Format | Text extraction | Structured records |
|---|---|---|
| `.pdf` | `pypdf` page-by-page text extraction | LLM extraction over the full text (batched for large files) |
| `.docx` | `python-docx` paragraphs + table cells | Tables are parsed directly (header row = field names, no LLM call). Prose-only DOCX falls back to LLM extraction. |
| `.csv` | Row values joined into readable lines | Parsed directly with `csv.DictReader` (delimiter auto-detected: comma, semicolon, tab, pipe). Never uses the LLM. |
| `.txt` / `.md` / `.markdown` | Raw decoded text | LLM extraction over the full text |

Text decoding tries `utf-8`, then `utf-8-sig`, then `latin-1`, so files exported from Excel or Windows tools don't break on encoding.

CSV and DOCX-table records are parsed programmatically rather than through the LLM: it is free, deterministic, and can't hallucinate a value that isn't actually in the row. PDFs and free-form text still need the LLM because their "table" is really just visually-aligned text, not real tabular structure.

## Large-document safety

`extract_records` (the LLM-based path) splits documents larger than `MAX_STRUCT_CHARS` (40,000 characters) into batches **on line boundaries only**, so a record can never be cut in half across two LLM calls. Each batch is extracted independently and the results are merged. A failure in one batch is skipped rather than aborting the whole upload.

## Multi-document querying

Earlier versions of this project only ever answered from the single most-recently-uploaded document — uploading a second file silently broke retrieval on the first. This is fixed:

- `query_document(question, document_id=None, ...)` with `document_id=None` (the default, and the "🌐 All documents" option in the UI) searches structured records and vector/BM25 chunks across **every** indexed document.
- Passing a specific `document_id` scopes both structured filtering and hybrid retrieval to that one file — useful once you've uploaded several unrelated datasets and want to ask about just one.
- Hybrid search results include the source `filename`, so multi-document answers are traceable back to the right file.

## Aggregation counts are never left to the LLM to recompute

Earlier behavior: for a count question (e.g. *"How many employees are on leave?"*), the code correctly filtered records in Python (`apply_filters`, `len(matched)`), but then handed the **entire list of matched records** to the LLM as JSON and asked it to write the final sentence. Even though the correct count was included in that same evidence blob, the model would sometimes recount from the enumerated list instead of trusting the given number — and get it wrong on a list as small as 23 items (observed: verified records = 23, LLM's answer said 20). LLMs are unreliable at counting items they've just enumerated in their own context; that's an attention/generation limitation, not a retrieval bug.

Fix: `generate_aggregation_answer()` treats `len(matched)` (computed in plain Python) as authoritative. The LLM is only asked to phrase that number into a sentence — it's shown the number as an already-verified fact plus a small 3-record sample for context, not the full matched list, removing the temptation to recount. As a second line of defense, the generated text is checked with a regex for the exact count; if the model's sentence doesn't contain that number, its prose is discarded and replaced with a deterministic sentence built directly from the verified count. `verify.py` includes a test that mocks a model which miscounts and confirms the safety net overrides it.

The **Verified Records** panel in the UI is always ground truth — it's the literal Python-filtered list, never LLM-generated — so it's the fastest way to sanity-check any structured answer.

## A dropped filter is never treated as "match everything"

A second, related failure: `apply_filters(records, filters)` iterates each record's *filter conditions* — if `filters` is `[]`, there are no conditions to fail, so **every** record passes. That's correct when a question genuinely has no condition (e.g. "how many employees are there in total"). But if the router classifies a question as `aggregation`/`structured_filter` and then — due to ordinary LLM variance — returns `filters: []` even though the question clearly names a value (e.g. *"How many employees are currently 'On Leave'?"*), the empty list silently means "count everyone," turning a correct 23 into an incorrect 100.

Fix, two layers:
1. `classify_query()`'s prompt now explicitly instructs the router that naming a specific value (a status, department, location, etc.) requires a filter, and that empty filters are only for genuinely unconditional questions.
2. `infer_filters_from_question()` is a router-independent safety net: it builds the actual set of distinct values seen in each field across the indexed records (a closed vocabulary — department names, statuses, locations, etc.) and scans the question text for any of those values. If the router came back with `structured_filter`/`aggregation` and an empty filter list, but the question contains a known value (e.g. "On Leave" appears in the `status` field's vocabulary), that filter is reconstructed automatically before matching runs. Confirmed with an end-to-end reproduction of the exact failure — router returns `filters: []` for the "On Leave" question — and the app now still returns 23, not 100.

This fallback is intentionally conservative: it only fires when the router's own filter list is empty, so it never overrides a filter the router did supply, and it produces no filter (leaving the question as a true unconditional count) when the question doesn't mention any known value.

## The router can't return a strategy outside its own enum anymore

A third failure, more revealing than the first two: for *"List all employees in Engineering department who are on leave"*, the router returned filters that were parsed perfectly — both `department=Engineering` and `status=On Leave` — but the `"strategy"` field itself came back as the literal string `"structured_filter | aggregation"`, i.e. the model echoed the prompt's own placeholder text (`"strategy":"structured_filter | aggregation | exact_lookup | hybrid_retrieval"`) instead of substituting one real value. That string matches none of the `if strategy == ...` branches in `query_document()`, so it silently fell through to the `hybrid_retrieval` default — discarding two perfectly good, correctly-parsed filters and answering from vector/BM25 search instead, which is why only one loosely-related employee came back instead of the actual filtered list.

Root cause: `classify_query()` used the loose `{"type": "json_object"}` response format, which only guarantees syntactically valid JSON — it enforces nothing about what a given field's *value* actually is. Writing the valid options into the prompt as `"structured_filter | aggregation | exact_lookup | hybrid_retrieval"` was itself the trap: nothing stopped the model from copying that text back verbatim.

Fix, three layers:
1. **Primary — OpenAI Structured Outputs.** `classify_query()` now calls the API with `response_format={"type": "json_schema", "json_schema": {...}, "strict": True}`, where `strategy` is a real JSON Schema `enum` of the four valid values and `filters[].operator` is constrained to `equals`/`contains`. The model is constrained at generation time — it cannot produce `"structured_filter | aggregation"` or any other value outside the enum, and stray extra keys (like the `"field": null, "operator": null, "value": null` seen alongside the real filters array) are rejected by `"additionalProperties": false`. This eliminates the failure at its source rather than patching around it.
2. **Fallback for older models/accounts.** If the Structured Outputs call errors (e.g. an account or model without support for it), `classify_query()` falls back to the original `json_object` mode.
3. **Defense in depth.** `_normalize_strategy()` coerces any value outside the enum into a real strategy (checking `structured_filter` before `aggregation`, so a garbled `"structured_filter | aggregation"` resolves to the listing behavior a "list all ... who ..." question actually wants) — this covers the fallback path and any future model quirk. Separately, in `query_document()`, if the strategy still ends up as `hybrid_retrieval` but the router did supply filters (which the prompt only ever populates for structured strategies), the filters are trusted over the label rather than discarded.

`verify.py` reproduces the exact malformed response from the incident (garbled strategy string, stray null keys, correctly-formed filters) and confirms it now resolves to `structured_filter` with both filters intact.

## `exact_lookup` was ignoring the router's filters entirely

A fourth failure, this one inherited unmodified from the original project (none of the earlier fixes touched this function): *"Who is the manager of Vikram Malhotra?"* was routed correctly — `strategy: exact_lookup`, `filters: [{"field":"name","operator":"equals","value":"Vikram Malhotra"}]` — but came back with 0 records and "the evidence provided is empty."

The old `exact_lookup(question, records)` never looked at `filters` at all:

```python
def exact_lookup(question, records):
    q = question.lower()
    return [r for r in records if any(q in str(v).lower() for v in r.values() if v is not None)]
```

It checked whether the **entire question string** was a substring of a single field's value. That only ever matches if the whole question *is* the value (e.g. typing just `"EMP001"`), never a natural question — `"who is the manager of vikram malhotra?"` is never a substring of a `"Vikram Malhotra"` field value, so this returned nothing regardless of how well the router classified the question.

Fix: `exact_lookup(question, records, filters=None)` now applies the router's filters first via `apply_filters()` — the same reliable, field-targeted mechanism `structured_filter` and `aggregation` already use — and only falls back to the old whole-question substring heuristic when no filters were supplied. `query_document()` now also passes `filters` through to `exact_lookup()` (previously it was hardcoded to `[]` in the returned result, so the UI couldn't even show what was applied) and includes `exact_lookup` in the dropped-filter recovery pass alongside `structured_filter`/`aggregation`. Verified end-to-end: the exact "Vikram Malhotra" scenario now returns the correct record instead of an empty result.

## Query strategies (unchanged)

Structured questions:
- filtering
- multi-condition filtering
- aggregation/count
- exact lookup

Unstructured questions:
- vector search
- BM25
- Reciprocal Rank Fusion (RRF)
- grounded LLM generation

## Evaluation harness (evaluate_rag.py)

Separate from `verify.py`'s offline unit tests of individual functions, `evaluate_rag.py` evaluates the real, live pipeline end to end — the router's actual classification decisions and the actual generated answers, not synthetic fixtures. It deliberately keeps three things separate rather than one blended score:

1. **Router accuracy** — does `classify_query()` pick the strategy a question actually calls for? Scored against a hand-labeled `expected_strategy` per ground-truth question.
2. **Filter/count correctness** — for the three structured strategies, does `result["record_count"]` match an *independent* oracle count? The oracle calls `apply_filters()` too (no reason to reinvent exact filtering, already unit-tested in `verify.py`), but with hand-specified filters, never the router's own output — this isolates "did the LLM extract the right filters from natural language" from "does `apply_filters()` work."
3. **RAGAS generation quality (hybrid_retrieval only)** — faithfulness, answer relevancy, context precision/recall, scored only on questions that land on the `hybrid_retrieval` path. The structured paths already force faithfulness by construction (exact filtered evidence, and the aggregation safety net above that discards the LLM's prose outright if it doesn't restate the verified count) — RAGAS on those paths would mostly be re-measuring a guarantee that's already there. Because some semantic/document-level questions may get correctly routed to a structured path anyway (structured filtering can sometimes answer a "compare X and Y" question well enough), `run_forced_hybrid_question()` temporarily monkeypatches `classify_query` to force `hybrid_retrieval` for those specific questions, runs the real retrieval + generation code through it, and restores the router immediately after — so RAGAS gets real coverage of the hybrid path even when the router's own (arguably correct) choice would have avoided it.

### Two real bugs in ragas 0.4.3 found and worked around, not in this app's own code

Running RAGAS against a fresh `pip install ragas` surfaced two upstream dependency bugs, both from the same cause: ragas's newer default resolution paths aren't drop-in compatible with its older metric implementations.

- **Import crash.** `pip install ragas` alone resolves `langchain-community` 0.4.2, which no longer ships the `langchain_community.chat_models.vertexai` module that ragas 0.4.3 still imports unconditionally at package load — `import ragas` crashes with `ModuleNotFoundError` in a completely clean environment. Fix: pin `langchain-community==0.3.0` alongside it (see `requirements.txt`), which lets pip resolve a compatible `langchain-core`/`langchain-openai` automatically.
- **`answer_relevancy` returns `nan`.** That metric calls `self.embeddings.embed_query(...)` internally, a method that only exists on ragas's legacy `LangchainEmbeddingsWrapper`. Left to its own defaults, `evaluate()` resolves embeddings through a newer, separate `ragas.embeddings.openai_provider.OpenAIEmbeddings` class that has no `embed_query` at all, raising `AttributeError` on every call and silently scoring the metric `nan`. Fix: `evaluate_rag.py` builds a `LangchainEmbeddingsWrapper(OpenAIEmbeddings(...))` explicitly and passes it as `evaluate(..., embeddings=...)`.
- **Silent 1-generation instead of 3.** `answer_relevancy`'s strictness setting asks for 3 reverse-engineered questions per answer to average similarity against the original question (`n=3`). With no explicit `llm=`, `evaluate()` resolves its LLM through `llm_factory()`, which returns an `instructor`-library adapter built for structured JSON output, not multi-sample generation — it always returns exactly 1 generation regardless of what `n` was, degrading the metric to a 1-sample estimate with no error raised (just a log warning: *"LLM returned 1 generations instead of requested 3"*), and adding a large per-call latency penalty (~47s/item observed vs. ~5s/item after the fix) from the structured-output validation loop running once per call instead of one plain completion. Fix: same pattern — build a `LangchainLLMWrapper(ChatOpenAI(...))` explicitly and pass it as `evaluate(..., llm=...)`; that wrapper reliably returns exactly `n` generations, either natively or via `n` sequential calls.

Both fixes construct legacy ragas classes on purpose, which triggers `DeprecationWarning`s on every run; `evaluate_rag.py` filters `DeprecationWarning` globally since this script has no other third-party deprecation surface and the warnings are for a path it's intentionally choosing, not accidentally hitting.

### Offline self-test (test_evaluate_rag_offline.py)

Same philosophy as `verify.py`: seed a scratch SQLite+Chroma store (never the real `storage/rag.db`) with a handful of fake employee records, monkeypatch `embed`/`classify_query`/`generate_answer`/`generate_aggregation_answer` with deterministic fakes, and prove `evaluate_rag.py`'s *scoring logic* — not the real LLM's accuracy — is correct against the real `process_document`/`apply_filters`/`query_document` code paths. Covers `structured_filter`, `aggregation`, `exact_lookup`, and `run_forced_hybrid_question()`'s monkeypatch-and-restore behavior (verifying both that forcing hybrid actually overrides the fake router, and that `classify_query` is genuinely restored afterward rather than leaking into later questions). This test needs no `OPENAI_API_KEY` and makes no network calls; it does not tell you whether the real LLM router is accurate, only that the harness measures it correctly once pointed at a live run.

## Known limitations / good next steps

- Scanned/image-only PDFs raise a clear error rather than silently returning nothing — OCR (e.g. `pytesseract` + `pdf2image`) is not wired in yet.
- No reranker model (a cross-encoder rerank stage over the RRF-fused top results would improve precision further).
- Structured extraction from PDF/TXT/MD/DOCX-prose still costs one or more LLM calls per upload; very large corpora would benefit from caching or a cheaper extraction model.
- `.xlsx` / `.pptx` / `.json` are not yet supported — see the README for how to add a new loader.
- The router misroutes some genuinely open-ended/comparative questions ("give a high-level overview...", "compare X and Y...") to a structured strategy instead of `hybrid_retrieval` — observed via `evaluate_rag.py` at 83% router accuracy across 12 test questions. Worth revisiting `_classify_query_prompt()`'s guidance for broad "overview"/"compare"/"summarize" phrasing, which currently has no equivalent to the strong "list"/"count"/"who is" signals it already handles well.