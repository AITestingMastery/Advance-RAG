# AI Mastery — Advanced RAG

A Streamlit app that lets you upload documents (**PDF, DOCX, CSV, TXT, MD**) and ask questions about them in plain English. It combines structured data extraction (so exact filters/counts/lookups are answered from verified records, not guessed) with hybrid vector+keyword search (so open-ended questions are answered from the most relevant passages) — and every answer is grounded strictly in the evidence retrieved, so the model isn't free to invent facts.

```
PDF / DOCX / CSV / TXT / MD
        ↓
Extract text + structured records
        ├──→ SQLite (structured records)
        └──→ ChromaDB (chunk embeddings) + BM25 keyword index
                        ↓
                  Query Router (LLM classifies the question)
                 /                              \
   Structured filter / count / lookup      Hybrid vector + BM25 (RRF fused)
                 \                              /
                        ↓
                  Grounded LLM answer
```

## What's new in this version

- **Live weather (v1.1)**: a fifth router strategy, `live_lookup`, answers "what's the weather in \<city\> right now?" style questions from a live API call — no document upload needed. See section 11 below.
- **Multiple file types**: upload PDF, DOCX, CSV, TXT, and Markdown files (previously PDF-only).
- **Multiple files at once**: the uploader accepts a batch of files in one go, in any mix of formats.
- **Multi-document questions fixed**: previously the app silently answered only from the single most-recently-uploaded file. You can now ask across *all* indexed documents at once, or pick one specific document to scope a question to.
- **More reliable structured data**: CSV rows and DOCX tables are parsed directly (no LLM involved, so values can never be misread or invented). PDFs and plain text still use LLM-based extraction, now batched so large files don't hit context limits or get a row cut in half.
- **Clearer errors**: unsupported file types, empty files, and scanned/image-only PDFs (which need OCR — not included) now fail with a specific, readable message instead of an unclear crash.

## 1. Prerequisites

- **Python 3.10 or newer.** Check with `python3 --version` (Mac/Linux) or `python --version` (Windows).
- **An OpenAI API key** with access to a chat model and an embeddings model. Get one at https://platform.openai.com/api-keys. This app calls the OpenAI API for structured-record extraction, query routing, embeddings, and answer generation — it will not work without a key with available quota.

## 2. Setup

### macOS / Linux

```bash
cd advanced-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
cd advanced-rag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (Command Prompt)

```cmd
cd advanced-rag
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

If `pip install` fails on a package, upgrade pip first: `python -m pip install --upgrade pip`, then retry.

## 3. Configure your API key

Copy the example environment file and edit it:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=sk-your-real-key-here
RAG_CHAT_MODEL=gpt-4o-mini
RAG_EMBED_MODEL=text-embedding-3-small
OPENWEATHER_API_KEY=your-openweathermap-key
```

- `RAG_CHAT_MODEL` — used for query routing, structured extraction, and answer generation. `gpt-4o-mini` is a good balance of cost/quality; swap in a stronger model (e.g. `gpt-4o`) if you need better reasoning on messy documents.
- `RAG_EMBED_MODEL` — used for chunk embeddings. Keep this consistent once you've indexed documents — switching embedding models after data is already indexed will make old and new vectors incompatible (see [Resetting data](#6-resetting--upgrading-data) below).
- `OPENWEATHER_API_KEY` — optional; only needed for live weather questions (the `live_lookup` strategy, see section 11 below). Everything else in this app works without it. Free tier at [openweathermap.org/api](https://openweathermap.org/api) is enough.

Never commit `.env` — it's already listed in `.gitignore`.

## 4. Run it

```bash
streamlit run app.py
```

This opens the app in your browser, usually at `http://localhost:8501`. The sidebar shows a green "OpenAI API key detected" message if `.env` is set up correctly.

## 5. Using the app

### Upload documents

1. Under **"1. Upload Knowledge Base"**, click the uploader and select one or more files. You can mix file types in a single batch — for example a CSV and a couple of PDFs at once.
2. Click **"🚀 Process Document(s)"**. Each file is processed independently; if one file fails (e.g. unsupported extension, empty file, scanned PDF with no extractable text), you'll get a specific error for that file while the others still succeed.
3. Each processed file shows how many chunks (for search) and structured records (for exact filtering/counting) were found. **Check this number** — if a document that should have structured records shows 0, the app will still answer questions about it, but only via general text search rather than exact filters/counts.

### Ask questions

1. Under **"2. Ask a Question"**, choose a **search scope**: `🌐 All documents` (default — searches everything you've uploaded) or a specific file (scopes both structured filtering and retrieval to just that one document).
2. Type a question, or pick one of the training examples from the dropdown.
3. Click **"💬 Ask"**. The answer panel shows:
   - The **strategy** the router chose (structured filtering, aggregation, exact lookup, or hybrid retrieval).
   - **Verified Records** — the exact structured rows used, when a structured strategy was used. This is your way to double-check the answer isn't hallucinated.
   - **Retrieved Evidence** — the actual text chunks used for hybrid (search-based) answers, each labeled with its source filename and its vector/BM25 rank, so you can trace exactly where the answer came from.

### Supported file types and what "good input" looks like

| Type | Works best when... |
|---|---|
| `.csv` | It has a header row and consistent columns. Delimiter is auto-detected (comma, semicolon, tab, or pipe). |
| `.docx` | Tabular data is in an actual Word table (not just tab-separated text) — those rows are extracted with 100% fidelity, no LLM guessing involved. |
| `.pdf` | It has real, selectable text (not a scan/photo). If you can highlight text in a PDF viewer, it will work here. |
| `.txt` / `.md` | Any readable text — notes, wikis, transcripts, documentation. |

## 6. Resetting / upgrading data

Indexed data lives in:
- `storage/rag.db` (SQLite — structured records + document list)
- `storage/chroma/` (ChromaDB — vector embeddings)

To start fresh (recommended after changing `RAG_EMBED_MODEL`, or if you're upgrading from an older version of this project with a different database schema), either:
- click **"🧹 Delete ALL documents"** in the sidebar, or
- stop the app and delete the `storage/rag.db` file and `storage/chroma/` folder — they'll be recreated automatically on next run.

## 7. Adding a new file type

The loader for each format lives in `advanced_rag.py` and follows one shared contract: given raw bytes and a filename, return `(text_for_search, unit_count, structured_records_or_None)`. To add support for a new format (e.g. `.xlsx`):

1. Write an `extract_xlsx(file_bytes)` function that returns `(text, records)` — text for the hybrid search index, and a list of row-dicts if the format is inherently tabular (pass records straight through, skipping the LLM, the way `.csv` and `.docx` tables do).
2. Add a branch for the new extension in `extract_content()`.
3. Add the extension to `SUPPORTED_EXTENSIONS` and to `requirements.txt` if a new library is needed.
4. Add an offline test case to `verify.py`.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Sidebar shows "OPENAI_API_KEY is missing" | `.env` wasn't created, is misnamed, or the key doesn't start with `sk-`. Restart Streamlit after editing `.env`. |
| "No readable text found in PDF" | The PDF is a scan/image with no real text layer. OCR isn't included in this version — try exporting the source as text first, or open an issue to add OCR support. |
| A structured question (e.g. "how many...") gives a wrong count | Check the **Verified Records** panel and the record count shown after upload — if the LLM's structured extraction missed rows, re-check the source formatting, or use CSV/DOCX-table format instead of PDF/free text for that data, since those are parsed without the LLM. |
| Answers mix data from documents you didn't mean to include | Set the **search scope** dropdown to a specific document instead of "🌐 All documents". |
| `pip install` errors on `chromadb` or similar | Make sure you're using Python 3.10+, and that your virtual environment is activated. On some systems you may need build tools (e.g. `xcode-select --install` on macOS, or the "Desktop development with C++" workload in Visual Studio Build Tools on Windows). |

## 9. Verifying your setup (no API key needed)

Run the offline smoke test to confirm every file loader works before you spend any API calls:

```bash
python verify.py
```

This exercises the PDF, DOCX, CSV, and TXT/MD loaders, chunking, and record normalization with synthetic in-memory files, and prints `All offline checks passed.` on success.

## 10. Evaluating the RAG pipeline

Two additional scripts evaluate the router + retrieval + generation pipeline itself, as opposed to `verify.py`'s offline unit tests of individual functions:

```bash
# Offline: proves the eval harness's scoring logic is correct, no API key needed
python3 test_evaluate_rag_offline.py

# Live: evaluates the real app against whatever's indexed in storage/rag.db
export OPENAI_API_KEY=sk-...
pip install ragas "langchain-community==0.3.0" datasets pandas
python3 evaluate_rag.py --out results/eval_report.csv
```

`evaluate_rag.py` needs a document already indexed (upload one via `streamlit run app.py` first, or call `advanced_rag.process_document()` directly) and measures four things, kept deliberately separate: **router accuracy** (did `classify_query()` pick the right strategy for each question, across all five strategies including `live_lookup`?), **filter/count correctness** (does `structured_filter`/`aggregation`/`exact_lookup`'s record count match an independent oracle computed from hand-specified filters, not the router's own output?), **live data (city) correctness** (does `live_lookup`'s resolved city match an independently hand-specified `expected_city` — needs `OPENWEATHER_API_KEY` set, same as any other question needs `OPENAI_API_KEY`), and **RAGAS generation quality** (faithfulness/answer relevancy/context precision/recall — computed only on questions that land on `hybrid_retrieval`, since the structured paths already force faithfulness by construction and `live_lookup`'s single verified API reading is already checked more precisely by city correctness). Add `--skip-ragas` to run only the first three, with no LLM-judge API calls.

`test_evaluate_rag_offline.py` mirrors `verify.py`'s approach — it seeds a scratch SQLite+Chroma store with fake records and monkeypatches the OpenAI-touching (and, for `live_lookup`, the weather-API-touching) functions with deterministic fakes, then proves `evaluate_rag.py`'s scoring logic (router-accuracy comparison, oracle-count comparison, city-correctness comparison, and the forced-hybrid isolation trick used to get RAGAS coverage on the hybrid path) is correct against the real `process_document`/`apply_filters`/`query_document` code, without needing an API key or real LLM/API calls.

## 11. Live data: asking about current weather (v1.1)

Alongside the four document-based strategies, the router also recognizes questions about real-world conditions that no uploaded document could ever answer — starting with current weather. Ask something like:

```
What is the weather in Hyderabad right now?
Current temperature in New York?
```

and the router picks a fifth strategy, `live_lookup`, which calls the OpenWeatherMap API directly (see `OPENWEATHER_API_KEY` in section 3) instead of retrieving anything from `storage/`. This works even if you haven't uploaded any document yet — live questions don't need an indexed corpus. If `OPENWEATHER_API_KEY` isn't set, or the city can't be resolved, you'll get a plain "insufficient evidence" style answer rather than a guess — see `IMPLEMENTATION.md`'s "Live data (v1.1)" section for the full design rationale (why this is never cached into Chroma/SQLite, and a real city-extraction bug it caught early).

Offline coverage: `python3 test_live_data_offline.py` (no API keys, no network calls — a fake `requests.get` stands in for the real weather API).

## Project structure

```
advanced-rag/
├── advanced_rag.py               # Core RAG logic: loaders, chunking, storage, routing, retrieval
├── live_data.py                   # v1.1: live weather lookup (the 5th, live_lookup strategy)
├── app.py                        # Streamlit UI
├── verify.py                     # Offline smoke test (no API key required)
├── evaluate_rag.py                # Live evaluation: router accuracy, count correctness, RAGAS
├── test_evaluate_rag_offline.py   # Offline self-test for evaluate_rag.py's scoring logic
├── test_live_data_offline.py      # Offline self-test for live_data.py + the live_lookup strategy
├── requirements.txt
├── .env.example
├── README.md            # This file — setup & usage
├── IMPLEMENTATION.md    # Architecture / design notes
├── results/              # Created by evaluate_rag.py: eval_report.csv
└── storage/             # Created automatically: rag.db + chroma/
```

## Training questions (sample employee dataset)

If you're testing with an employee-style dataset (like the sample `Employee_Details` file), these exercise every strategy the router supports:

1. List all employees in the Engineering department *(structured filter)*
2. List all employees in the Engineering department who are on leave *(multi-condition structured filter)*
3. Which employees work in Bengaluru and have the status "Active"? *(structured filter)*
4. How many employees are currently "On Leave"? *(aggregation/count)*
5. Tell me about EMP001 *(exact lookup)*
6. What's the company's leave policy? *(hybrid retrieval — falls back to vector+BM25 search if no such field exists in the structured records)*