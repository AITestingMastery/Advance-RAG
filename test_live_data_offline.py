"""
Offline self-test for live_data.py and advanced_rag.py's live_lookup
strategy -- proves the weather-lookup logic is correct WITHOUT calling the
real OpenWeatherMap API or the real OpenAI API. Mirrors the same approach as
test_evaluate_rag_offline.py and verify.py: monkeypatch the network-touching
calls with deterministic fakes, exercise the real code paths through them.

This does NOT validate that the real router reliably classifies weather
questions as live_lookup, or that the real OpenWeatherMap API behaves as
expected -- only a live run with both OPENAI_API_KEY and
OPENWEATHER_API_KEY set can tell you that. It validates that IF the router
picks live_lookup and IF the weather API responds, query_document() wires
everything together correctly -- and, just as importantly, that a weather
question works even when NO document has ever been uploaded, which was a
real structural bug this test would have caught: query_document() used to
raise RuntimeError("No documents are indexed.") before the router ever ran.

Run with: python3 test_live_data_offline.py
"""
import os
import shutil
import tempfile
from pathlib import Path

tmp_storage = Path(tempfile.mkdtemp())
os.environ["OPENAI_API_KEY"] = "sk-test-fake-not-used"

import advanced_rag as rag  # noqa: E402
import live_data  # noqa: E402

rag.STORAGE_DIR = tmp_storage
rag.SQLITE_PATH = tmp_storage / "rag.db"
rag.CHROMA_PATH = tmp_storage / "chroma"
import chromadb  # noqa: E402
rag._chroma = chromadb.PersistentClient(path=str(rag.CHROMA_PATH))
rag._collection = rag._chroma.get_or_create_collection(name="rag_chunks", metadata={"hnsw:space": "cosine"})
rag.init_db()

rag.generate_answer = lambda question, evidence, strategy: (
    f"[fake answer for {strategy}] error={evidence['error']}" if isinstance(evidence, dict) and "error" in evidence
    else f"[fake answer for {strategy}] {evidence}"
)

# =============================================================================
# 1. extract_city() -- best-effort location extraction
# =============================================================================
assert live_data.extract_city("What is the weather in Hyderabad?") == "Hyderabad"
assert live_data.extract_city("current temperature in New York") == "New York"
assert live_data.extract_city("What's the weather like in London?") == "London"
assert live_data.extract_city("What is the weather in Hyderabad right now?") == "Hyderabad", (
    "regression: 'right now' trailing filler must not be swallowed into the city name"
)
assert live_data.extract_city("current temperature in Mumbai today") == "Mumbai"
assert live_data.extract_city("tell me a joke") is None
print("OK: extract_city() parses 'in/for/at <city>' phrasing and returns None when there's nothing to extract")

# =============================================================================
# 2. get_current_weather() -- success path, with a fake requests.get so no
#    real network call happens and no real API key is needed
# =============================================================================
_call_count = {"n": 0}


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _fake_requests_get(url, params=None, timeout=None):
    _call_count["n"] += 1
    return _FakeResponse(200, {
        "name": params["q"],
        "main": {"temp": 31.5, "feels_like": 34.0, "humidity": 60},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 2.5},  # m/s -> ~9.0 kph
        "dt": 1234567890,
    })


live_data.OPENWEATHER_API_KEY = "fake-key-for-test"
live_data.requests.get = _fake_requests_get
live_data._cache.clear()

result = live_data.get_current_weather("Hyderabad")
assert "error" not in result, f"unexpected error: {result}"
assert result["city"] == "Hyderabad"
assert result["temperature_c"] == 31.5
assert result["condition"] == "clear sky"
assert result["wind_kph"] == 9.0, f"m/s -> kph conversion wrong: {result['wind_kph']}"
print(f"OK: get_current_weather() success path maps the API payload correctly -> {result}")

# --- Cache: a second call for the same city within the TTL must NOT re-hit
# the (fake) API ---------------------------------------------------------
calls_before = _call_count["n"]
result2 = live_data.get_current_weather("Hyderabad")
assert _call_count["n"] == calls_before, "cache did not prevent a redundant API call within the TTL"
assert result2 == result
print("OK: get_current_weather() serves a repeated same-city question from cache within the TTL window")

# =============================================================================
# 3. get_current_weather() -- failure paths must return {"error": ...},
#    never raise, and never fabricate a reading
# =============================================================================
live_data._cache.clear()
live_data.OPENWEATHER_API_KEY = None
# Also blank the real environment variable, not just the module attribute --
# on a machine with a real OPENWEATHER_API_KEY in .env (loaded via
# advanced_rag's load_dotenv() at import time), get_current_weather()'s
# os.getenv() fallback would otherwise find that real key and this "no key
# configured" test would falsely hit the real API instead of erroring.
_real_key = os.environ.pop("OPENWEATHER_API_KEY", None)
no_key_result = live_data.get_current_weather("Hyderabad")
if _real_key is not None:
    os.environ["OPENWEATHER_API_KEY"] = _real_key
assert "error" in no_key_result, "missing API key should produce an error dict, not a fake reading"
# --- Regression: a real bug found in live testing. If the module-level
# OPENWEATHER_API_KEY constant froze in as None (e.g. because live_data was
# imported before the app's load_dotenv() call ran), the key must still be
# picked up from a live process environment variable at call time instead
# of failing forever. ---------------------------------------------------
os.environ["OPENWEATHER_API_KEY"] = "fake-key-from-process-env"
live_data._cache.clear()
live_data.requests.get = _fake_requests_get
env_fallback_result = live_data.get_current_weather("Hyderabad")
assert "error" not in env_fallback_result, (
    "regression: get_current_weather() did not fall back to a live os.getenv() read when "
    "the module-level OPENWEATHER_API_KEY constant was None -- this is exactly the import-"
    "order bug caught during live testing (live_data imported before load_dotenv() ran)"
)
del os.environ["OPENWEATHER_API_KEY"]
print("OK: get_current_weather() falls back to a live os.getenv() read when the module-level "
      "constant is None (import-order regression guard)")
print(f"OK: missing OPENWEATHER_API_KEY -> {no_key_result['error']!r}")

live_data.OPENWEATHER_API_KEY = "fake-key-for-test"
live_data._cache.clear()
live_data.requests.get = lambda url, params=None, timeout=None: _FakeResponse(404, {})
not_found_result = live_data.get_current_weather("Nowhereville")
assert "error" in not_found_result, "unknown city should produce an error dict"
print(f"OK: unknown city (HTTP 404) -> {not_found_result['error']!r}")

live_data._cache.clear()


def _raise(*a, **k):
    raise live_data.requests.RequestException("connection refused")


live_data.requests.get = _raise
network_error_result = live_data.get_current_weather("Hyderabad")
assert "error" in network_error_result, "a network exception should produce an error dict, not propagate"
print(f"OK: network failure -> {network_error_result['error']!r}")

# =============================================================================
# 4. query_document()'s live_lookup dispatch, with ZERO documents indexed --
#    this is the structural fix: previously query_document() raised
#    RuntimeError("No documents are indexed.") before classify_query() ever
#    ran, making a weather question impossible on a fresh install with
#    nothing uploaded yet.
# =============================================================================
live_data._cache.clear()
live_data.requests.get = _fake_requests_get

assert rag.list_documents() == [], "test setup assumption broken: expected a fresh, empty document store"


def _fake_classify_query_live(question, records):
    return {"strategy": "live_lookup", "operation": None, "filters": [], "reason": "fake router for offline test"}


_original_classify_query = rag.classify_query
rag.classify_query = _fake_classify_query_live

live_result = rag.query_document("What is the weather in Hyderabad right now?")
assert live_result["strategy"] == "Live data lookup", f"unexpected strategy: {live_result['strategy']}"
assert live_result["record_count"] == 1
assert live_result["sources"] and live_result["sources"][0]["type"] == "live_api"
assert live_result["sources"][0]["city"] == "Hyderabad", f"city extraction regression: {live_result['sources'][0]}"
assert live_result["document_name"] == "No documents indexed"
print(f"OK: query_document() answers a live_lookup question with zero documents indexed -> "
      f"strategy={live_result['strategy']!r}, sources={live_result['sources']}")

# --- Failure inside the live path (bad city) must still come back as a
# normal, non-crashing answer -- never a raised exception -----------------
live_data._cache.clear()
live_data.requests.get = lambda url, params=None, timeout=None: _FakeResponse(404, {})
live_result_fail = rag.query_document("What is the weather in Nowhereville right now?")
assert live_result_fail["strategy"] == "Live data lookup"
assert live_result_fail["record_count"] == 0
assert live_result_fail["sources"] == []
assert "error" in live_result_fail["live_evidence"]
print("OK: an unresolvable city surfaces as a graceful 'insufficient evidence' answer, not a crash")

rag.classify_query = _original_classify_query

# =============================================================================
# Cleanup
# =============================================================================
shutil.rmtree(tmp_storage, ignore_errors=True)
print("\nAll offline self-tests passed -- live_data.py's weather lookup and advanced_rag.py's "
      "live_lookup dispatch (including the zero-documents-indexed case) are verified, using "
      "fakes only for the network- and OpenAI-touching calls.")