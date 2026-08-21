"""Live weather lookup — Phase 1 of the live-data roadmap (see
IMPLEMENTATION.md's "Live data (v1.1)" section for the full design writeup).

Deliberately kept separate from advanced_rag.py's ingestion path: a current
temperature is stale within seconds, so it must never be embedded into
Chroma or written into SQLite as if it were a stable document fact. Every
call hits the API fresh, behind a short cache only to avoid re-fetching on
back-to-back questions about the same city in one session.
"""
from __future__ import annotations

import os
import re
import time

import requests

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
CACHE_TTL_SECONDS = 60

# city (lowercased) -> (monotonic_fetch_time, result_dict)
_cache: dict[str, tuple[float, dict]] = {}


_TRAILING_FILLER = r"(?:\s+(?:right\s+now|now|today|currently|please|this\s+week|at\s+the\s+moment))?"


def extract_city(question: str) -> str | None:
    """Best-effort location extraction, in the same pragmatic, regex-driven
    spirit as advanced_rag.infer_filters_from_question(): look for
    "in/for/at <City>" patterns. This is a deliberate v1 simplification —
    a production version would have the router itself extract the location
    as a structured field, the same way it extracts filter values today.
    Kept as a documented limitation rather than silently hoping it works;
    see IMPLEMENTATION.md.

    The trailing-filler group exists because "what's the weather in
    Hyderabad right now?" would otherwise greedily capture "Hyderabad right
    now" as the city -- the non-greedy city group has no way to know where
    the city name ends without SOME list of common non-city trailing words
    to stop at first. This is exactly the kind of phrasing used in this
    app's own router prompt example, so it was caught by
    test_live_data_offline.py rather than shipped silently broken."""
    m = re.search(
        r"\b(?:in|for|at)\s+([A-Za-z][A-Za-z\s]{1,40}?)" + _TRAILING_FILLER + r"(?:\?|\.|!|$)",
        question, re.IGNORECASE,
    )
    if not m:
        return None
    city = m.group(1).strip()
    return city.title() if city else None


def get_current_weather(city: str) -> dict:
    """Returns a small evidence dict on success. On any failure — missing API
    key, unknown city, network error, non-200 response — returns
    {"error": "..."} instead of raising, so the caller can hand that
    straight to generate_answer() and get an honest "insufficient evidence"
    answer rather than a stack trace or (worse) a guessed number. This
    mirrors every other strategy in this app: never let a failure upstream
    turn into a hallucinated answer downstream.

    Re-checks os.getenv() here rather than trusting only the module-level
    OPENWEATHER_API_KEY constant above: that constant is read at import
    time, and if this module gets imported before the app's load_dotenv()
    call runs, it silently freezes in as None even when .env has a real
    key. This was a real bug found during live testing -- the router chose
    live_lookup correctly, but every call failed with "not configured"
    because of import order, not a missing key."""
    api_key = OPENWEATHER_API_KEY or os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHER_API_KEY is not configured."}

    key = city.strip().lower()
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    try:
        resp = requests.get(
            OPENWEATHER_URL,
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=8,
        )
    except requests.RequestException as e:
        return {"error": f"Weather API request failed: {e}"}

    if resp.status_code == 404:
        return {"error": f"Unknown city: {city!r}."}
    if resp.status_code != 200:
        return {"error": f"Weather API returned HTTP {resp.status_code}."}

    data = resp.json()
    result = {
        "city": data.get("name", city),
        "temperature_c": data.get("main", {}).get("temp"),
        "feels_like_c": data.get("main", {}).get("feels_like"),
        "condition": (data.get("weather") or [{}])[0].get("description"),
        "humidity_pct": data.get("main", {}).get("humidity"),
        "wind_kph": round((data.get("wind", {}).get("speed") or 0) * 3.6, 1),
        "observed_at_unix": data.get("dt"),
    }
    _cache[key] = (now, result)
    return result