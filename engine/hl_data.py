"""
Hyperliquid public market data fetcher.
Fetches 1h OHLCV candles for the active universe.

429 hardening (2026-05-11):
  - Per-engine rate limiter (token bucket via min-interval lock)
  - 429-specific exponential backoff with jitter
  - Honor Retry-After header
  - Distinct log line for 429 vs other failures
"""
from __future__ import annotations
import json
import os
import random
import threading
import time
import urllib.request
import urllib.error
from typing import Optional
import pandas as pd

from .config import HL_REST


# ─────────────────────────────────────────────────────────────────────────
# Per-engine HTTP throttle. Configurable via HL_MIN_INTERVAL_MS env.
# Default 250ms = max 4 calls/sec from this engine. Combined with the
# other v-engines this gives HL's per-IP budget headroom.
# ─────────────────────────────────────────────────────────────────────────
_MIN_INTERVAL_S = max(0.05, float(os.environ.get("HL_MIN_INTERVAL_MS", "250")) / 1000.0)
_rate_lock = threading.Lock()
_last_call_t = [0.0]


def _throttle() -> None:
    with _rate_lock:
        dt = time.monotonic() - _last_call_t[0]
        if dt < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - dt)
        _last_call_t[0] = time.monotonic()


def _post(payload: dict, retries: int = 5, timeout: int = 15) -> Optional[list]:
    body = json.dumps(payload).encode()
    for i in range(retries):
        _throttle()
        try:
            req = urllib.request.Request(
                HL_REST, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_s = 0.0
                ra = e.headers.get("Retry-After", "") if e.headers else ""
                try:
                    if ra:
                        wait_s = float(ra)
                except ValueError:
                    wait_s = 0.0
                if wait_s <= 0:
                    wait_s = min(60.0, (2 ** (i + 2)) + random.uniform(0.0, 3.0))
                if i == retries - 1:
                    print(f"[hl_data] POST 429 after {retries} retries (last wait {wait_s:.1f}s)", flush=True)
                    return None
                time.sleep(wait_s)
            else:
                if i == retries - 1:
                    print(f"[hl_data] POST failed after {retries}: HTTP {e.code} {e.reason}", flush=True)
                    return None
                time.sleep(min(10.0, 2 ** i) + random.uniform(0.0, 1.0))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if i == retries - 1:
                print(f"[hl_data] POST failed after {retries}: {e}", flush=True)
                return None
            time.sleep(min(10.0, 2 ** i) + random.uniform(0.0, 1.0))
    return None


def fetch_candles(coin: str, interval: str = "1h", n_bars: int = 200) -> Optional[pd.DataFrame]:
    """
    Fetch last `n_bars` candles for `coin` from HL.
    Returns DataFrame with [open, high, low, close, volume] indexed by timestamp.
    """
    end_ms = int(time.time() * 1000)
    bar_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}.get(interval, 3_600_000)
    start_ms = end_ms - n_bars * bar_ms

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    data = _post(payload)
    if not data:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None

    rows = []
    for c in data:
        try:
            rows.append({
                "ts": int(c["t"]),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "volume": float(c["v"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df


def fetch_meta() -> Optional[dict]:
    """Get HL universe metadata (sz_decimals, max_leverage, etc.)."""
    return _post({"type": "meta"})


def fetch_mids() -> Optional[dict]:
    """Get current mid prices for all coins."""
    return _post({"type": "allMids"})



def fetch_l2_book(coin: str) -> Optional[dict]:
    """Fetch L2 orderbook from HL. Returns {"bids":[...], "asks":[...]} or None."""
    try:
        body = json.dumps({"type": "l2Book", "coin": coin}).encode()
        req = urllib.request.Request(_HL_INFO_URL, data=body,
                                       headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read())
            # Returns: {"coin":..., "time":..., "levels":[[bids],[asks]]}
            levels = data.get("levels", [])
            if len(levels) < 2: return None
            return {
                "bids": [{"px": float(b["px"]), "sz": float(b["sz"])} for b in levels[0]],
                "asks": [{"px": float(a["px"]), "sz": float(a["sz"])} for a in levels[1]],
                "ts": data.get("time"),
            }
    except Exception as e:
        return None


def compute_book_imbalance(book: dict, range_pct: float = 0.005) -> Optional[float]:
    """Compute bid/ask depth ratio within `range_pct` of mid.

    Returns:
       imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
       > 0  : more bids than asks (bullish lean — support nearby)
       < 0  : more asks than bids (bearish lean — resistance nearby)
       0    : balanced

    Returns None if book is empty or sparse.
    """
    if not book or not book.get("bids") or not book.get("asks"):
        return None
    best_bid = book["bids"][0]["px"]
    best_ask = book["asks"][0]["px"]
    mid = (best_bid + best_ask) / 2
    band_lo = mid * (1 - range_pct)
    band_hi = mid * (1 + range_pct)
    bid_depth = sum(b["sz"] for b in book["bids"] if b["px"] >= band_lo)
    ask_depth = sum(a["sz"] for a in book["asks"] if a["px"] <= band_hi)
    total = bid_depth + ask_depth
    if total <= 0: return None
    return (bid_depth - ask_depth) / total
