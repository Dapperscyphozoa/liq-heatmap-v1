"""
liq-heatmap-v1 — Stop-hunt cluster fader.
Fades sweep wicks into clusters of equal highs/lows (where retail SLs cluster).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from .config import STRATEGY_PARAMS, TRADE_PARAMS


def calc_atr(highs, lows, closes, period: int = 14) -> float:
    h_s = pd.Series(highs); l_s = pd.Series(lows); pc = pd.Series(closes).shift(1)
    tr = pd.concat([h_s - l_s, (h_s - pc).abs(), (l_s - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _pivots_high(a, lb):
    return [i for i in range(lb, len(a)-lb)
            if a[i] >= a[i-lb:i].max() and a[i] >= a[i+1:i+1+lb].max()]


def _pivots_low(a, lb):
    return [i for i in range(lb, len(a)-lb)
            if a[i] <= a[i-lb:i].min() and a[i] <= a[i+1:i+1+lb].min()]


def _cluster(prices: List[float], band: float, min_n: int) -> List[Tuple[float, int]]:
    if not prices: return []
    s = sorted(prices); clusters: List[List[float]] = [[s[0]]]
    for p in s[1:]:
        m = sum(clusters[-1]) / len(clusters[-1])
        if abs(p - m) / m <= band: clusters[-1].append(p)
        else: clusters.append([p])
    out = [(sum(c)/len(c), len(c)) for c in clusters if len(c) >= min_n]
    return sorted(out, key=lambda x: -x[1])


import os as _lh1_os
_LH1_INVERTED = _lh1_os.environ.get("LH1_INVERTED", "0") == "1"

def _evaluate_with_thresholds(df: pd.DataFrame, min_pivots: int, vol_mult: float) -> Optional[dict]:
    LB = STRATEGY_PARAMS.get("cluster_lookback", 120)
    PIV = STRATEGY_PARAMS.get("pivot_lookback", 5)
    BAND = STRATEGY_PARAMS.get("cluster_band_pct", 0.003)
    MIN = min_pivots
    SWEEP = STRATEGY_PARAMS.get("sweep_threshold_pct", 0.002)
    VSPIKE = vol_mult
    PROX = STRATEGY_PARAMS.get("max_cluster_proximity_pct", 0.020)

    if df is None or len(df) < LB + 20: return None
    r = df.iloc[-LB:]
    highs = r["high"].values; lows = r["low"].values; closes = r["close"].values
    opens = r["open"].values
    vols = r["volume"].values if "volume" in r.columns else np.ones(len(r))

    sh_px = [float(highs[i]) for i in _pivots_high(highs, PIV)]
    sl_px = [float(lows[i])  for i in _pivots_low(lows, PIV)]
    bsl = _cluster(sh_px, BAND, MIN)
    ssl = _cluster(sl_px, BAND, MIN)
    if not bsl and not ssl: return None

    last_c = float(closes[-1]); last_h = float(highs[-1]); last_l = float(lows[-1])
    last_o = float(opens[-1]);  last_v = float(vols[-1])
    rng = last_h - last_l
    if rng <= 0: return None

    atr = calc_atr(highs, lows, closes, TRADE_PARAMS["atr_period"])
    if not atr or atr <= 0: return None

    avg_v = float(np.mean(vols[-21:-1])) if len(vols) >= 21 else float(np.mean(vols[:-1]))
    vspike = last_v / avg_v if avg_v > 0 else 0
    if vspike < VSPIKE: return None
    body_ratio = abs(last_c - last_o) / rng
    if body_ratio > 0.35: return None

    is_long = None; fired_pool = None; swept_lo = swept_hi = None
    for px, n in bsl:
        if abs(last_c - px) / px > PROX: continue
        bhi = px * (1 + BAND); blo = px * (1 - BAND)
        if last_h >= bhi and last_c < bhi:
            if (last_h - bhi) / px >= SWEEP * 0.5:
                is_long, fired_pool = False, (px, n, "BSL")
                swept_lo, swept_hi = blo, bhi
                break

    if is_long is None:
        for px, n in ssl:
            if abs(last_c - px) / px > PROX: continue
            bhi = px * (1 + BAND); blo = px * (1 - BAND)
            if last_l <= blo and last_c > blo:
                if (blo - last_l) / px >= SWEEP * 0.5:
                    is_long, fired_pool = True, (px, n, "SSL")
                    swept_lo, swept_hi = blo, bhi
                    break

    if is_long is None: return None
    pool_px, n_mem, pool_type = fired_pool

    # LH1 inversion test (2026-05-16): backtest OOS PF was 0.84/0.83 train/test
    # on the fade direction. Test the continuation direction by inverting.
    # Enable via env LH1_INVERTED=1.
    # Note: SL/TP block below uses swept_lo for LONG and swept_hi for SHORT.
    # Those are tied to the SWEPT zone (the wick), not the direction. On inversion
    # we swap them so SL still sits on the swept-zone side. Also swap pools used
    # for TP search (bsl <-> ssl) since the targets flip too.
    if _LH1_INVERTED:
        is_long = not is_long
        swept_lo, swept_hi = swept_hi, swept_lo
        bsl, ssl = ssl, bsl

    if is_long:
        sl_p = swept_lo * (1 - 0.003)
        sl_d = last_c - sl_p
        opp = [p for p, _ in bsl if p > last_c * 1.005]
        tp_p = min(opp) if opp else last_c + 2 * sl_d
        if (tp_p - last_c) < 2 * sl_d: tp_p = last_c + 2 * sl_d
    else:
        sl_p = swept_hi * (1 + 0.003)
        sl_d = sl_p - last_c
        opp = [p for p, _ in ssl if p < last_c * 0.995]
        tp_p = max(opp) if opp else last_c - 2 * sl_d
        if (last_c - tp_p) < 2 * sl_d: tp_p = last_c - 2 * sl_d

    sl_pct = abs(last_c - sl_p) / last_c
    if sl_pct < 0.002 or sl_pct > 0.05: return None

    return {
        "fire_ts": df.index[-1], "ref_price": last_c, "atr": atr,
        "trade_side": "B" if is_long else "A", "is_long": is_long,
        "sl_px": float(sl_p), "tp_px": float(tp_p),
        "max_hold_bars": TRADE_PARAMS["max_hold_bars"],
        "fire_reason": f"sweep_{pool_type}_n={n_mem}",
        "raw_direction": "LONG" if is_long else "SHORT",
        "fade_direction": "LONG" if is_long else "SHORT",
        "pool_price": float(pool_px), "pool_type": pool_type,
        "pool_members": int(n_mem), "vol_spike": float(vspike),
        "body_ratio": float(body_ratio),
    }


def evaluate_latest_bar(df) -> Optional[dict]:
    """Tiered conviction scanner.

    Tries strict threshold first (full size). If no fire, retries with weak
    threshold (quarter size). Returns a signal dict augmented with
    `conviction` and `size_multiplier` fields.

    Strict / weak thresholds come from STRATEGY_PARAMS:
      strict: min_cluster_pivots, vol_spike_mult            (defaults: 3, 1.8)
      weak:   min_cluster_pivots_weak, vol_spike_mult_weak  (defaults: 2, 1.3)
    """
    # Strict tier: full conviction, full size
    strict_min = STRATEGY_PARAMS.get("min_cluster_pivots", 3)
    strict_vol = STRATEGY_PARAMS.get("vol_spike_mult", 1.8)
    sig = _evaluate_with_thresholds(df, strict_min, strict_vol)
    if sig is not None:
        sig["conviction"] = "strong"
        sig["size_multiplier"] = 1.0
        sig["fire_reason"] = f"{sig.get('fire_reason','')}_STRONG"
        return sig

    # Weak tier: lower conviction, quarter size
    weak_min = STRATEGY_PARAMS.get("min_cluster_pivots_weak", 2)
    weak_vol = STRATEGY_PARAMS.get("vol_spike_mult_weak", 1.3)
    if weak_min >= strict_min and weak_vol >= strict_vol:
        return None   # weak is not actually weaker than strict — skip
    sig = _evaluate_with_thresholds(df, weak_min, weak_vol)
    if sig is not None:
        sig["conviction"] = "weak"
        sig["size_multiplier"] = 0.25
        sig["fire_reason"] = f"{sig.get('fire_reason','')}_WEAK"
        return sig

    return None
