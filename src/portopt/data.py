"""Data acquisition for the portfolio-optimization project.

Primary source: Yahoo Finance via ``yfinance`` (daily adjusted closes).
Fallback: Stooq daily CSVs (free, no auth) if Yahoo is unreachable.
Downloaded data is cached under ``data/`` so the analysis is reproducible
without re-hitting the network.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# A documented, fixed universe: 37 liquid large-cap US equities spanning 11
# GICS-like sectors. Fixed (not "top-N by market cap at runtime") so the
# analysis is reproducible and the sector-cap constraints are meaningful.
TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "ORCL",
    # Communication Services
    "GOOGL", "META",
    # Financials
    "JPM", "BAC", "V", "MA", "GS",
    # Health Care
    "JNJ", "UNH", "LLY", "PFE", "ABBV",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD",
    # Consumer Staples
    "PG", "KO", "WMT", "PEP",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "HON", "UPS", "BA",
    # Utilities
    "NEE", "DUK",
    # Real Estate
    "AMT",
    # Materials
    "LIN",
]

SECTORS: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "ORCL": "Technology",
    "GOOGL": "Communication", "META": "Communication",
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "MA": "Financials", "GS": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "ABBV": "Healthcare",
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "PG": "ConsStaples", "KO": "ConsStaples", "WMT": "ConsStaples",
    "PEP": "ConsStaples",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "BA": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities",
    "AMT": "RealEstate",
    "LIN": "Materials",
}

BENCHMARK_TICKER = "SPY"  # S&P 500 ETF, buy-and-hold benchmark

DATA_START = "2019-01-01"
DATA_END = "2026-10-01"  # exclusive upper bound


@dataclass(frozen=True)
class DataManifest:
    tickers: list[str]
    start: str
    end: str
    source: str  # "yahoo" or "stooq"
    n_rows: int


def _download_yahoo(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False,
        threads=True,
    )
    # yfinance returns MultiIndex columns for multi-ticker; take Close.
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:  # single ticker
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    return df


def _download_stooq(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fallback: Stooq daily CSVs, one HTTP request per ticker (free, no key)."""
    import urllib.request

    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    frames = {}
    # Stooq rejects the default Python-urllib User-Agent (HTTP 404); use a
    # browser UA string instead.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (X11; Linux x86_64)")]
    for t in tickers:
        symbol = f"{t.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
        with opener.open(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"])
        if df.empty or "Close" not in df.columns:
            raise RuntimeError(f"Stooq returned no data for {t}")
        frames[t] = df.set_index("Date")["Close"]
    out = pd.DataFrame(frames).sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def download_prices(
    tickers: list[str],
    start: str = DATA_START,
    end: str = DATA_END,
    cache_path: str | Path | None = None,
) -> tuple[pd.DataFrame, DataManifest]:
    """Download daily closes, trying Yahoo first and Stooq on failure."""
    source = "yahoo"
    try:
        prices = _download_yahoo(tickers, start, end)
        if prices.empty or prices.isna().all().all():
            raise RuntimeError("yfinance returned empty data")
    except Exception as exc:  # noqa: BLE001 - intentional fallback
        log.warning("yfinance failed (%s); falling back to Stooq", exc)
        prices = _download_stooq(tickers, start, end)
        source = "stooq"

    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="any")
    kept = [t for t in tickers if t in prices.columns]
    prices = prices[kept]
    manifest = DataManifest(
        tickers=kept, start=start, end=end, source=source, n_rows=len(prices)
    )
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
        log.info("cached %d x %d prices -> %s", *prices.shape, cache_path)
    return prices, manifest


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, r_t = ln(P_t / P_{t-1})."""
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    return returns


def load_cached_prices(cache_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return df
