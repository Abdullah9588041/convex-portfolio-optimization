# Data

`prices.csv` (daily closes for the 37-ticker universe) and `spy.csv` (SPY
benchmark) are **downloaded, not committed** — regenerate them with:

```bash
python scripts/run_analysis.py --download
```

* Primary source: Yahoo Finance via `yfinance` (daily adjusted closes).
* Fallback: Stooq daily CSVs (`https://stooq.com/q/d/l/`, free, no API key)
  if Yahoo is unreachable. The source used is printed at download time.

Universe: 37 liquid large-cap US equities across 11 sectors, fixed in
`src/portopt/data.py` (`TICKERS` / `SECTORS`). Fixed — not "top-N by market
cap at runtime" — so the analysis is reproducible and sector-cap constraints
are meaningful.

Date range: 2019-01-01 to 2026-10-01 (exclusive).
