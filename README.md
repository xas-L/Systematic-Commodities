# Systematic-Commodities

An attempt at prod‑style, **curve RV** research for commodity futures that trades **exchange combo calendars and butterflies**. The stack emphasizes **economics first** (carry/roll, momentum, seasonality, PCA, optional cointegration), **cost‑aware walk‑forwards**, and **clean attribution**. No deep learning; ML overkill is not needed.

---

## Contents

* [Why this exists](#why-this-exists)
* [Project structure](#project-structure)
* [Quick start](#quick-start)
* [Configuration](#configuration)

  * [settings.yaml](#settingsyaml)
  * [fees_slippage.yaml](#feesslippageyaml)
  * [risk_limits.yaml](#risk_limitsyaml)
* [Data expectations](#data-expectations)
* [Models & signals](#models--signals)
* [Execution simulation & costs](#execution-simulation--costs)
* [Walk‑forward & attribution](#walkforward--attribution)
* [Outputs (PM brief)](#outputs-pm-brief)
* [Testing](#testing)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [License](#license)

---

## Why this exists

AFAIK Research projects in commodities often rely on back‑adjusted continuous series and level‑based predictors, which are **not tradable** in the way desks actually run risk. This repo:

* Builds curves from **individual contract histories** (no back‑adjusted series).
* Trades **shapes** (adjacent calendars; 1:‑2:1 flies) as exchange‑defined combo strategies.
* Provides a **cost‑aware** simulator (fees, half‑spreads, partial fills) and PM‑grade **walk‑forward** evaluation.
* Produces **attribution** against carry, momentum, seasonality, PCA, and (optionally) cointegration.

---

## Project structure

```
Systematic-Commodities/
  config/
    settings.yaml          # universe, folds, model settings, ops
    fees_slippage.yaml     # product fee+slippage profiles (combo-aware)
    risk_limits.yaml       # position, expiry, kill criteria
    calendars/             # optional exchange holiday CSVs
  data/
    raw/                   # vendor dumps of INDIVIDUAL contracts (CSV/Parquet)
    meta/                  # per-contract specs (tick, multiplier, FND/LTD)
    curated/               # point-in-time Parquet snapshots
  src/
    core/                  # types, utils, logging, health, scheduling
    data/                  # loaders, curve construction, spreads, tags
    models/                # hub + PCA, Nelson-Siegel, Carry/Momo/Season, Cointegration
    signals/               # sizing + attribution helpers
    execsim/               # combo simulator, cost model, backtester, walk-forward
    ops/                   # engine, execution/order/roll mgmt, kill-switch, hot-reload
  tests/
    unit/                  # fast unit tests
    integration/           # smoke integration tests
  reports/
    pm_brief/              # artifacts (pnl_path.csv, trade_log.csv, summaries)
  scripts/
    build_curve_snapshot.py
    run_walkforward.py
    gen_pm_brief.py
  Makefile
  pytest.ini
  README.md
```

---

## Quick start

> Requires Python **3.10+**.

1. **Create a venv & install deps**

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas scipy scikit-learn statsmodels pyyaml pyarrow rich
```

(Optionally add `flake8` and `black` for linting.)

2. **Drop raw contract files** under `data/raw/<SYMBOL>/` (CSV or Parquet). Each file should contain columns at least:

```
Date, Symbol, Expiry, Settle (or Last), Bid, Ask, Volume, OpenInterest
```

3. **Configure the universe & models** in `config/settings.yaml` (CL is pre‑wired as an example).

4. **Build a curve snapshot** (optional):

```bash
python scripts/build_curve_snapshot.py --symbol CL
```

5. **Run a walk‑forward** backtest and write outputs to `reports/pm_brief/<SYMBOL>/`:

```bash
python scripts/run_walkforward.py --symbol CL
```

6. **Generate PM brief tables**:

```bash
python scripts/gen_pm_brief.py --symbol CL
```

> Or use `make` targets: `make build-snapshot`, `make run`, `make brief`, `make test`.

---

## Configuration

### `settings.yaml`

Key sections:

* **universe.curves** - which roots to trade and their curve depth (e.g., `min_contracts: 6`).
* **curve_construction** - filters (min volume/OI), price field order, target tenors.
* **models** - enable/disable models; hub order; parameters (e.g., PCA spans, CMS momentum window, cointegration lookback).
* **signal_and_sizing** - z‑clip, risk per curve (USD), optional vol targeting (lookback, annual target).
* **execution_sim** - choose fee/slippage profile (refers into `fees_slippage.yaml`).
* **walkforward** - explicit folds or defaults (anchored with embargo in code).
* **ops.logging** - log level and rotation.

### `fees_slippage.yaml`

Per‑product tick/multiplier and combo‑aware half‑spread/penalties. Profiles like `conservative` and `aggressive` are supported; the simulator uses the selected profile to add **realistic costs**.

### `risk_limits.yaml`

Budget per curve, gross/net caps, expiry ladder rules (auto‑flatten before FND/LTD), concentration caps, and **kill criteria** thresholds (slippage divergence, drawdown, model drift) - wired to health/guards.

---

## Data expectations

* **Individual** futures contracts, not back‑adjusted continuous series.
* `date` and `expiry` parsed as dates; one row per (date, expiry).
* Price resolution preference: `settle` → `last` (configurable).
* Metadata under `data/meta/contracts/<SYMBOL>_meta.csv` improves roll/FND/LTD handling (columns: `expiry, first_notice_date, last_trade_date, tick_size, multiplier, currency`).

---

## Models & signals

All models implement a common interface through `src/models/hub.py`:

* **Carry/Momentum/Seasonality (CMS)** - carry proxy on log‑adjacent spreads (optionally residualised for significant month effects) + k‑day momentum; EWMA z‑scored.
* **PCA** - PCA over spread panel; EWMA z‑scored factor scores. Used for attribution/health and as a baseline.
* **Nelson–Siegel (optional)** - level/slope/curvature via fixed‑lambda least squares per cross‑section.
* **Cointegration (optional)** - rolling Engle–Granger p‑values for adjacent spreads.

Signals are averaged into per‑combo intents (e.g., average of `carry_<combo>` and `momoK_<combo>`), then sized risk‑aware with optional vol targeting.

---

## Execution simulation & costs

* **Exchange combo** semantics (calendars, flies). No synthetic legging by default.
* Costs: **half‑spread + slippage penalties + round‑turn fees** from `fees_slippage.yaml`.
* Partial fills: configurable fraction of TOB; timeouts and queue proxy.
* Order lifecycle & logs captured; simple order/roll managers included in `src/ops/`.

> In live/paper mode, swap the simulator for a broker adapter; the module boundaries are designed for that.

---

## Walk‑forward & attribution

* **Anchored walk‑forward** with embargo; models refit per fold on train, evaluated on test only.
* **Attribution**: residualises factor P&L (Gram–Schmidt style) to show net contributions of carry, momentum, seasonality, PCA, and cointegration.
* **Stress windows**: the repo provides hooks to slice outputs by dates (e.g., 2008, 2014 oil, April 2020 WTI, 2022 energy); provide those ranges in your brief.

---

## Outputs (PM brief)

Under `reports/pm_brief/<SYMBOL>/` you’ll find:

* `pnl_path.csv` - daily path‑wise P&L (net of simulated costs).
* `trade_log.csv` - order logs (fills, avg price, fees).
* `pm_summary_*.csv` - total/daily μ/σ/Sharpe.
* `pm_trades_*.csv` - trades aggregated by combo.

Use these to work on a write-up/brief: **Method**, **Execution & Costs**, **Results & Capacity**, **Risk & Kill Criteria**.

---

## Testing

Run unit & integration tests:

```bash
pytest -q
# or
make test
```

Tests use synthetic data; they’re fast and don't need vendor files.

---

## Troubleshooting

* **No data files found** - ensure `data/raw/<SYMBOL>/` contains CSV/Parquet with `Date/Symbol/Expiry/...` columns.
* **Insufficient surface depth** - you need ≥2 expiries per day to build calendars; increase `min_contracts` or check filters.
* **PCA not fitted** - verify there are enough non‑NaN rows (default `min_samples=100`).
* **Weird fills** - check `fees_slippage.yaml` (tick sizes, half‑spread ticks) and the `tob_size` stub in scripts.

---

## Roadmap

* Add exchange holiday calendars per venue.
* Extend quotes pipeline for **flies** (mid as 1:‑2:1 of legs) and capacity checks (%ADV, TOB depth).
* Optional regime tags (`src/data/state_tags.py`) in the signal combiner.
* Plug‑in broker adapter for paper/live modes.

---

## License

This was a project to get my feet wet, thus is for evaluation and research use. FAFO.

