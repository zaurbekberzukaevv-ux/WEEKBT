# WeekBT

Leakage-safe Python backtesting framework for a similarity-based market regime analog strategy.

## Setup

The project targets Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts, run this once for the current shell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

## Run

Default run uses `ETH/USDT`, `1d`, and automatically chooses 5 years of history. If 5 years are not available, it falls back to 4, then 3 years, then the earliest available data.

```powershell
python -m weekbt.cli
```

Example with explicit parameters:

```powershell
python -m weekbt.cli --symbol ETH/USDT --timeframe 1d --window-size 20 --horizon 5 --top-n 10 --exclusion-radius 5
```

Example with explicit dates:

```powershell
python -m weekbt.cli --symbol ETH/USDT --timeframe 4h --start-date 2021-01-01 --end-date 2025-12-31
```

## Outputs

Downloaded OHLCV CSV cache is stored in:

```text
data/cache/
```

Per-cycle reports are written to:

```text
reports/
```

Each cycle writes:

- `cycle_report_{cycle_start}_{cycle_end}.json`
- `cycle_predictions_{cycle_start}_{cycle_end}.csv`

## Tests

```powershell
python -m pytest
```
