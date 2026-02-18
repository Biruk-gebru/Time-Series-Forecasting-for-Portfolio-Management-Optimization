# Time Series Forecasting for Portfolio Management

**10 Academy – KAIM Week 9**

## Overview
End-to-end pipeline for time series forecasting and portfolio optimization using TSLA, BND, and SPY data from YFinance (2015–2026).

## Project Structure
```
prod/
├── .github/workflows/unittests.yml  # CI: pytest on every PR
├── data/processed/                  # Cleaned CSVs
├── notebooks/                       # Jupyter notebooks per task
│   └── images/                      # Saved plots for reports
├── src/                             # Reusable modules
├── tests/                           # Unit tests
└── scripts/                         # Utility scripts
```

## Tasks
| Task | Branch | Description |
|------|--------|-------------|
| 1 | `task/task-1` | EDA, cleaning, stationarity, risk metrics |
| 2 | `task/task-2` | ARIMA/SARIMA + LSTM models |
| 3 | `task/task-3` | Future forecasts with confidence intervals |
| 4 | `task/task-4` | Efficient Frontier & portfolio optimization |
| 5 | `task/task-5` | Backtesting vs benchmark |

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Running Tests
```bash
pytest tests/ -v
```
