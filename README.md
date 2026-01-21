# Crisis-Globalisation

## Globalization Increases During Crises, Not Stability

This repository contains the complete empirical analysis pipeline for the study **“Globalization Increases During Crises, Not Stability.”** The project examines whether global financial market integration is stable over time or intensifies during periods of heightened uncertainty and crisis.

---

## Repository Structure (Before Execution)

```
.
├── main.py
├── requirements.txt
└── README.md
```

All data files, output folders, figures, and tables are automatically generated after running the analysis.

---

## Research Objective

The study challenges the conventional assumption that globalization provides stable diversification benefits. It empirically tests whether global equity market correlations increase during crises, thereby eroding diversification when it is most needed.

Key objectives:

* Test the Decoupling Hypothesis
* Measure crisis-induced changes in global equity correlations
* Evaluate emerging markets as diversification assets
* Estimate the persistence of panic-driven synchronization

---

## Markets Covered (Global 7)

* United States (S&P 500)
* United Kingdom (FTSE 100)
* Germany (DAX)
* Japan (Nikkei 225)
* Hong Kong (Hang Seng)
* Brazil (Bovespa)
* India (BSE Sensex)

---

## Data Description

* Frequency: Daily
* Sample Period: January 2000 – January 2026
* Equity Prices: Adjusted closing prices
* Volatility Proxy: CBOE VIX Index

Data Sources:

* Yahoo Finance (via `yfinance`)
* CBOE Global Markets (VIX)

---

## Methodology

1. **Data Extraction**
   Daily equity index prices and VIX data are downloaded and aligned.

2. **Variable Construction**
   Prices are converted into logarithmic returns and global correlation measures are constructed.

3. **Crisis Regime Identification**
   Crisis periods are defined using the 75th percentile of the VIX, classifying observations into Stable and Crisis regimes.

4. **Statistical Analysis**
   Rolling correlations (30, 60, and 90-day windows), difference-in-means (Delta) analysis, Welch’s t-tests, and OLS regression of global correlation on VIX are performed.

5. **Decay Dynamics**
   An exponential decay model is fitted to post-crisis correlations to estimate the half-life of panic-induced synchronization.

---

## How to Run the Analysis

1. Clone the repository

```
git clone https://github.com/DebugDatta/Crisis-Globalisation.git
cd Crisis-Globalisation
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Execute the analysis

```
python main.py
```

No command-line arguments are required. All parameters are defined within `main.py`.

---

## Files and Folders Generated After Execution

```
data/
├── raw/
│   ├── prices.csv
│   └── vix.csv
└── processed/

outputs/
├── figures/
│   ├── 01_market_overview.png
│   ├── 02_crisis_identification.png
│   ├── 03_correlation_matrices.png
│   ├── 04_rolling_integration.png
│   ├── 05_regional_comparison.png
│   ├── 06_decay_dynamics.png
│   ├── 07_robustness_check.png
│   └── 08_regression_analysis.png
└── tables/
    ├── descriptive_stats.csv
    ├── robustness_full.csv
    └── regression_summary.txt

execution_log.txt
```

---

## Output Description

* **Figures:** Normalized market trajectories, VIX with crisis regimes, rolling global integration index, correlation heatmaps, regional comparisons, decay dynamics, robustness checks, and regression diagnostics.
* **Tables:** Descriptive statistics, robustness test results, and OLS regression summaries.
* **Logs:** Execution diagnostics and run status.

All outputs correspond directly to figures and tables reported in the research paper.

---

## Reproducibility

All methodological choices, including rolling windows, crisis thresholds, and regression specifications, are explicitly defined in the code. Running `main.py` from a clean repository state fully reproduces the analysis and results.
