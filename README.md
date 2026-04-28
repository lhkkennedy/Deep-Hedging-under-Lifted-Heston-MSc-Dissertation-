# Deep Hedging under Lifted Heston

MSc dissertation research project on **deep hedging delta-hedged ATM SPX straddles under lifted Heston**, with a focus on variance risk premium capture, transaction costs, volatility-state representation, and machine-learning hedging policies.

This repository contains the notebook pipeline for calibrating classical Heston and lifted Heston models, generating hedging experiment seeds, training and evaluating learned hedge policies, running robustness checks, and connecting the simulated hedging results back to a realistic 30-day ATM straddle strategy setting.

> **Note on filenames:** the exact filenames in this README may differ from the final repository names. The dissertation notebooks went through several versioned iterations, because naturally the only thing more unstable than stochastic volatility is notebook naming. The intended logical structure is stable even if filenames are later cleaned up.

## Project aim

The project investigates whether machine-learning-based hedging can improve the practical performance of near-ATM SPX index straddles when volatility is stochastic and transaction costs are present.

The central idea is to compare learned hedge policies against simpler benchmark hedge rules in both:

- a **classical Heston world**, with a single variance state;
- a **lifted Heston world**, which uses a finite-factor Markovian approximation to rougher volatility dynamics.

The key research question is whether richer volatility-state information improves cost-aware hedging performance, especially when the strategy is exposed to variance risk premium, trading frictions, and downside risk.

## Research hypotheses

The empirical work is built around three main hypotheses:

1. **Deep hedging improves cost-adjusted performance relative to standard delta hedging.**
2. **The gain from learned hedging is larger under lifted Heston than under classical Heston.**
3. **The marginal benefit of more frequent rebalancing declines once transaction costs are introduced.**

The project also asks whether lifted Heston can act as a tractable bridge between classical Heston and rougher non-Markovian volatility models for machine-learning hedging experiments.

## Repository structure

A cleaned final repo should look roughly like this:

```text
.
├── README.md
├── .gitignore
├── .gitattributes
├── Part_A_Calibration.ipynb
├── Part_B_Deep_Hedging.ipynb
├── Part_C_Robustness_Checks.ipynb
├── part_D_Strategy_Backtest.ipynb
├── docs/
│   ├── Dissertation_Proposal.docx
│   ├── Progress_Report_19-04.docx
│   ├── results_and_progress_report.docx
│   ├── progress_report_update_since_19-04.md
│   └── dissertation_results_section.md
├── data/
│   ├── README.md
│   └── sample/
└── outputs/
    └── README.md
```

In the current working version, some notebooks and outputs may use versioned names such as:

```text
part_A_v16_short_and_wide_fiterrorfix.ipynb
part_B_v9_short_and_wide.ipynb
part_C_sell_side_credibility_audit.ipynb
part_D_30d_atm_straddle_strategy_backtest.ipynb
```

These correspond to the logical Part A to Part D workflow described below.

## Notebook workflow

### Part A: Calibration and seed construction

**Purpose:** clean option-market data, construct calibration surfaces, calibrate classical Heston and lifted Heston models, and export the episode seeds used by the hedging experiment.

Main tasks:

- Load CRSP/SPX and OptionMetrics-style input data.
- Clean and filter option chains.
- Apply liquidity, spread, and maturity filters.
- Use put-call parity diagnostics where appropriate.
- Focus on the tighter OTM side of the market.
- Screen for static-arbitrage issues.
- Apply no-arbitrage repair where needed.
- Construct calibration targets.
- Fit classical Heston and lifted Heston models.
- Export episode seeds and calibration summaries.

The final seed panel includes three experiment families:

| Experiment family | Description | Seeds |
|---|---|---:|
| `baseline_local` | 2w, 1m and 2m traded buckets with local calibration | 54 |
| `baseline_wide` | same traded buckets but wider calibration surface | 54 |
| `short_end_local` | 1w and 2w traded buckets with local calibration | 36 |

Total: **144 episode seeds**.

### Part B: Deep hedging experiment

**Purpose:** evaluate hedge policies out of sample on simulated paths generated from calibrated Heston and lifted Heston worlds.

Main tasks:

- Load Part A episode seeds.
- Create chronological train/validation/test splits.
- Simulate paths under calibrated Heston and lifted Heston settings.
- Evaluate benchmark policies:
  - unhedged straddle;
  - Black-Scholes delta hedge;
  - model/state-vol delta hedge;
  - other delta-style benchmark rules where available.
- Train learned hedge policies.
- Train a tail-aware learned hedge with greater emphasis on downside outcomes.
- Evaluate performance across transaction-cost and hedge-frequency grids.
- Produce seed-level, experiment-level, bucket-level and regime-level summaries.
- Run date-clustered bootstrap inference.
- Assess training stability across repeated random initialisations.

The final Part B configuration evaluates:

- **16 robustness scenarios** per experiment;
- **4 transaction-cost levels**: 0.25, 0.5, 1.0 and 2.0 bps;
- **4 hedge frequencies**: 0.5, 1, 2 and 4 adjustments per day;
- **10,000 paths** for test evaluation;
- **5 training runs** for learned policies.

### Part C: Sell-side credibility and robustness audit

**Purpose:** test whether the project would survive the sort of practical questions asked by a derivatives, quant research, or sell-side desk.

Main tasks:

- Load Part A and Part B outputs.
- Replay a real-market SPX path as an anchor check.
- Audit calibration stability and runtime.
- Check whether the learned hedge is merely “trading more”.
- Review turnover discipline.
- Assess training stability.
- Review clustered-bootstrap evidence.
- Summarise desk-relevant strengths and weaknesses.

This section is designed to address practical objections such as:

- Are the results only simulated-path artefacts?
- Is the calibration too slow or unstable?
- Does the learned hedge just trade aggressively and pay the cost later?
- Are the results stable across random seeds and dates?
- Would the methodology be credible to a finance practitioner?

### Part D: 30-day ATM straddle strategy backtest

**Purpose:** connect the model and policy results to a more recognisable trading-strategy setting: rolling 30-day ATM SPX straddles.

Main tasks:

- Select the strongest Part B model/policy combination.
- Build a rolling 30-day ATM straddle sample.
- Implement pricing and delta helpers.
- Replay a realised 30-day delta-hedged strategy.
- Compare against benchmark strategy outputs.
- Produce strategy diagnostics and summary files.

Part D is not a replacement for the simulated deep-hedging experiment. It is a practical bridge showing how the best-performing policy insights relate to a more familiar implementable straddle strategy.

## Input data

The full raw input datasets are **not recommended for direct GitHub storage** because they may be too large for GitHub and Git LFS limits, and may be subject to licensing restrictions.

Expected local raw data files may include:

```text
CRSP S&P 2010 - 2024.csv
CRSP S&P 2010 - 2024.parquet
OptionMetrics ATM IV 2010 - 2025.csv
OptionMetrics ATM IV 2010 - 2025.parquet
```

Depending on the notebook version, alternative names may appear, for example:

```text
crsp_sp_2010_2024.csv
crsp_sp_2010_2024.parquet
optionmetrics_atm_iv_2010_2025.csv
optionmetrics_atm_iv_2010_2025.parquet
```

Place the required data files in the expected local directory before running the notebooks. If you rename files, update the path variables at the top of each notebook.

## Data availability

The raw CRSP and OptionMetrics data are not included in the recommended public repository structure.

Suggested approach:

- Keep full raw datasets locally.
- Add small sample files only if needed to demonstrate notebook structure.
- Store large licensed data outside GitHub.
- Use `.gitignore` to prevent accidental commits of raw data and generated outputs.

Recommended `.gitignore` entries:

```gitignore
# Large raw data
CRSP S&P 2010 - 2024.csv
CRSP S&P 2010 - 2024.parquet
OptionMetrics ATM IV 2010 - 2025.csv
OptionMetrics ATM IV 2010 - 2025.parquet
*.parquet

# Generated calibration and hedging outputs
project/lifted_heston_calibration_outputs/
lifted_heston_calibration_outputs/
outputs/
*.png

# Jupyter / Python
.ipynb_checkpoints/
__pycache__/
*.py[cod]

# Environments
.venv/
venv/
env/
.env

# OS / editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

If using Git LFS for smaller data files, track them deliberately:

```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.parquet"
git add .gitattributes
```

Do **not** rely on Git LFS for very large multi-GB raw files unless you have checked the hosting limits first. GitHub has a talent for waiting until the end of an upload before saying “no”. Charming little goblin.

## Key generated outputs

Depending on the notebook version, Part A may generate files such as:

```text
episode_seeds_v15_short_wide.csv
episode_fit_summaries_v15_short_wide.csv
episode_seed_selection_diagnostics_v15_short_wide.csv
calibration_meta.json
calibration_targets.csv
fit_summary.csv
heston_fit.csv
heston_params.json
lifted_fit.csv
lifted_params.json
iv_surface_compare.csv
kernel_diagnostics.csv
parity_detail.csv
parity_summary.csv
surface_noarb_detail.csv
surface_noarb_summary.json
target_diagnostics.csv
```

Part B may generate:

```text
part_b_v9_summary_results.csv
part_b_v9_seed_results.csv
part_b_v9_seed_results_raw.csv
part_b_v9_bootstrap_ci.csv
part_b_v9_bucket_summary_results.csv
part_b_v9_regime_summary_results.csv
part_b_v9_training_stability_results.csv
```

Part C may generate:

```text
part_c_sell_side_audit_summary.csv
```

Part D may generate:

```text
part_d_30d_atm_straddle_strategy_panel.csv
part_d_30d_atm_straddle_strategy_summary.csv
part_d_30d_atm_straddle_strategy_diagnostics.csv
```

Most generated files should be excluded from version control unless they are small, final, and directly needed to reproduce the dissertation tables.

## Main evaluation metrics

The final evaluation focuses on:

| Metric | Meaning |
|---|---|
| Mean residual P&L vs unhedged | Average economic gain relative to leaving the straddle unhedged |
| Raw P&L variance | Dispersion of hedging outcomes |
| Mean turnover | Trading intensity and cost exposure |
| 5% CVaR | Downside-tail performance |
| Bootstrap confidence intervals | Statistical support clustered by trade date |
| Training-run standard deviation | Stability across random initialisations |

The project deliberately avoids relying only on raw mean P&L, because that can mix entry pricing effects with hedging quality. Humanity occasionally learns.

## Headline findings

### 1. Learned hedging outperforms delta-style benchmarks

Across the main experiments, the standard learned hedge is the strongest overall policy when mean residual P&L, variance, turnover and CVaR are considered jointly.

In the **baseline local Heston world**, the standard learned hedge achieves:

| Metric | Standard learned hedge | Unhedged benchmark |
|---|---:|---:|
| Mean residual P&L vs unhedged | 4.29 | 0.00 |
| Raw P&L variance | 4,458 | 39,436 |
| 5% CVaR | -124.3 | -213.8 |

In the **baseline local lifted world**, the standard learned hedge achieves:

| Metric | Standard learned hedge | Unhedged benchmark |
|---|---:|---:|
| Mean residual P&L vs unhedged | 20.99 | 0.00 |
| Raw P&L variance | 28,624 | 70,438 |
| 5% CVaR | -206.0 | -221.8 |

### 2. The learned-hedge gain is larger under lifted Heston

The lifted-Heston world consistently gives the learned hedge more room to add value. This is one of the clearest empirical findings: the residual-P&L advantage is materially larger under lifted Heston than under classical Heston across the main experiment families.

### 3. Wide calibration does not overturn the main result

The baseline wide-calibration experiment asks whether the result depends on a narrow local calibration window.

In the lifted world, the standard learned hedge remains strong:

| Experiment | Average residual P&L |
|---|---:|
| Baseline local lifted | 20.99 |
| Baseline wide lifted | 21.60 |

The wide-calibration setup is more demanding and increases variance, but the core conclusion survives.

### 4. Short-end trades are harder to monetise

The short-end local experiment tests 1w and 2w traded maturities. The learned hedge still outperforms benchmark delta policies, but gains are much smaller:

| World | Standard learned hedge residual P&L |
|---|---:|
| Short-end Heston | 0.57 |
| Short-end lifted | 3.25 |

This does **not** support the simplistic claim that shorter maturity automatically strengthens the value of lifted-Heston-style state information.

### 5. Tail-aware learning gives a real risk-preference trade-off

The tail-aware learned hedge behaves as intended. It often gives up some mean residual P&L in exchange for better downside protection.

In the baseline local Heston world:

| Policy | Mean residual P&L | Variance | 5% CVaR |
|---|---:|---:|---:|
| Standard learned hedge | 4.29 | 4,458 | -124.3 |
| Tail-aware learned hedge | -0.16 | 2,918 | -95.9 |

In the baseline local lifted world:

| Policy | Mean residual P&L | Variance | 5% CVaR |
|---|---:|---:|---:|
| Standard learned hedge | 20.99 | 28,624 | -206.0 |
| Tail-aware learned hedge | 19.40 | 28,892 | -202.8 |

This shows the machine-learning component is not just a black-box “beat delta” model. It can encode different economic objectives.

### 6. More frequent hedging does not pay proportionately under costs

The robustness grid supports the hypothesis that increasing hedge frequency raises turnover without producing a proportional economic gain.

For example, in the baseline local Heston world, increasing frequency from 0.5 to 4 hedge adjustments per day causes standard learned hedge residual P&L to fall from 5.00 to 3.96 while turnover rises from 1.30 to 2.90.

### 7. Bootstrap evidence supports the main result

For the standard learned hedge in the baseline local experiment, the 95% date-clustered bootstrap interval for mean residual P&L is fully positive in:

| World | Positive bootstrap scenarios |
|---|---:|
| Heston | 15 / 16 |
| Lifted Heston | 16 / 16 |

This supports the interpretation that the learned hedge has a genuine out-of-sample edge, especially in the lifted-Heston setting.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels pyarrow fastparquet numba tqdm joblib optuna torch jupyter
```

Depending on the final notebook version, you may also need:

```bash
pip install openpyxl xlrd
```

For GPU-accelerated PyTorch, install the appropriate PyTorch build for your system rather than relying blindly on the default pip install. Blindly installing ML packages and hoping CUDA behaves is how people discover new emotions.

## Running the notebooks

Run the notebooks in order:

```text
1. Part_A_Calibration.ipynb
2. Part_B_Deep_Hedging.ipynb
3. Part_C_Robustness_Checks.ipynb
4. part_D_Strategy_Backtest.ipynb
```

Recommended workflow:

1. Confirm raw data paths in Part A.
2. Run Part A to generate calibration outputs and episode seeds.
3. Run Part B using the Part A seed file.
4. Run Part C after Part B outputs exist.
5. Run Part D after the Part B summary/seed outputs are available.

If using versioned notebook names, the equivalent order is:

```text
1. part_A_v16_short_and_wide_fiterrorfix.ipynb
2. part_B_v9_short_and_wide.ipynb
3. part_C_sell_side_credibility_audit.ipynb
4. part_D_30d_atm_straddle_strategy_backtest.ipynb
```

## Reproducibility notes

The project uses chronological train/validation/test splits by trade date and repeated learned-hedge training runs. To improve reproducibility:

- Fix random seeds before simulation and model training.
- Keep generated seed files from Part A if you want to reproduce a particular Part B run.
- Avoid manually editing intermediate CSV files.
- Record transaction-cost and hedge-frequency settings.
- Save package versions for any final dissertation run.

For a final archival version, consider exporting:

```bash
pip freeze > requirements.txt
```

## Practical interpretation

The results suggest that learned hedging is most valuable when the volatility-state representation is rich enough to create useful hedging information, but not so complicated that the experiment becomes computational mythology.

The strongest dissertation narrative is:

1. Classical Heston may fit more cleanly in some cross-sectional calibration settings.
2. Lifted Heston still creates a richer state environment for downstream hedging.
3. Learned hedging materially improves residual P&L and risk metrics relative to simple delta benchmarks.
4. The advantage is largest in lifted-Heston worlds.
5. Transaction costs make naive frequent delta hedging unattractive.
6. Tail-aware objectives allow the learned policy to express different risk preferences.

This distinction between **pricing fit** and **hedging usefulness** is the project’s most important methodological point.

## Limitations

Important limitations include:

- Raw OptionMetrics and CRSP data may be large, licensed, and difficult to distribute.
- The seed panel is moderate rather than industrial-scale.
- Calibration remains computationally expensive.
- Lifted Heston is not always the strongest cross-sectional pricing model.
- Simulated-path evidence still depends on model assumptions.
- Short-end trades are harder to monetise than the theory alone might imply.
- Results should not be interpreted as direct trading advice.

## Suggested future work

Possible extensions:

- Increase the number of test dates and calibration seeds.
- Add more maturity buckets.
- Test alternative lifted-factor specifications.
- Add explicit no-trade-band benchmark policies.
- Compare against rough-Heston simulations directly.
- Run a larger realised-data strategy backtest.
- Explore SHAP or sensitivity-style diagnostics for the learned hedge.
- Include realistic funding, margin and execution assumptions.

## Documents

Associated dissertation documents may include:

```text
Dissertation_Proposal.docx
Progress_Report_19-04.docx
progress_report_update_since_19-04.md
dissertation_results_section.md
results_and_progress_report.docx
```

These documents summarise the project design, progress, results and final interpretation.

## Author

Lewis Hikari Kawase Kennedy

MSc Finance and Machine Learning  
Queen Mary University of London
