# Parameter Optimization Guide

## Overview

This guide explains how to optimize the XAUUSD trading bot parameters using Optuna, a hyperparameter optimization framework.

## Prerequisites

1. **Install Optuna and dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure MT5 is connected:**
   - IC Markets account configured in `config/mt5_config.yaml`
   - MT5 terminal running

## Optimization Process

### Step 1: Run Optimization (200 trials, 6 months)

```bash
python scripts/run_optimization.py --trials 200 --months 6
```

**What this does:**
- Runs 200 different parameter combinations
- Tests each on 6 months of historical data
- Finds the best parameters that maximize:
  - Profit Factor × Win Rate / Max Drawdown

**Options:**
```bash
# Custom number of trials
python scripts/run_optimization.py --trials 100

# Custom date range
python scripts/run_optimization.py --start-date 2024-06-01 --end-date 2024-12-31

# Different timeframe
python scripts/run_optimization.py --timeframe M5 --trials 150

# Parallel processing (faster, but uses more CPU)
python scripts/run_optimization.py --trials 200 --n-jobs 4

# Custom study name
python scripts/run_optimization.py --trials 200 --study-name "my_optimization_v1"
```

### Step 2: Review Results

After optimization completes, check:
```
data/optimization_results/
  └── xauusd_opt_YYYY-MM-DD_YYYY-MM-DD_results.json      # Optimization results
  └── xauusd_opt_YYYY-MM-DD_YYYY-MM-DD_best_config.yaml  # Best parameters
```

**Results include:**
- Best score
- Best parameters
- Number of trials completed
- Optimization date

### Step 3: Apply Best Parameters

**Dry run (preview changes):**
```bash
python scripts/apply_optimized_params.py xauusd_opt_2024-08-01_2025-02-01 --dry-run
```

**Apply with backup:**
```bash
python scripts/apply_optimized_params.py xauusd_opt_2024-08-01_2025-02-01 --backup
```

This updates:
- `config/trading_rules.yaml` - Strategy and indicator parameters
- `config/risk_config.yaml` - SL/TP and risk parameters
- `config/session_config.yaml` - Session weights

### Step 4: Validate with Backtest

```bash
# Test optimized parameters on recent data
python scripts/run_backtest.py --start-date 2025-01-01 --months 1
```

Compare metrics:
- Before vs After optimization
- Win rate should be >= 55%
- Profit factor should be >= 1.5
- Max drawdown should be <= 15%

### Step 5: Forward Test on Demo

```bash
python main.py --mode demo
```

Run for at least 2 weeks to validate live performance.

## Parameters Being Optimized

### 1. Strategy Parameters
- `min_confluence_score` (0.55 - 0.75)
  - Minimum confidence threshold for entry signals

### 2. Risk Management
- `sl_atr_multiplier` (1.5 - 3.5)
  - Stop loss distance in ATR multiples
- `tp_atr_multiplier` (3.0 - 7.0)
  - Take profit distance in ATR multiples
- `trailing_activation_percent` (5 - 15%)
  - When to activate trailing stop
- `trailing_distance_percent` (3 - 10%)
  - Trailing stop distance from peak

### 3. SMC Indicators
- `fvg_min_gap_atr_ratio` (0.3 - 0.8)
  - Fair Value Gap minimum size
- `ob_min_body_percent` (50 - 80%)
  - Order Block minimum candle body
- `liquidity_lookback_bars` (10 - 30)
  - Liquidity sweep detection period

### 4. Session Weights
- `overlap_weight` (1.1 - 1.3)
  - London-NY overlap importance
- `london_ny_weight` (0.9 - 1.1)
  - London/NY session importance
- `asian_weight` (0.5 - 0.8)
  - Asian session importance

### 5. Confluence Weights
- `smc_weight` (0.35 - 0.50)
  - SMC indicators contribution
- `technical_weight` (0.20 - 0.35)
  - Technical indicators contribution
- `market_weight` (0.15 - 0.30)
  - Market condition contribution
- `mtf_weight` (0.10 - 0.25)
  - Multi-timeframe contribution

## Optimization Score

The optimizer maximizes this score:

```
Score = (Profit Factor × Win Rate × 100) / Max Drawdown %
```

**What makes a good score:**
- Score > 100: Excellent
- Score > 50: Good
- Score > 25: Acceptable
- Score < 25: Poor

## Tips for Best Results

### 1. Data Quality
- Use at least 6 months of data
- Avoid periods with market anomalies
- Include different market conditions

### 2. Number of Trials
- Minimum: 100 trials
- Recommended: 200 trials
- More trials = better exploration, but longer time

### 3. Validation
- Always backtest optimized parameters on different data
- Never deploy without forward testing
- Monitor first week closely on demo

### 4. Re-optimization
- Re-optimize every 3-6 months
- Market conditions change
- Parameters may need adjustment

### 5. Parallel Processing
```bash
# Use all CPU cores for faster optimization
python scripts/run_optimization.py --trials 200 --n-jobs -1
```

## Troubleshooting

### Issue: Optimization takes too long
**Solution:**
- Reduce trials: `--trials 100`
- Use parallel processing: `--n-jobs 4`
- Reduce data period: `--months 3`

### Issue: All trials score 0.0
**Solution:**
- Check MT5 connection
- Verify historical data availability
- Check logs for errors
- Ensure sufficient bars (>100)

### Issue: Best score is very low (<10)
**Solution:**
- Try wider parameter ranges
- Check if strategy logic is correct
- Verify data quality
- Consider different optimization period

### Issue: Parameters don't apply
**Solution:**
- Check study name is correct
- Verify config files exist
- Use `--dry-run` to preview changes
- Check file permissions

## Example Workflow

```bash
# 1. Run optimization
python scripts/run_optimization.py \
  --trials 200 \
  --months 6 \
  --study-name "gold_opt_v1"

# 2. Review results
cat data/optimization_results/gold_opt_v1_results.json

# 3. Preview changes
python scripts/apply_optimized_params.py gold_opt_v1 --dry-run

# 4. Apply with backup
python scripts/apply_optimized_params.py gold_opt_v1 --backup

# 5. Validate on recent data
python scripts/run_backtest.py --months 1

# 6. Forward test on demo
python main.py --mode demo

# 7. Monitor for 2 weeks

# 8. Deploy to live (only after successful demo testing)
python main.py --mode live
```

## Best Practices

1. **Never skip forward testing**
   - Optimized parameters may overfit historical data
   - Always test on demo first

2. **Compare before/after**
   - Run backtest before optimization
   - Compare results after optimization
   - Ensure improvement is significant

3. **Keep records**
   - Save optimization results
   - Document parameter changes
   - Track performance over time

4. **Regular re-optimization**
   - Every 3-6 months
   - After major market events
   - When performance degrades

5. **Diversify testing**
   - Test on different periods
   - Test on different timeframes
   - Validate robustness

## Expected Results

After proper optimization, expect:
- **Win Rate:** 55-65%
- **Profit Factor:** 1.5-2.5
- **Max Drawdown:** 5-15%
- **Sharpe Ratio:** 1.0-2.0
- **Total Return (6 months):** 15-30%

Results may vary based on:
- Market conditions
- Data period
- Parameter ranges
- Initial balance

---

**Remember:** Optimization improves historical performance but doesn't guarantee future results. Always forward test thoroughly before live deployment.
