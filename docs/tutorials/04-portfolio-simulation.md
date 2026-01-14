---
layout: default
title: "Tutorial 4: Portfolio Simulation"
---

# Tutorial 4: Portfolio Simulation

Run Monte Carlo simulations for portfolio analytics.

---

## Step 1: Configure Portfolio

```python
from privatecredit.data import LoanTapeGenerator
from privatecredit.models import PortfolioAggregator, WaterfallConfig

# Generate portfolio
generator = LoanTapeGenerator(
    n_loans=10000,
    n_months=60,
    asset_class_weights={
        'corporate': 0.40,
        'consumer': 0.25,
        'realestate': 0.25,
        'receivables': 0.10
    },
    random_seed=42
)

loans_df, _ = generator.generate_static_features()

print(f"Portfolio size: ${loans_df['original_balance'].sum():,.0f}")
print(f"Number of loans: {len(loans_df)}")
print(f"Average loan: ${loans_df['original_balance'].mean():,.0f}")
```

---

## Step 2: Configure Waterfall

```python
# CLO-style waterfall structure
waterfall_config = WaterfallConfig(
    tranches=[
        {'name': 'AAA', 'size': 0.60, 'coupon': 0.045, 'priority': 1},
        {'name': 'AA', 'size': 0.10, 'coupon': 0.055, 'priority': 2},
        {'name': 'A', 'size': 0.08, 'coupon': 0.065, 'priority': 3},
        {'name': 'BBB', 'size': 0.07, 'coupon': 0.080, 'priority': 4},
        {'name': 'BB', 'size': 0.05, 'coupon': 0.100, 'priority': 5},
        {'name': 'Equity', 'size': 0.10, 'coupon': None, 'priority': 6},
    ],
    oc_trigger=1.20,
    ic_trigger=1.05,
    reinvestment_period=24,
    management_fee=0.005
)

aggregator = PortfolioAggregator(waterfall_config)
```

---

## Step 3: Run Monte Carlo Simulation

```python
import torch
from privatecredit.models import MacroVAE, TransitionTransformer, LoanTrajectoryModel

# Load trained models
checkpoint = torch.load('models/trained_models.pt')
macro_vae = MacroVAE(checkpoint['configs']['macro_vae'])
macro_vae.load_state_dict(checkpoint['macro_vae'])
# ... load other models

# Run simulation
n_simulations = 10000

results = aggregator.monte_carlo_simulate(
    loans_df=loans_df,
    macro_vae=macro_vae,
    transition_model=transition_model,
    trajectory_model=trajectory_model,
    n_simulations=n_simulations,
    scenario_mix={'baseline': 0.6, 'adverse': 0.3, 'severe': 0.1}
)

print(f"Completed {n_simulations} simulations")
```

---

## Step 4: Analyze Loss Distribution

```python
import matplotlib.pyplot as plt
import numpy as np

# Portfolio loss distribution
losses = results.portfolio_losses

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(losses * 100, bins=50, density=True, alpha=0.7, color='steelblue')
axes[0].axvline(results.var_99 * 100, color='red', linestyle='--', label=f'VaR 99%: {results.var_99:.1%}')
axes[0].axvline(results.cvar_99 * 100, color='orange', linestyle='--', label=f'CVaR 99%: {results.cvar_99:.1%}')
axes[0].set_xlabel('Portfolio Loss (%)')
axes[0].set_ylabel('Density')
axes[0].set_title('Loss Distribution')
axes[0].legend()

# CDF
sorted_losses = np.sort(losses)
cdf = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
axes[1].plot(sorted_losses * 100, cdf * 100)
axes[1].axhline(99, color='red', linestyle='--', alpha=0.5)
axes[1].axvline(results.var_99 * 100, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Portfolio Loss (%)')
axes[1].set_ylabel('Cumulative Probability (%)')
axes[1].set_title('Loss CDF')

plt.tight_layout()
plt.savefig('loss_distribution.pdf')
```

---

## Step 5: Calculate Risk Metrics

```python
# Key risk metrics
print("=== Portfolio Risk Metrics ===")
print(f"Expected Loss:     {results.expected_loss:.2%}")
print(f"Loss Volatility:   {results.loss_std:.2%}")
print(f"VaR 95%:           {results.var_95:.2%}")
print(f"VaR 99%:           {results.var_99:.2%}")
print(f"VaR 99.9%:         {results.var_999:.2%}")
print(f"CVaR 99%:          {results.cvar_99:.2%}")
print(f"Max Observed Loss: {results.max_loss:.2%}")
```

---

## Step 6: Tranche-Level Analysis

```python
# Tranche returns
print("\n=== Tranche Returns ===")
print(f"{'Tranche':<10} {'IRR':<10} {'Yield':<10} {'Loss Rate':<12} {'WAL':<8}")
print("-" * 50)

for tranche in results.tranche_results:
    print(f"{tranche.name:<10} "
          f"{tranche.irr:.2%}    "
          f"{tranche.yield_:.2%}    "
          f"{tranche.loss_rate:.2%}      "
          f"{tranche.wal:.1f}")

# Tranche loss probability
print("\n=== Tranche Loss Probability ===")
for tranche in results.tranche_results:
    print(f"{tranche.name}: P(Loss > 0) = {tranche.prob_any_loss:.2%}, "
          f"P(Total Loss) = {tranche.prob_total_loss:.4%}")
```

---

## Step 7: Cashflow Projections

```python
# Monthly cashflow projections
cashflows = results.aggregate_cashflows()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Interest collections
axes[0, 0].plot(cashflows['collected_interest'].mean(axis=0) / 1e6)
axes[0, 0].fill_between(
    range(60),
    cashflows['collected_interest'].quantile(0.05, axis=0) / 1e6,
    cashflows['collected_interest'].quantile(0.95, axis=0) / 1e6,
    alpha=0.3
)
axes[0, 0].set_title('Interest Collections ($ millions)')
axes[0, 0].set_xlabel('Month')

# Principal collections
axes[0, 1].plot(cashflows['collected_principal'].mean(axis=0) / 1e6)
axes[0, 1].fill_between(
    range(60),
    cashflows['collected_principal'].quantile(0.05, axis=0) / 1e6,
    cashflows['collected_principal'].quantile(0.95, axis=0) / 1e6,
    alpha=0.3
)
axes[0, 1].set_title('Principal Collections ($ millions)')
axes[0, 1].set_xlabel('Month')

# Defaults
axes[1, 0].plot(cashflows['defaults'].mean(axis=0) / 1e6)
axes[1, 0].set_title('Defaults ($ millions)')
axes[1, 0].set_xlabel('Month')

# Cumulative losses
axes[1, 1].plot(cashflows['losses'].cumsum(axis=1).mean(axis=0) / 1e6)
axes[1, 1].set_title('Cumulative Losses ($ millions)')
axes[1, 1].set_xlabel('Month')

plt.tight_layout()
plt.savefig('cashflow_projections.pdf')
```

---

## Step 8: Tranche Waterfall Visualization

```python
# Visualize waterfall mechanics
from privatecredit.evaluation import plot_waterfall

fig = plot_waterfall(
    results,
    month=36,  # Show waterfall at month 36
    scenario='mean'
)
plt.savefig('waterfall_month36.pdf')

# Animate waterfall over time
from privatecredit.evaluation import animate_waterfall
animate_waterfall(results, output='waterfall_animation.mp4')
```

---

## Step 9: Export Results

```python
# Export to Excel
results.to_excel('simulation_results.xlsx', include_paths=False)

# Export detailed paths (large file)
results.to_parquet('simulation_paths.parquet')

# Summary statistics
summary = results.summary()
print(summary.to_markdown())
```

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Expected Loss | 2.3% |
| VaR 99% | 5.8% |
| CVaR 99% | 7.2% |
| AAA Tranche IRR | 4.5% |
| Equity Tranche IRR | 12.8% |
| Equity Loss Probability | 15.3% |

---

## Next Steps

- [Back to Tutorials](index.html)
- [API Reference](../api/index.html)
- [Research Methodology](../research/methodology.html)
