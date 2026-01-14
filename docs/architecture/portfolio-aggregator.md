---
layout: default
title: Portfolio Aggregator
---

# Portfolio Aggregator

Differentiable waterfall simulation for portfolio and tranche-level analytics.

---

## Architecture

```
Loan Trajectories (from Loan Trajectory Model)
    |
    v
Cashflow Aggregation
    |
    v
Waterfall Engine
    |
    v
Tranche Cashflows & Returns
    |
    v
Loss Distribution & Risk Metrics
```

---

## Waterfall Structure

Typical CLO/SPV waterfall:

```
Collections (Interest + Principal)
    |
    +---> Senior Management Fees
    |
    +---> Senior Tranche Interest
    |
    +---> Senior Tranche Principal (if OC test fails)
    |
    +---> Mezzanine Tranche Interest
    |
    +---> Mezzanine Tranche Principal
    |
    +---> Junior Tranche Interest
    |
    +---> Subordinated Fees
    |
    +---> Residual to Equity
```

---

## Tranche Configuration

```python
from privatecredit.models import PortfolioAggregator, WaterfallConfig

tranches = [
    {'name': 'Senior', 'size': 0.70, 'coupon': 0.05, 'priority': 1},
    {'name': 'Mezz', 'size': 0.15, 'coupon': 0.08, 'priority': 2},
    {'name': 'Junior', 'size': 0.10, 'coupon': 0.12, 'priority': 3},
    {'name': 'Equity', 'size': 0.05, 'coupon': None, 'priority': 4},
]

config = WaterfallConfig(
    tranches=tranches,
    oc_trigger=1.20,      # OC test threshold
    ic_trigger=1.05,      # IC test threshold
    reinvestment_period=24  # Months
)

aggregator = PortfolioAggregator(config)
```

---

## Portfolio Aggregation

From individual loan trajectories to portfolio cashflows:

```python
# Aggregate loan trajectories
portfolio_cf = aggregator.aggregate_cashflows(
    trajectories=loan_trajectories,
    loan_features=loans_df
)

print(portfolio_cf.columns)
# ['month', 'scheduled_interest', 'scheduled_principal',
#  'collected_interest', 'collected_principal',
#  'defaults', 'recoveries', 'losses']
```

---

## Monte Carlo Simulation

Run full simulation for loss distribution:

```python
results = aggregator.simulate(
    loan_trajectories=trajectories,  # (n_simulations, n_loans, n_months)
    n_simulations=10000
)

# Portfolio-level metrics
print(f"Expected Loss: {results.expected_loss:.2%}")
print(f"VaR 99%: {results.var_99:.2%}")
print(f"CVaR 99%: {results.cvar_99:.2%}")

# Tranche-level returns
for tranche in results.tranche_returns:
    print(f"{tranche.name}: IRR = {tranche.irr:.2%}")
```

---

## Coverage Tests

The waterfall implements standard CLO tests:

| Test | Formula | Trigger |
|------|---------|---------|
| OC Test | Par Value / Tranche Balance | < 1.20 |
| IC Test | Interest Collections / Interest Due | < 1.05 |

When tests fail, cashflows are redirected to senior tranches.

---

## Loss Distribution

The aggregator estimates the full loss distribution:

```python
import matplotlib.pyplot as plt

losses = results.portfolio_losses
plt.hist(losses, bins=50, density=True)
plt.axvline(results.var_99, color='r', label='VaR 99%')
plt.axvline(results.cvar_99, color='orange', label='CVaR 99%')
plt.xlabel('Portfolio Loss Rate')
plt.ylabel('Density')
plt.legend()
```

---

## Differentiability

The entire waterfall is differentiable via soft approximations:

```
Hard: if x > threshold then action
Soft: sigmoid((x - threshold) / temperature) * action
```

This enables end-to-end training with portfolio-level objectives:

```python
# Joint training with tranche return targets
loss = model.compute_loss(
    predicted_returns=tranche_returns,
    target_returns=historical_returns
)
loss.backward()
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| Expected Loss | Mean of loss distribution |
| VaR | Value at Risk at confidence level |
| CVaR | Conditional VaR (Expected Shortfall) |
| Tranche IRR | Internal rate of return per tranche |
| Attachment/Detachment | Loss levels where tranche is affected |
| WAL | Weighted average life |

---

[Back to Architecture](index.html)
