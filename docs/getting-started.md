---
layout: default
title: Getting Started
---

# Getting Started

Get up and running with Private Credit in 5 minutes.

---

## Installation

```bash
pip install privatecredit
```

Or install from source:

```bash
git clone https://github.com/Digital-AI-Finance/private-credit.git
cd private-credit
pip install -e .
```

---

## Quick Example

### 1. Generate Synthetic Data

```python
from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator

# Generate loan portfolio
generator = LoanTapeGenerator(
    n_loans=10000,
    n_months=60,
    n_vintages=24,
    random_seed=42
)

# Generate loan tape and monthly panel
loans_df, panel_df = generator.generate()

print(f"Generated {len(loans_df)} loans")
print(f"Asset class distribution:")
print(loans_df['asset_class'].value_counts())
```

**Output:**
```
Generated 10000 loans
Asset class distribution:
corporate      4000
consumer       2500
realestate     2500
receivables    1000
```

### 2. Generate Macro Scenarios

```python
# Generate macro scenarios
macro_gen = MacroScenarioGenerator(n_months=60)

baseline = macro_gen.generate_scenario('baseline')
adverse = macro_gen.generate_scenario('adverse')
severe = macro_gen.generate_scenario('severely_adverse')

print(f"Baseline GDP growth: {baseline['gdp_growth_yoy'].mean():.2%}")
print(f"Adverse unemployment: {adverse['unemployment_rate'].max():.2%}")
```

### 3. Run Baseline Model

```python
from privatecredit.models import MarkovTransitionModel, PortfolioLossModel

# Create Markov model
markov = MarkovTransitionModel(asset_class='corporate')

# Compute PD curve
pd_curve = markov.compute_pd_curve(horizon_months=60)
print(f"12-month PD: {pd_curve[11]:.2%}")
print(f"Lifetime PD: {pd_curve[-1]:.2%}")

# Simulate portfolio loss
loss_model = PortfolioLossModel(markov)
results = loss_model.simulate_portfolio_loss(
    loans_df.head(1000),
    baseline,
    n_simulations=500
)

print(f"Expected Loss: {results['expected_loss_pct']:.2%}")
print(f"VaR 99%: EUR {results['var_99']:,.0f}")
```

---

## Next Steps

1. **[Tutorials](tutorials/)** - Step-by-step guides for each component
2. **[Architecture](architecture/)** - Understand the model hierarchy
3. **[API Reference](api/)** - Complete API documentation

---

## Example Notebooks

Interactive Jupyter notebooks are available in the `notebooks/` directory:

- `01_data_exploration.ipynb` - Explore synthetic data
- `02_model_training.ipynb` - Train deep generative models
- `03_scenario_analysis.ipynb` - Stress testing and scenarios

---

[Back to Home](index.html)
