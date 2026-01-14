---
layout: default
title: "Tutorial 1: Data Generation"
---

# Tutorial 1: Data Generation

Learn to generate synthetic loan portfolios and macro scenarios.

---

## Step 1: Configure Loan Generator

```python
from privatecredit.data import LoanTapeGenerator

generator = LoanTapeGenerator(
    n_loans=10000,
    n_months=60,
    n_vintages=24,
    asset_class_weights={
        'corporate': 0.40,
        'consumer': 0.25,
        'realestate': 0.25,
        'receivables': 0.10
    },
    random_seed=42
)
```

---

## Step 2: Generate Loan Tape

```python
# Generate static features
loans_df = generator.generate_static_features()

print(loans_df.columns.tolist())
# ['loan_id', 'origination_date', 'maturity_date', 'original_balance',
#  'interest_rate', 'rate_type', 'asset_class', 'ltv_origination', ...]

print(loans_df['asset_class'].value_counts())
# corporate      4000
# consumer       2500
# realestate     2500
# receivables    1000
```

---

## Step 3: Generate Macro Scenarios

```python
from privatecredit.data import MacroScenarioGenerator

macro_gen = MacroScenarioGenerator(
    n_months=60,
    start_date='2020-01-01'
)

# Generate different scenarios
baseline = macro_gen.generate_scenario('baseline')
adverse = macro_gen.generate_scenario('adverse')
severe = macro_gen.generate_scenario('severely_adverse')

print(baseline.columns.tolist())
# ['reporting_month', 'gdp_growth_yoy', 'unemployment_rate', 'inflation_rate',
#  'policy_rate', 'yield_10y', 'credit_spread_ig', 'credit_spread_hy', ...]
```

---

## Step 4: Generate Full Panel

```python
# Generate complete loan tape with performance history
loans_df, panel_df = generator.generate(macro_df=baseline)

print(f"Loans: {len(loans_df)}")
print(f"Loan-month observations: {len(panel_df)}")

# Check final states
final_states = panel_df.groupby('loan_id')['loan_state'].last()
print(final_states.value_counts())
# performing    7500
# prepaid       1200
# default        800
# matured        500
```

---

## Step 5: Explore Asset Class Parameters

```python
from privatecredit.data import ASSET_CONFIGS

for name, config in ASSET_CONFIGS.items():
    print(f"\n{name.upper()}")
    print(f"  Balance: {config.balance_range}")
    print(f"  Rate: {config.rate_range}")
    print(f"  Default rate: {config.annual_default_rate}")
    print(f"  LGD: {config.lgd_range}")
```

---

## Next Steps

- [Tutorial 2: Training Models](02-training-models.html)
- [Back to Tutorials](index.html)
