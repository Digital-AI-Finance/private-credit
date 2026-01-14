---
layout: default
title: Data API
---

# Data API

Data generation and processing utilities.

---

## LoanTapeGenerator

Generate synthetic loan portfolios with realistic characteristics.

### Constructor

```python
LoanTapeGenerator(
    n_loans: int = 10000,
    n_months: int = 60,
    n_vintages: int = 24,
    asset_class_weights: dict = None,
    random_seed: int = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_loans | int | 10000 | Number of loans to generate |
| n_months | int | 60 | Loan lifetime in months |
| n_vintages | int | 24 | Number of vintage cohorts |
| asset_class_weights | dict | Equal weights | Distribution across asset classes |
| random_seed | int | None | Random seed for reproducibility |

### Methods

#### generate_static_features()

Generate loan-level static features only.

```python
loans_df = generator.generate_static_features()
```

**Returns:** `pd.DataFrame` with columns:
- `loan_id`: Unique identifier
- `origination_date`: Loan start date
- `maturity_date`: Contractual end date
- `original_balance`: Initial loan amount
- `interest_rate`: Contractual rate
- `rate_type`: Fixed or floating
- `asset_class`: Corporate, consumer, realestate, receivables
- `ltv_origination`: Loan-to-value ratio
- `vintage_month`: Origination cohort

#### generate()

Generate complete loan tape with monthly panel.

```python
loans_df, panel_df = generator.generate(macro_df=macro_data)
```

**Parameters:**
- `macro_df`: DataFrame with macro scenario (optional)

**Returns:** Tuple of (loans_df, panel_df)
- `panel_df` contains monthly observations with:
  - `loan_id`, `reporting_month`
  - `loan_state`: Current state (0-6)
  - `scheduled_payment`, `actual_payment`
  - `outstanding_balance`
  - `loss_amount` (if defaulted)

---

## MacroScenarioGenerator

Generate macroeconomic time series for different scenarios.

### Constructor

```python
MacroScenarioGenerator(
    n_months: int = 60,
    start_date: str = '2020-01-01'
)
```

### Methods

#### generate_scenario()

Generate a specific scenario.

```python
macro_df = generator.generate_scenario(scenario='baseline')
```

**Parameters:**
- `scenario`: One of 'baseline', 'adverse', 'severely_adverse', 'stagflation'

**Returns:** `pd.DataFrame` with columns:
- `reporting_month`: Date
- `gdp_growth_yoy`: Year-over-year GDP growth
- `unemployment_rate`: Unemployment rate
- `inflation_rate`: CPI inflation
- `policy_rate`: Central bank rate
- `yield_10y`: 10-year government yield
- `credit_spread_ig`: Investment grade spread (bps)
- `credit_spread_hy`: High yield spread (bps)
- `property_index`: Property price index

---

## Asset Class Configurations

### ASSET_CONFIGS

Pre-defined configurations for each asset class.

```python
from privatecredit.data import ASSET_CONFIGS

corporate = ASSET_CONFIGS['corporate']
print(corporate.balance_range)    # (500000, 50000000)
print(corporate.rate_range)       # (0.04, 0.12)
print(corporate.annual_default_rate)  # 0.02
```

### AssetClassConfig

```python
@dataclass
class AssetClassConfig:
    balance_range: Tuple[float, float]
    rate_range: Tuple[float, float]
    term_range: Tuple[int, int]
    ltv_range: Tuple[float, float]
    annual_default_rate: float
    lgd_range: Tuple[float, float]
    prepay_rate: float
```

---

## State Definitions

### LOAN_STATES

```python
LOAN_STATES = {
    0: 'performing',
    1: '30dpd',
    2: '60dpd',
    3: '90dpd',
    4: 'default',
    5: 'prepaid',
    6: 'matured'
}
```

### ABSORBING_STATES

```python
ABSORBING_STATES = {4, 5, 6}  # default, prepaid, matured
```

---

## Utility Functions

### extract_transitions()

Extract transition counts from panel data.

```python
from privatecredit.data import extract_transitions

transitions = extract_transitions(panel_df)
# Returns: Tensor of shape (n_months, n_states, n_states)
```

### prepare_trajectories()

Prepare trajectory tensors for training.

```python
from privatecredit.data import prepare_trajectories

trajectories = prepare_trajectories(panel_df, loans_df)
# Returns: Dict with 'states', 'payments', 'balances'
```

---

## Data Schemas

### Loan Tape Schema

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Unique identifier |
| origination_date | date | Loan start date |
| original_balance | float | Initial principal |
| interest_rate | float | Annual rate |
| rate_type | enum | FIXED, FLOATING |
| asset_class | enum | CORPORATE, CONSUMER, REALESTATE, RECEIVABLES |
| ltv_origination | float | LTV at origination |
| dscr | float | Debt service coverage (commercial) |
| fico_origination | int | Credit score (consumer) |
| geography | string | State/country code |
| industry | string | Industry classification |

### Panel Schema

| Field | Type | Description |
|-------|------|-------------|
| loan_id | string | Loan identifier |
| reporting_month | date | Observation date |
| loan_state | int | Current state (0-6) |
| days_past_due | int | Days delinquent |
| scheduled_payment | float | Contractual payment |
| actual_payment | float | Received payment |
| outstanding_balance | float | Current principal |
| loss_amount | float | Realized loss |

---

[Back to API Reference](index.html)
