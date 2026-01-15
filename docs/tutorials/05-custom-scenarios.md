---
layout: default
title: Custom Scenarios
parent: Tutorials
nav_order: 5
---

# Tutorial 5: Creating Custom Macro Scenarios

Learn how to create user-defined macro scenarios, interpolate between standard scenarios, and replay historical data.

## Prerequisites

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from privatecredit.data import MacroScenarioGenerator
```

## 1. Understanding Scenario Structure

Each macro scenario contains 9 variables over 60 months:

| Variable | Description | Baseline Range |
|----------|-------------|----------------|
| `gdp_growth_yoy` | Year-over-year GDP growth | 1-4% |
| `unemployment_rate` | Unemployment rate | 3-6% |
| `inflation_yoy` | Year-over-year inflation | 1-3% |
| `fed_funds_rate` | Federal funds rate | 2-5% |
| `treasury_10y` | 10-year Treasury yield | 2-4% |
| `credit_spread_ig` | Investment grade spread | 80-150 bps |
| `credit_spread_hy` | High yield spread | 300-500 bps |
| `property_index` | Commercial property index | 95-110 |
| `equity_return` | Equity market return | -5% to +20% |

## 2. Using the Standard Generator

### Generate Built-in Scenarios

```python
# Initialize generator
generator = MacroScenarioGenerator(
    n_months=60,
    start_date='2024-01-01',
    seed=42
)

# Generate standard scenarios
baseline = generator.generate_scenario('baseline')
adverse = generator.generate_scenario('adverse')
severe = generator.generate_scenario('severely_adverse')
stagflation = generator.generate_scenario('stagflation')

print(f"Scenario shape: {baseline.shape}")
print(f"Variables: {list(baseline.columns)}")
```

### View Scenario Summary

```python
def scenario_summary(scenario, name):
    """Print summary statistics for a scenario."""
    print(f"\n{name.upper()} Scenario:")
    print(f"  GDP Growth: {scenario['gdp_growth_yoy'].mean():.2%}")
    print(f"  Unemployment: {scenario['unemployment_rate'].mean():.1%}")
    print(f"  Inflation: {scenario['inflation_yoy'].mean():.2%}")
    print(f"  HY Spread: {scenario['credit_spread_hy'].mean():.0f} bps")

for name, scen in [('Baseline', baseline), ('Adverse', adverse),
                    ('Severe', severe), ('Stagflation', stagflation)]:
    scenario_summary(scen, name)
```

## 3. Creating Custom Scenarios

### Method 1: Parameter Override

```python
# Create custom scenario with specific parameters
custom_params = {
    'gdp_growth_mean': 0.01,      # 1% GDP growth
    'unemployment_peak': 0.08,    # Peak at 8%
    'inflation_mean': 0.035,      # 3.5% inflation
    'hy_spread_peak': 800,        # Peak spread 800 bps
    'duration_stress': 18         # Stress lasts 18 months
}

custom_scenario = generator.generate_custom_scenario(**custom_params)
```

### Method 2: Direct DataFrame Construction

```python
def create_manual_scenario(n_months=60):
    """Create a scenario manually with specific paths."""
    months = pd.date_range('2024-01-01', periods=n_months, freq='M')

    # Define time-varying paths
    t = np.linspace(0, 1, n_months)

    scenario = pd.DataFrame({
        'date': months,
        # V-shaped recovery
        'gdp_growth_yoy': 0.025 - 0.05 * np.sin(np.pi * t) * np.exp(-2*t),
        # Rising then falling unemployment
        'unemployment_rate': 0.04 + 0.04 * np.sin(np.pi * t * 0.8),
        # Persistent inflation
        'inflation_yoy': 0.03 + 0.02 * (1 - np.exp(-3*t)),
        # Rate hike cycle
        'fed_funds_rate': 0.025 + 0.025 * np.minimum(t * 2, 1),
        # 10Y follows short rates with lag
        'treasury_10y': 0.03 + 0.015 * np.minimum(t * 1.5, 1),
        # Credit spreads widen then normalize
        'credit_spread_ig': 100 + 100 * np.sin(np.pi * t) * np.exp(-2*t),
        'credit_spread_hy': 400 + 400 * np.sin(np.pi * t) * np.exp(-1.5*t),
        # Property decline
        'property_index': 100 - 10 * np.sin(np.pi * t * 0.6),
        # Equity volatility
        'equity_return': 0.05 - 0.15 * np.sin(np.pi * t * 0.5) * np.exp(-1.5*t)
    })

    return scenario.set_index('date')

custom = create_manual_scenario()
```

### Method 3: Scenario Templates

```python
class ScenarioTemplate:
    """Reusable scenario templates."""

    @staticmethod
    def rate_shock(n_months=60, shock_size=0.02):
        """Interest rate shock scenario."""
        t = np.linspace(0, 1, n_months)
        base = MacroScenarioGenerator(n_months=n_months).generate_scenario('baseline')

        # Apply rate shock
        base['fed_funds_rate'] += shock_size * (1 - np.exp(-5*t))
        base['treasury_10y'] += shock_size * 0.8 * (1 - np.exp(-3*t))

        # Ripple effects
        base['gdp_growth_yoy'] -= shock_size * 0.5 * np.exp(-t)
        base['property_index'] -= shock_size * 500 * (1 - np.exp(-2*t))

        return base

    @staticmethod
    def credit_crunch(n_months=60, spread_peak=1000):
        """Credit market stress scenario."""
        t = np.linspace(0, 1, n_months)
        base = MacroScenarioGenerator(n_months=n_months).generate_scenario('baseline')

        # Credit spread spike
        spread_path = spread_peak * np.sin(np.pi * t * 0.6) * np.exp(-t)
        base['credit_spread_hy'] += spread_path
        base['credit_spread_ig'] += spread_path * 0.3

        # Economic impact
        base['gdp_growth_yoy'] -= 0.02 * np.sin(np.pi * t * 0.6)

        return base

# Use templates
rate_shock = ScenarioTemplate.rate_shock(shock_size=0.03)
credit_crunch = ScenarioTemplate.credit_crunch(spread_peak=800)
```

## 4. Interpolating Between Scenarios

### Linear Interpolation

```python
def interpolate_scenarios(scenario1, scenario2, weight):
    """
    Linearly interpolate between two scenarios.

    Args:
        scenario1: First scenario DataFrame
        scenario2: Second scenario DataFrame
        weight: Weight for scenario2 (0 = scenario1, 1 = scenario2)
    """
    return scenario1 * (1 - weight) + scenario2 * weight

# Create intermediate scenario (30% adverse, 70% baseline)
mild_stress = interpolate_scenarios(baseline, adverse, weight=0.3)

# Verify
print(f"Baseline GDP: {baseline['gdp_growth_yoy'].mean():.2%}")
print(f"Adverse GDP: {adverse['gdp_growth_yoy'].mean():.2%}")
print(f"Mild Stress GDP: {mild_stress['gdp_growth_yoy'].mean():.2%}")
```

### Scenario Blending Over Time

```python
def time_varying_blend(scenario1, scenario2, transition_start, transition_end):
    """
    Blend scenarios with time-varying weights.

    Starts in scenario1, transitions to scenario2.
    """
    n_months = len(scenario1)
    weights = np.zeros(n_months)

    # Create smooth transition weights
    for i in range(n_months):
        if i < transition_start:
            weights[i] = 0
        elif i >= transition_end:
            weights[i] = 1
        else:
            # Smooth cosine transition
            progress = (i - transition_start) / (transition_end - transition_start)
            weights[i] = 0.5 * (1 - np.cos(np.pi * progress))

    # Apply weights
    blended = scenario1.copy()
    for col in blended.columns:
        blended[col] = scenario1[col] * (1 - weights) + scenario2[col] * weights

    return blended, weights

# Start baseline, transition to adverse over months 12-24
blended, weights = time_varying_blend(baseline, adverse,
                                      transition_start=12,
                                      transition_end=24)
```

## 5. Historical Scenario Replay

### Load Historical Data

```python
def load_historical_data(start_date, end_date):
    """
    Load historical macro data from FRED (simulated here).

    In production, use fredapi or similar.
    """
    # Simulated historical data
    dates = pd.date_range(start_date, end_date, freq='M')
    n = len(dates)

    np.random.seed(123)

    # 2008-2009 style recession
    historical = pd.DataFrame({
        'date': dates,
        'gdp_growth_yoy': np.concatenate([
            np.linspace(0.02, -0.04, n//3),
            np.linspace(-0.04, 0.02, n//3),
            np.linspace(0.02, 0.025, n - 2*(n//3))
        ]) + np.random.normal(0, 0.003, n),
        'unemployment_rate': np.concatenate([
            np.linspace(0.05, 0.10, n//2),
            np.linspace(0.10, 0.06, n - n//2)
        ]) + np.random.normal(0, 0.002, n),
        # ... other variables
    }).set_index('date')

    return historical

# Load and extend historical scenario
historical = load_historical_data('2007-01-01', '2011-12-31')
```

### Extend Historical to 60 Months

```python
def extend_scenario(historical_data, target_months=60):
    """
    Extend historical scenario to target length using mean reversion.
    """
    current_months = len(historical_data)
    if current_months >= target_months:
        return historical_data.iloc[:target_months]

    # Mean reversion parameters
    long_run_means = {
        'gdp_growth_yoy': 0.025,
        'unemployment_rate': 0.045,
        'inflation_yoy': 0.02,
        # ...
    }

    reversion_speed = 0.1

    # Extend each variable
    extended = historical_data.copy()
    last_values = historical_data.iloc[-1]

    for month in range(current_months, target_months):
        new_row = {}
        for col in extended.columns:
            if col in long_run_means:
                prev = extended.iloc[-1][col]
                mean = long_run_means[col]
                # AR(1) with mean reversion
                new_row[col] = prev + reversion_speed * (mean - prev) + np.random.normal(0, 0.002)
            else:
                new_row[col] = extended.iloc[-1][col]

        new_date = extended.index[-1] + pd.DateOffset(months=1)
        extended.loc[new_date] = new_row

    return extended
```

## 6. Scenario Validation

### Check Correlations

```python
def validate_correlations(scenario):
    """
    Validate that macro variables have reasonable correlations.
    """
    expected = {
        ('gdp_growth_yoy', 'unemployment_rate'): (-0.8, -0.4),
        ('gdp_growth_yoy', 'credit_spread_hy'): (-0.7, -0.2),
        ('inflation_yoy', 'fed_funds_rate'): (0.3, 0.8),
    }

    print("Correlation Validation:")
    for (var1, var2), (low, high) in expected.items():
        if var1 in scenario.columns and var2 in scenario.columns:
            corr = scenario[var1].corr(scenario[var2])
            status = "PASS" if low <= corr <= high else "WARN"
            print(f"  {var1} vs {var2}: {corr:.3f} [{status}]")

validate_correlations(custom)
```

### Check Economic Plausibility

```python
def check_plausibility(scenario):
    """
    Check if scenario values are economically plausible.
    """
    checks = [
        ('unemployment_rate', 0.02, 0.15, 'Unemployment'),
        ('inflation_yoy', -0.02, 0.10, 'Inflation'),
        ('credit_spread_hy', 200, 2000, 'HY Spread'),
    ]

    print("\nPlausibility Checks:")
    for col, min_val, max_val, name in checks:
        if col in scenario.columns:
            actual_min = scenario[col].min()
            actual_max = scenario[col].max()

            if actual_min < min_val or actual_max > max_val:
                print(f"  {name}: [{actual_min:.3f}, {actual_max:.3f}] - WARNING")
            else:
                print(f"  {name}: [{actual_min:.3f}, {actual_max:.3f}] - OK")

check_plausibility(custom)
```

## 7. Visualization

```python
def plot_scenario_comparison(scenarios, names, variables=None):
    """
    Plot multiple scenarios side by side.
    """
    if variables is None:
        variables = ['gdp_growth_yoy', 'unemployment_rate',
                    'inflation_yoy', 'credit_spread_hy']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, var in zip(axes.flat, variables):
        for scenario, name in zip(scenarios, names):
            ax.plot(scenario[var], label=name)
        ax.set_title(var.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_scenario_comparison(
    [baseline, custom, adverse],
    ['Baseline', 'Custom', 'Adverse']
)
```

## Summary

| Method | Use Case | Complexity |
|--------|----------|------------|
| Parameter Override | Quick variations | Low |
| Manual DataFrame | Full control | Medium |
| Templates | Reusable patterns | Medium |
| Interpolation | Scenario gradations | Low |
| Historical Replay | Backtesting | High |

**Next:** [Calibration Guide](06-calibration-guide.md)
