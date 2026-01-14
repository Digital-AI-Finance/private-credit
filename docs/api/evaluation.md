---
layout: default
title: Evaluation API
---

# Evaluation API

Metrics, visualization, and reporting utilities.

---

## Metrics

### calculate_metrics()

Calculate standard regression metrics.

```python
from privatecredit.evaluation import calculate_metrics

metrics = calculate_metrics(
    predicted=predicted_losses,
    actual=actual_losses
)

print(metrics)
# {'rmse': 0.015, 'mae': 0.012, 'r2': 0.85, 'mape': 8.5}
```

**Returns:**
- `rmse`: Root mean squared error
- `mae`: Mean absolute error
- `r2`: R-squared score
- `mape`: Mean absolute percentage error

### calculate_transition_accuracy()

Evaluate transition probability predictions.

```python
from privatecredit.evaluation import calculate_transition_accuracy

accuracy = calculate_transition_accuracy(
    predicted_P=predicted_matrices,
    actual_transitions=observed_counts
)

print(accuracy)
# {'kl_divergence': 0.05, 'total_variation': 0.03}
```

---

## Loss Distribution Metrics

### calculate_risk_metrics()

Calculate portfolio risk metrics from loss distribution.

```python
from privatecredit.evaluation import calculate_risk_metrics

metrics = calculate_risk_metrics(
    losses=loss_samples,
    confidence_levels=[0.95, 0.99, 0.999]
)

print(metrics)
# {
#   'expected_loss': 0.023,
#   'loss_std': 0.018,
#   'var_95': 0.045,
#   'var_99': 0.062,
#   'var_999': 0.095,
#   'cvar_95': 0.055,
#   'cvar_99': 0.078,
#   'cvar_999': 0.112
# }
```

### calculate_tail_metrics()

Analyze tail behavior of loss distribution.

```python
from privatecredit.evaluation import calculate_tail_metrics

tail = calculate_tail_metrics(losses, threshold=0.05)

print(tail)
# {'exceedance_prob': 0.08, 'mean_excess': 0.025, 'tail_index': 2.5}
```

---

## Visualization

### plot_loss_distribution()

Plot portfolio loss distribution with risk metrics.

```python
from privatecredit.evaluation import plot_loss_distribution

fig = plot_loss_distribution(
    losses=loss_samples,
    var_levels=[0.95, 0.99],
    title='Portfolio Loss Distribution'
)
fig.savefig('loss_dist.pdf')
```

### plot_scenario_comparison()

Compare loss distributions across scenarios.

```python
from privatecredit.evaluation import plot_scenario_comparison

fig = plot_scenario_comparison(
    scenarios={
        'Baseline': baseline_losses,
        'Adverse': adverse_losses,
        'Severe': severe_losses
    }
)
fig.savefig('scenario_comparison.pdf')
```

### plot_transition_matrices()

Visualize transition probability matrices.

```python
from privatecredit.evaluation import plot_transition_matrices

fig = plot_transition_matrices(
    P_matrices=transition_matrices,
    months=[0, 12, 24, 36],
    state_names=['Perf', '30', '60', '90', 'Def', 'Pre', 'Mat']
)
fig.savefig('transition_matrices.pdf')
```

### plot_macro_paths()

Visualize generated macro scenarios.

```python
from privatecredit.evaluation import plot_macro_paths

fig = plot_macro_paths(
    paths=macro_paths,
    variables=['gdp_growth_yoy', 'unemployment_rate', 'credit_spread_hy'],
    confidence_band=0.90
)
fig.savefig('macro_paths.pdf')
```

### plot_waterfall()

Visualize waterfall cashflow distribution.

```python
from privatecredit.evaluation import plot_waterfall

fig = plot_waterfall(
    results=simulation_results,
    month=36,
    scenario='mean'
)
fig.savefig('waterfall.pdf')
```

---

## Reporting

### StressTestReport

Generate comprehensive stress test reports.

```python
from privatecredit.evaluation import StressTestReport

report = StressTestReport(
    portfolio=loans_df,
    scenarios=scenario_definitions,
    results=stress_results,
    sensitivities=sensitivity_analysis
)

# Generate PDF report
report.generate_pdf('stress_test_report.pdf')

# Generate HTML report
report.generate_html('stress_test_report.html')

# Export data to Excel
report.to_excel('stress_test_data.xlsx')
```

### TrancheAnalysisReport

Generate tranche-level analysis report.

```python
from privatecredit.evaluation import TrancheAnalysisReport

report = TrancheAnalysisReport(
    simulation_results=results,
    waterfall_config=waterfall_config
)

report.generate_pdf('tranche_analysis.pdf')
```

---

## Reverse Stress Testing

### reverse_stress_test()

Find scenarios that produce target loss levels.

```python
from privatecredit.evaluation import reverse_stress_test

scenario = reverse_stress_test(
    macro_vae=macro_vae,
    transition_model=transition_model,
    trajectory_model=trajectory_model,
    loans_df=loans_df,
    target_loss=0.10,  # 10% target loss
    n_iterations=100,
    tolerance=0.005
)

print(scenario)
# {'gdp_growth_yoy': -0.05, 'unemployment_rate': 0.12, ...}
```

---

## Model Validation

### validate_macro_vae()

Validate macro VAE generation quality.

```python
from privatecredit.evaluation import validate_macro_vae

validation = validate_macro_vae(
    model=macro_vae,
    historical_data=historical_macro,
    n_samples=1000
)

print(validation)
# {
#   'mean_error': 0.002,
#   'covariance_error': 0.05,
#   'autocorrelation_error': 0.08,
#   'scenario_separation': 0.92
# }
```

### validate_transitions()

Validate transition model predictions.

```python
from privatecredit.evaluation import validate_transitions

validation = validate_transitions(
    model=transition_model,
    observed_transitions=test_transitions,
    macro_data=test_macro
)

print(validation)
# {'accuracy': 0.85, 'calibration_error': 0.03}
```

### validate_trajectories()

Validate loan trajectory generation.

```python
from privatecredit.evaluation import validate_trajectories

validation = validate_trajectories(
    model=trajectory_model,
    observed_panel=test_panel,
    n_samples=100
)

print(validation)
# {'state_accuracy': 0.88, 'payment_rmse': 50.2, 'balance_rmse': 1200.5}
```

---

## Backtesting

### backtest_portfolio()

Run historical backtest.

```python
from privatecredit.evaluation import backtest_portfolio

results = backtest_portfolio(
    model_suite={'macro_vae': ..., 'transition': ..., 'trajectory': ...},
    historical_loans=historical_loans_df,
    historical_panel=historical_panel_df,
    historical_macro=historical_macro_df,
    test_periods=24  # Hold out last 24 months
)

print(results.summary())
```

---

[Back to API Reference](index.html)
