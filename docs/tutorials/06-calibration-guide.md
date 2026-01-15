---
layout: default
title: Calibration Guide
parent: Tutorials
nav_order: 6
---

# Tutorial 6: Model Calibration Guide

Learn how to calibrate models to historical data using the calibration module.

## Prerequisites

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from privatecredit.calibration import (
    HistoricalCalibrator,
    TransitionCalibrator,
    ParameterEstimator,
    BayesianEstimator
)
```

## 1. Historical Default Rate Calibration

### Prepare Historical Data

```python
# Generate synthetic historical data (replace with actual data)
np.random.seed(42)
n_periods = 60

# Monthly observations
true_pd = 0.025  # 2.5% annual default rate
exposure_per_period = np.random.poisson(1000, n_periods)
defaults = np.random.binomial(exposure_per_period, true_pd / 12)

print(f"Total periods: {n_periods}")
print(f"Total defaults: {defaults.sum()}")
print(f"Total exposure: {exposure_per_period.sum()}")
print(f"Observed default rate: {defaults.sum() / exposure_per_period.sum():.4f}")
```

### Fit Default Rate Distribution

```python
# Initialize calibrator
calibrator = HistoricalCalibrator()

# Fit to historical data
params = calibrator.fit_default_rates(defaults, exposure_per_period)

print("\nCalibrated Parameters:")
print(f"  Mean default rate: {params['mean_default_rate']:.4f}")
print(f"  Std default rate: {params['std_default_rate']:.4f}")
print(f"  Beta alpha: {params['beta_alpha']:.2f}")
print(f"  Beta beta: {params['beta_beta']:.2f}")
```

### Bootstrap Confidence Intervals

```python
# Compute observed rates
observed_rates = defaults / (exposure_per_period + 1e-10)

# Bootstrap CI
ci = calibrator.bootstrap_confidence_intervals(
    observed_rates,
    statistic='mean',
    n_bootstrap=10000,
    confidence_level=0.95
)

print(f"\n95% Bootstrap CI for mean default rate:")
print(f"  Point estimate: {ci['point_estimate']:.4f}")
print(f"  Lower bound: {ci['lower_bound']:.4f}")
print(f"  Upper bound: {ci['upper_bound']:.4f}")
print(f"  Standard error: {ci['std_error']:.4f}")
```

## 2. Transition Matrix Estimation

### From Cohort Data

```python
# Create cohort transition data
states = ['Performing', '30DPD', '60DPD', '90DPD', 'Default', 'Prepaid', 'Matured']

# Simulated cohort transitions (n_from x n_to counts)
transition_counts = np.array([
    [850, 30, 10, 5, 2, 12, 91],    # From Performing
    [400, 350, 100, 50, 20, 30, 50], # From 30DPD
    [200, 200, 300, 150, 50, 50, 50], # From 60DPD
    [100, 100, 150, 350, 150, 50, 100], # From 90DPD
    [0, 0, 0, 0, 1000, 0, 0],       # From Default (absorbing)
    [0, 0, 0, 0, 0, 1000, 0],       # From Prepaid (absorbing)
    [0, 0, 0, 0, 0, 0, 1000]        # From Matured (absorbing)
])

# Fit transition matrix
trans_calibrator = TransitionCalibrator()
P_mle = trans_calibrator.fit_from_counts(transition_counts)

print("MLE Transition Matrix:")
print(pd.DataFrame(P_mle, index=states, columns=states).round(4))
```

### Calibrate to Target Rates

```python
# Calibrate to hit specific default/prepayment targets
target_default = 0.02  # 2% annual default
target_prepayment = 0.15  # 15% annual prepayment
maturity = 60  # months

P_calibrated = trans_calibrator.fit_to_target_rates(
    target_default_rate=target_default,
    target_prepayment_rate=target_prepayment,
    maturity=maturity
)

# Verify
state_probs = np.zeros(7)
state_probs[0] = 1.0  # Start in Performing

for _ in range(maturity):
    state_probs = state_probs @ P_calibrated

print(f"\nVerification ({maturity}-month projection):")
print(f"  Target default: {target_default:.2%}, Achieved: {state_probs[4]:.2%}")
print(f"  Target prepay: {target_prepayment:.2%}, Achieved: {state_probs[5]:.2%}")
```

### Add Dirichlet Prior (Regularization)

```python
# With prior for smoothing
prior_alpha = np.ones((7, 7)) * 0.1  # Weak uniform prior

P_regularized = trans_calibrator.fit_with_prior(
    transition_counts,
    prior_alpha=prior_alpha
)

print("\nRegularized Transition Matrix:")
print(pd.DataFrame(P_regularized, index=states, columns=states).round(4))
```

## 3. LGD Distribution Fitting

### Maximum Likelihood Estimation

```python
# Generate synthetic LGD data (bimodal: secured vs unsecured)
np.random.seed(42)
lgd_secured = np.random.beta(2, 8, size=300)  # Mean ~20%
lgd_unsecured = np.random.beta(4, 4, size=200)  # Mean ~50%
lgd_data = np.concatenate([lgd_secured, lgd_unsecured])
lgd_data = np.clip(lgd_data, 0.01, 0.99)  # Avoid boundaries

# Fit with MLE
estimator = ParameterEstimator()

# Beta distribution fit
beta_params = estimator.fit_lgd_mle(lgd_data, model='beta')

print("Beta Distribution MLE:")
print(f"  Alpha: {beta_params['alpha']:.3f}")
print(f"  Beta: {beta_params['beta']:.3f}")
print(f"  Mean LGD: {beta_params['mean']:.3f}")
print(f"  Mode LGD: {beta_params['mode']:.3f}")
print(f"  Variance: {beta_params['variance']:.4f}")
```

### Compare Distribution Models

```python
# Try different distributions
models = ['beta', 'kumaraswamy', 'mixture_beta']
results = {}

for model in models:
    params = estimator.fit_lgd_mle(lgd_data, model=model)
    results[model] = params

# Compare via AIC/BIC
print("\nModel Comparison:")
print(f"{'Model':<15} {'AIC':<10} {'BIC':<10} {'Mean':<10}")
print("-" * 45)
for model, params in results.items():
    print(f"{model:<15} {params.get('aic', 'N/A'):<10.1f} "
          f"{params.get('bic', 'N/A'):<10.1f} {params['mean']:<10.3f}")
```

### Plot Fitted Distribution

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of observed data
ax.hist(lgd_data, bins=30, density=True, alpha=0.6, label='Observed', color='steelblue')

# Fitted distributions
x = np.linspace(0.01, 0.99, 100)
for model, params in results.items():
    if model == 'beta':
        pdf = stats.beta.pdf(x, params['alpha'], params['beta'])
        ax.plot(x, pdf, label=f"Beta (a={params['alpha']:.2f}, b={params['beta']:.2f})")

ax.set_xlabel('LGD')
ax.set_ylabel('Density')
ax.set_title('LGD Distribution Fitting')
ax.legend()
plt.tight_layout()
plt.show()
```

## 4. Bayesian Parameter Estimation

### Default Rate with Prior

```python
# Bayesian estimation
bayes_estimator = BayesianEstimator()

# Use informative prior based on industry knowledge
# Prior: Beta(2, 80) implies ~2.5% mean with moderate uncertainty
bayes_results = bayes_estimator.fit_default_rate_bayesian(
    defaults,
    exposure_per_period,
    prior_alpha=2.0,
    prior_beta=80.0,
    n_samples=10000
)

print("Bayesian Estimation Results:")
print(f"  Prior mean: {2.0 / (2.0 + 80.0):.4f}")
print(f"  Posterior mean: {bayes_results['mean']:.4f}")
print(f"  Posterior median: {bayes_results['median']:.4f}")
print(f"  95% Credible Interval: [{bayes_results['ci_lower']:.4f}, {bayes_results['ci_upper']:.4f}]")
print(f"  95% HPD Interval: [{bayes_results['hpd_lower']:.4f}, {bayes_results['hpd_upper']:.4f}]")
```

### Prior Sensitivity Analysis

```python
def prior_sensitivity(defaults, exposure, priors):
    """
    Analyze how different priors affect posterior.
    """
    results = []
    for name, (alpha, beta) in priors.items():
        posterior = BayesianEstimator().fit_default_rate_bayesian(
            defaults, exposure, prior_alpha=alpha, prior_beta=beta
        )
        results.append({
            'Prior': name,
            'Prior_Mean': alpha / (alpha + beta),
            'Posterior_Mean': posterior['mean'],
            'CI_Width': posterior['ci_upper'] - posterior['ci_lower']
        })
    return pd.DataFrame(results)

# Different priors
priors = {
    'Uninformative': (1, 1),
    'Weak': (2, 80),
    'Strong': (5, 200),
    'Pessimistic': (5, 100)
}

sensitivity = prior_sensitivity(defaults, exposure_per_period, priors)
print("\nPrior Sensitivity Analysis:")
print(sensitivity.to_string(index=False))
```

### Posterior Visualization

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Plot prior
x = np.linspace(0, 0.10, 200)
prior_pdf = stats.beta.pdf(x, 2, 80)
ax.plot(x, prior_pdf, 'b--', linewidth=2, label='Prior')

# Plot posterior
ax.hist(bayes_results['samples'], bins=50, density=True,
        alpha=0.7, color='steelblue', label='Posterior')

# Mark estimates
ax.axvline(bayes_results['mean'], color='red', linestyle='-',
           linewidth=2, label=f"Mean: {bayes_results['mean']:.4f}")
ax.axvline(true_pd / 12, color='black', linestyle='-',
           linewidth=2, label=f"True: {true_pd/12:.4f}")

ax.set_xlabel('Monthly Default Rate')
ax.set_ylabel('Density')
ax.set_title('Bayesian Default Rate Estimation')
ax.legend()
plt.tight_layout()
plt.show()
```

## 5. Bayesian vs MLE Comparison

```python
def compare_mle_bayesian(defaults, exposure, true_rate):
    """
    Compare MLE and Bayesian estimates.
    """
    # MLE
    mle_rate = defaults.sum() / exposure.sum()

    # Bayesian with different sample sizes
    results = []
    for n in [10, 30, 60]:
        bayes = BayesianEstimator().fit_default_rate_bayesian(
            defaults[:n], exposure[:n],
            prior_alpha=2, prior_beta=80
        )
        results.append({
            'N_Months': n,
            'MLE': defaults[:n].sum() / exposure[:n].sum(),
            'Bayesian_Mean': bayes['mean'],
            'Bayesian_CI_Width': bayes['ci_upper'] - bayes['ci_lower'],
            'True_Rate': true_rate
        })

    return pd.DataFrame(results)

comparison = compare_mle_bayesian(defaults, exposure_per_period, true_pd/12)
print("\nMLE vs Bayesian by Sample Size:")
print(comparison.round(5).to_string(index=False))
```

## 6. Calibrating Model Parameters

### Macro VAE Calibration

```python
from privatecredit.models import MacroVAE, MacroVAEConfig
from privatecredit.calibration import ModelCalibrator

# Load historical macro data
historical_macro = pd.read_csv('historical_macro.csv')  # Replace with your data

# Initialize model calibrator
model_calibrator = ModelCalibrator()

# Calibrate VAE to historical distributions
calibration_results = model_calibrator.calibrate_macro_vae(
    historical_data=historical_macro,
    target_metrics={
        'gdp_mean': 0.025,
        'gdp_std': 0.015,
        'unemployment_mean': 0.045,
        'correlation_gdp_unemp': -0.65
    },
    n_iterations=100
)

print("Calibration Results:")
for metric, value in calibration_results['achieved_metrics'].items():
    target = calibration_results['target_metrics'].get(metric, 'N/A')
    print(f"  {metric}: Target={target}, Achieved={value:.4f}")
```

## 7. Validation and Backtesting

### Out-of-Sample Validation

```python
from privatecredit.calibration import BacktestFramework

# Split data
train_defaults = defaults[:48]
train_exposure = exposure_per_period[:48]
test_defaults = defaults[48:]
test_exposure = exposure_per_period[48:]

# Fit on training data
train_params = HistoricalCalibrator().fit_default_rates(
    train_defaults, train_exposure
)

# Validate on test data
backtester = BacktestFramework()
validation = backtester.validate_default_model(
    predicted_pd=train_params['mean_default_rate'],
    actual_defaults=test_defaults,
    actual_exposure=test_exposure
)

print("\nOut-of-Sample Validation:")
print(f"  Predicted PD: {train_params['mean_default_rate']:.4f}")
print(f"  Actual PD: {validation['actual_pd']:.4f}")
print(f"  Brier Score: {validation['brier_score']:.4f}")
print(f"  Log Loss: {validation['log_loss']:.4f}")
```

## Summary

| Method | Use Case | Data Requirements |
|--------|----------|-------------------|
| MLE | Point estimates | Moderate sample |
| Bootstrap CI | Uncertainty quantification | Moderate sample |
| Bayesian | Prior incorporation | Any sample size |
| Transition MLE | Panel/cohort data | State observations |
| LGD Fitting | Loss distribution | Resolved defaults |

**Key Takeaways:**
1. Use MLE for large samples, Bayesian for small samples or prior knowledge
2. Bootstrap provides non-parametric uncertainty estimates
3. Transition matrices should be validated against target rates
4. Compare multiple distribution models via AIC/BIC
5. Always validate out-of-sample

**Next:** [Model Selection Guide](07-model-selection.md)
