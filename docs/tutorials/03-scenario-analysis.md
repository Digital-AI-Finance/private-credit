---
layout: default
title: "Tutorial 3: Scenario Analysis"
---

# Tutorial 3: Scenario Analysis

Perform stress testing and scenario analysis.

---

## Step 1: Load Trained Models

```python
import torch
from privatecredit.models import MacroVAE, TransitionTransformer, LoanTrajectoryModel

# Load saved models
checkpoint = torch.load('models/trained_models.pt')

macro_vae = MacroVAE(checkpoint['configs']['macro_vae'])
macro_vae.load_state_dict(checkpoint['macro_vae'])

transition_model = TransitionTransformer(checkpoint['configs']['transition'])
transition_model.load_state_dict(checkpoint['transition_model'])

trajectory_model = LoanTrajectoryModel(checkpoint['configs']['trajectory'])
trajectory_model.load_state_dict(checkpoint['trajectory_model'])
```

---

## Step 2: Define Scenarios

```python
# Standard regulatory scenarios
SCENARIOS = {
    0: 'Baseline',
    1: 'Adverse',
    2: 'Severely Adverse',
    3: 'Stagflation',
}

# Custom scenario: rapid rate hike
custom_scenario = {
    'gdp_growth_yoy': -0.02,        # Mild recession
    'unemployment_rate': 0.06,       # Moderate unemployment
    'inflation_rate': 0.08,          # High inflation
    'policy_rate': 0.07,             # Aggressive rate hikes
    'credit_spread_hy': 600,         # Wide spreads
}
```

---

## Step 3: Generate Macro Paths

```python
# Generate paths for each scenario
macro_paths = {}

for scenario_id, name in SCENARIOS.items():
    paths = macro_vae.generate(
        scenario=scenario_id,
        seq_length=60,
        n_samples=1000
    )
    macro_paths[name] = paths
    print(f"{name}: Generated {len(paths)} paths")

# Visualize scenarios
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
variables = ['gdp_growth_yoy', 'unemployment_rate',
             'credit_spread_hy', 'policy_rate']

for ax, var in zip(axes.flat, variables):
    for name, paths in macro_paths.items():
        mean_path = paths[var].mean(axis=0)
        ax.plot(mean_path, label=name)
    ax.set_title(var)
    ax.legend()

plt.tight_layout()
plt.savefig('scenario_comparison.pdf')
```

---

## Step 4: Conditional Generation

```python
# Condition on specific macro outcomes
conditioned_paths = macro_vae.generate_conditional(
    conditions={
        'gdp_growth_yoy': {'month': 12, 'value': -0.04},  # -4% GDP at month 12
        'unemployment_rate': {'month': 24, 'value': 0.10}, # 10% unemployment at month 24
    },
    seq_length=60,
    n_samples=500
)

print(f"Generated {len(conditioned_paths)} conditioned paths")
```

---

## Step 5: Run Stress Test

```python
from privatecredit.data import LoanTapeGenerator

# Generate test portfolio
generator = LoanTapeGenerator(n_loans=5000, random_seed=123)
loans_df, _ = generator.generate_static_features()

# Run stress test for each scenario
stress_results = {}

for name, paths in macro_paths.items():
    # Get cohort transitions
    cohort_transitions = transition_model.predict(
        cohort_features=loans_df,
        macro_paths=paths
    )

    # Generate loan trajectories
    trajectories = trajectory_model.generate(
        loan_features=loans_df,
        cohort_transitions=cohort_transitions,
        macro_paths=paths,
        n_samples=1000
    )

    # Calculate losses
    losses = trajectories['loss'].sum(dim=-1) / loans_df['original_balance'].sum()

    stress_results[name] = {
        'mean_loss': losses.mean().item(),
        'var_99': losses.quantile(0.99).item(),
        'max_loss': losses.max().item(),
    }

    print(f"{name}: Mean Loss = {stress_results[name]['mean_loss']:.2%}")
```

---

## Step 6: Sensitivity Analysis

```python
# Analyze sensitivity to individual macro shocks
sensitivities = {}

base_scenario = macro_paths['Baseline'].mean(axis=0)
shock_size = 0.01  # 1% shock

for var in ['gdp_growth_yoy', 'unemployment_rate', 'credit_spread_hy']:
    # Apply shock
    shocked_scenario = base_scenario.copy()
    shocked_scenario[var] += shock_size

    # Recompute losses
    shocked_transitions = transition_model.predict(
        cohort_features=loans_df,
        macro_paths=shocked_scenario.unsqueeze(0)
    )

    shocked_trajectories = trajectory_model.generate(
        loan_features=loans_df,
        cohort_transitions=shocked_transitions,
        macro_paths=shocked_scenario.unsqueeze(0),
        n_samples=500
    )

    shocked_loss = shocked_trajectories['loss'].sum() / loans_df['original_balance'].sum()
    base_loss = stress_results['Baseline']['mean_loss']

    sensitivities[var] = (shocked_loss - base_loss) / shock_size
    print(f"dLoss/d{var} = {sensitivities[var]:.4f}")
```

---

## Step 7: Reverse Stress Testing

```python
from privatecredit.evaluation import reverse_stress_test

# Find scenarios that produce specific loss levels
target_losses = [0.05, 0.10, 0.15]  # 5%, 10%, 15%

for target in target_losses:
    scenario = reverse_stress_test(
        macro_vae=macro_vae,
        transition_model=transition_model,
        trajectory_model=trajectory_model,
        loans_df=loans_df,
        target_loss=target,
        n_iterations=100
    )

    print(f"\nScenario for {target:.0%} loss:")
    print(f"  GDP: {scenario['gdp_growth_yoy']:.1%}")
    print(f"  Unemployment: {scenario['unemployment_rate']:.1%}")
    print(f"  HY Spread: {scenario['credit_spread_hy']:.0f} bps")
```

---

## Step 8: Generate Report

```python
from privatecredit.evaluation import StressTestReport

report = StressTestReport(
    portfolio=loans_df,
    scenarios=SCENARIOS,
    results=stress_results,
    sensitivities=sensitivities
)

report.generate_pdf('stress_test_report.pdf')
print("Report generated: stress_test_report.pdf")
```

---

## Summary Table

| Scenario | Expected Loss | VaR 99% | Max Loss |
|----------|---------------|---------|----------|
| Baseline | 2.0% | 3.5% | 5.0% |
| Adverse | 4.5% | 7.0% | 10.0% |
| Severely Adverse | 8.0% | 12.0% | 18.0% |
| Stagflation | 6.0% | 9.5% | 14.0% |

---

## Next Steps

- [Tutorial 4: Portfolio Simulation](04-portfolio-simulation.html)
- [Back to Tutorials](index.html)
