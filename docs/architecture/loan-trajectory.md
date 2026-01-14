---
layout: default
title: Loan Trajectory Model
---

# Loan Trajectory Model

Autoregressive transformer with diffusion head for individual loan path generation.

---

## Architecture

```
Loan Static Features
    |
    v
Feature Embedding
    |
    +--- Cohort Transitions (from Transition Transformer)
    |
    v
Causal Transformer Decoder
    |
    v
Diffusion Head (for continuous values)
    |
    v
State Head (for discrete states)
    |
    v
Generated Loan Trajectory
```

---

## Two-Stage Generation

### Stage 1: State Sequence

Generate discrete loan states autoregressively:

```
s_t ~ Categorical(f_state(h_t, P_cohort(t)))
```

Where the state probability combines:
- Loan-specific hidden state `h_t`
- Cohort-level transition matrix `P_cohort(t)`
- Idiosyncratic noise

### Stage 2: Continuous Values

For each state, generate continuous outcomes via diffusion:

```
x_t ~ Diffusion(mu_t, sigma_t | s_t, loan_features)
```

Continuous values include:
- Scheduled payment
- Actual payment
- Outstanding balance
- Loss given default (if default)

---

## Input Features

| Feature | Type | Description |
|---------|------|-------------|
| loan_id | ID | Unique identifier |
| original_balance | Float | Initial loan amount |
| interest_rate | Float | Contractual rate |
| rate_type | Categorical | Fixed/floating |
| asset_class | Categorical | Corporate/consumer/RE/AR |
| ltv_origination | Float | Loan-to-value at origination |
| debt_service_coverage | Float | DSCR for commercial |
| fico_origination | Float | Credit score (consumer) |
| vintage_month | Integer | Origination cohort |

---

## Output Trajectory

For each loan, generates monthly trajectory:

```python
trajectory = {
    'state': [0, 0, 0, 1, 1, 0, ...],      # State indices
    'scheduled_payment': [1000, 1000, ...], # Contractual
    'actual_payment': [1000, 800, ...],     # Realized
    'balance': [100000, 99500, ...],        # Outstanding
    'loss': [0, 0, 0, ...],                 # Loss if default
}
```

---

## Diffusion Head

The continuous values are generated using a score-based diffusion model:

```
Forward: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon
Reverse: x_{t-1} = (x_t - sqrt(1-alpha_t) * score(x_t, t)) / sqrt(alpha_t)
```

This captures the complex distributions of payment amounts and recovery values.

---

## Configuration

```python
from privatecredit.models import LoanTrajectoryModel, TrajectoryConfig

config = TrajectoryConfig(
    n_loan_features=24,
    n_states=7,
    d_model=256,
    n_heads=8,
    n_layers=6,
    diffusion_steps=100
)

model = LoanTrajectoryModel(config)

# Generate trajectories
trajectories = model.generate(
    loan_features=loans_tensor,
    cohort_transitions=P_matrices,
    macro_path=macro_tensor,
    n_samples=1000
)
```

---

## Idiosyncratic vs Systematic Risk

The model captures both:

| Risk Type | Mechanism |
|-----------|-----------|
| Systematic | Cohort transitions from macro path |
| Idiosyncratic | Loan-specific deviation via diffusion |

The balance is controlled by the `idio_scale` parameter:

```python
config = TrajectoryConfig(
    idio_scale=0.3  # 30% idiosyncratic variance
)
```

---

## State Transition Rules

Hard constraints ensure valid trajectories:

```
- Default, Prepaid, Matured are absorbing states
- Can only cure from 30/60/90 DPD
- Matured only at contractual term end
- Prepaid requires sufficient payment
```

---

[Back to Architecture](index.html)
