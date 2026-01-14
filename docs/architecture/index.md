---
layout: default
title: Architecture
---

# Architecture Overview

Private Credit uses a **hierarchical deep generative framework** with four levels:

```
Level 1: MACRO SCENARIO GENERATOR
    |
    v
Level 2: TRANSITION TRANSFORMER
    |
    v
Level 3: LOAN TRAJECTORY MODEL
    |
    v
Level 4: PORTFOLIO AGGREGATOR
```

---

## Level 1: Macro VAE

**Purpose**: Generate correlated macroeconomic time series

- **Architecture**: Conditional Variational Autoencoder with LSTM encoder/decoder
- **Input**: Historical macro series, scenario specification
- **Output**: Correlated macro path (GDP, unemployment, spreads)
- **Key Feature**: Scenario conditioning (baseline/adverse/severe)

[Learn more about Macro VAE](macro-vae.html)

---

## Level 2: Transition Transformer

**Purpose**: Predict cohort-level transition matrices

- **Architecture**: Transformer Encoder with cross-attention
- **Input**: Macro path, cohort features (vintage, asset class)
- **Output**: Time-varying transition probabilities
- **Key Feature**: Captures systematic risk and macro sensitivity

[Learn more about Transition Transformer](transition-transformer.html)

---

## Level 3: Loan Trajectory Model

**Purpose**: Generate individual loan paths

- **Architecture**: Autoregressive Transformer Decoder + Diffusion Head
- **Input**: Loan features, cohort transitions, macro path
- **Output**: State sequence, payment sequence, default timing
- **Key Feature**: Captures idiosyncratic risk within cohorts

[Learn more about Loan Trajectory Model](loan-trajectory.html)

---

## Level 4: Portfolio Aggregator

**Purpose**: Aggregate to portfolio and tranche level

- **Architecture**: Differentiable waterfall simulation
- **Input**: Loan trajectories
- **Output**: Portfolio cashflows, loss distribution, tranche returns
- **Key Feature**: End-to-end differentiable for joint training

[Learn more about Portfolio Aggregator](portfolio-aggregator.html)

---

## Correlation Structure

Correlation is captured at multiple levels:

| Level | Mechanism |
|-------|-----------|
| Macro | All loans affected by same macro path |
| Cohort | Loans in same cohort share transition dynamics |
| Factor | Latent factors for industry/geography clustering |
| Idiosyncratic | Residual loan-specific variation |

---

## Training Strategy

1. **Stage 1**: Pre-train components separately
2. **Stage 2**: End-to-end fine-tuning with portfolio targets
3. **Stage 3**: Calibration to historical data

---

[Back to Home](../index.html)
