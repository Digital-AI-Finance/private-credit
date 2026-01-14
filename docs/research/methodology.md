---
layout: default
title: Methodology
---

# Methodology

Detailed description of the modeling approach.

---

## Problem Formulation

Given a portfolio of loans with features `X`, we aim to generate:

1. **Macro paths** `M_{1:T}`: Correlated macroeconomic time series
2. **State sequences** `S_{1:T}`: Loan state trajectories
3. **Continuous values** `V_{1:T}`: Payments, balances, losses
4. **Portfolio outcomes** `L`: Loss distribution, tranche returns

---

## Hierarchical Generative Model

### Level 1: Macro Scenario Generation

We use a Conditional Variational Autoencoder (CVAE) to generate macro paths.

**Encoder:**
```
q_phi(z | M, s) = N(mu_phi(M, s), sigma_phi(M, s))
```

**Decoder:**
```
p_theta(M | z, s) = Prod_t N(m_t | g_theta(z, s, m_{<t}))
```

**Loss:**
```
L_macro = E_q[||M - M_hat||^2] + beta * KL(q_phi || p(z))
```

The scenario label `s` enables conditional generation under different economic regimes.

### Level 2: Cohort Transitions

A Transformer encoder predicts time-varying transition matrices:

```
P(t) = f_psi(cohort_features, M_{1:t})
```

Cross-attention allows the macro path to modulate transition dynamics:

```
Attention(Q_cohort, K_macro, V_macro)
```

**Loss:**
```
L_trans = -sum_{t,i,j} n_ij(t) * log(P_ij(t))
```

### Level 3: Loan Trajectories

An autoregressive transformer generates individual loan paths:

**State Generation:**
```
s_t ~ Cat(softmax(h_t * W_s + P_cohort(s_{t-1}, :)))
```

**Continuous Values (Diffusion):**
```
v_t = Denoise(epsilon_t, s_t, x_loan)
```

The diffusion head captures complex distributions of payments and recoveries.

### Level 4: Portfolio Aggregation

Cashflows are aggregated and distributed via waterfall rules:

```
Collections = sum_i (payment_i)
Losses = sum_i (loss_i)
Tranche_CF = Waterfall(Collections, Losses, Rules)
```

Differentiable via soft approximations:
```
gate(x, threshold) = sigmoid((x - threshold) / temperature)
```

---

## Correlation Structure

Correlation is induced at multiple levels:

| Source | Mechanism | Magnitude |
|--------|-----------|-----------|
| Macro | Shared macro path | ~60% of total |
| Cohort | Vintage/asset class grouping | ~25% of total |
| Factor | Latent industry/geography | ~10% of total |
| Idiosyncratic | Diffusion noise | ~5% of total |

The hierarchical structure naturally captures:
- **Systematic risk**: All loans exposed to same macro
- **Concentrated risk**: Cohort-level clustering
- **Diversification**: Residual idiosyncratic variation

---

## Training Strategy

### Stage 1: Component Pre-training

Train each component separately:

| Component | Objective | Data |
|-----------|-----------|------|
| Macro VAE | Reconstruction + KL | Historical macro series |
| Transition Transformer | Cross-entropy on transitions | Cohort transition counts |
| Loan Trajectory | State + diffusion loss | Loan-month panel |

### Stage 2: End-to-End Fine-tuning

Joint training with portfolio objectives:

```
L_total = L_macro + L_trans + L_traj + lambda * L_portfolio
```

Where `L_portfolio` matches:
- Historical loss rates
- Tranche return distributions
- Tail risk measures

### Stage 3: Calibration

Final calibration to match:
- Observed default rates by cohort
- Historical macro correlations
- Recovery rate distributions

---

## Scenario Conditioning

### Standard Scenarios

| Scenario | GDP Shift | Unemp. Shift | Spread Mult. |
|----------|-----------|--------------|--------------|
| Baseline | 0% | 0% | 1.0x |
| Adverse | -3% | +3% | 2.0x |
| Severely Adverse | -6% | +8% | 4.0x |
| Stagflation | -2% | +3% | 2.5x |

### Custom Conditioning

Condition on specific outcomes:

```python
model.generate_conditional({
    'gdp_growth_yoy': {'month': 12, 'value': -0.04},
    'unemployment_rate': {'month': 24, 'value': 0.10}
})
```

---

## Evaluation Metrics

### Generation Quality

| Metric | Target |
|--------|--------|
| Macro reconstruction RMSE | < 0.5% |
| Transition accuracy | > 85% |
| State sequence accuracy | > 80% |
| Payment RMSE | < $100 |

### Portfolio Metrics

| Metric | Validation |
|--------|------------|
| Expected Loss | Within 10% of historical |
| VaR 99% | Conservative vs historical |
| Scenario ordering | Severe > Adverse > Baseline |
| Tranche attachment | Consistent with ratings |

---

## Computational Considerations

### Memory

| Component | Memory (10k loans, 60 months) |
|-----------|------------------------------|
| Macro VAE | ~100 MB |
| Transition Transformer | ~500 MB |
| Loan Trajectory | ~2 GB |
| Full simulation (10k sims) | ~8 GB |

### Runtime

| Operation | Time (GPU) |
|-----------|------------|
| Train Macro VAE (200 epochs) | ~10 min |
| Train Transitions (100 epochs) | ~30 min |
| Train Trajectories (50 epochs) | ~2 hours |
| Monte Carlo (10k sims) | ~5 min |

---

## Limitations

1. **Data requirements**: Needs historical loan-level data
2. **Stationarity**: Assumes stable regime (may need retraining)
3. **Tail estimation**: Limited by simulation sample size
4. **Model risk**: Deep learning opacity

---

## Future Directions

- **Continuous-time models**: Replace discrete monthly steps
- **Graph neural networks**: Explicit loan relationship modeling
- **Online learning**: Adapt to new data without full retraining
- **Explainability**: Attribution of losses to factors

---

[Back to Research](index.html)
