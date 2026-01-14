---
layout: default
title: Research
---

# Research

Academic foundations and methodology.

---

## Overview

Private Credit implements a **hierarchical deep generative framework** for modeling structured credit portfolios. The approach addresses key challenges in credit risk modeling:

1. **Complex correlation structures** across macro, cohort, and loan levels
2. **Non-stationary dynamics** driven by macroeconomic conditions
3. **Path-dependent outcomes** (state transitions, prepayment, default)
4. **Fat-tailed loss distributions** requiring accurate tail estimation

---

## Key Contributions

### 1. Hierarchical Generation

Unlike flat approaches that model loans independently, our framework captures:

| Level | Component | Captures |
|-------|-----------|----------|
| Macro | Conditional VAE | Systematic risk, scenario conditioning |
| Cohort | Transformer | Vintage effects, asset class dynamics |
| Loan | AR + Diffusion | Idiosyncratic risk, path dependence |
| Portfolio | Differentiable Waterfall | Aggregation, tranche structuring |

### 2. Conditional Scenario Generation

The Macro VAE enables:
- **Scenario interpolation**: Generate paths between baseline and stress
- **Conditional generation**: Fix endpoints, generate consistent paths
- **Reverse stress testing**: Find scenarios producing target losses

### 3. End-to-End Differentiability

The entire pipeline is differentiable, enabling:
- Joint training with portfolio-level objectives
- Gradient-based calibration to historical data
- Sensitivity analysis via automatic differentiation

---

## Methodology

[Full methodology documentation](methodology.html)

---

## Publications

*Coming soon*

---

## References

[Complete bibliography](references.html)

---

## Research Team

- Digital Finance Research Group
- FHGR - University of Applied Sciences of the Grisons

---

[Back to Home](../index.html)
