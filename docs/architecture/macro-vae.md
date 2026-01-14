---
layout: default
title: Macro VAE
---

# Macro VAE

Conditional Variational Autoencoder for macroeconomic scenario generation.

---

## Architecture

```
Encoder (Bidirectional LSTM)
    |
    v
Latent Space z ~ N(mu, sigma)
    |
    v
Decoder (Autoregressive LSTM)
    |
    v
Generated Macro Path
```

---

## Mathematical Formulation

**Encoder**:
```
z ~ q_phi(z | m_{1:T}, s) = N(mu_phi, sigma_phi)
```

**Decoder**:
```
m_hat_{1:T} ~ p_theta(m | z, s)
```

**Loss**:
```
L = E_q[||m - m_hat||^2] + beta * KL(q || p)
```

---

## Macro Variables

| Variable | Description | Range |
|----------|-------------|-------|
| GDP Growth | Year-over-year GDP growth | [-15%, 15%] |
| Unemployment | Unemployment rate | [2%, 25%] |
| Inflation | CPI inflation | [-5%, 15%] |
| Policy Rate | Central bank rate | [0%, 15%] |
| 10Y Yield | Government bond yield | [0%, 10%] |
| IG Spread | Investment grade spread | [50, 500] bps |
| HY Spread | High yield spread | [200, 2000] bps |
| Property Index | Property price index | [50, 200] |

---

## Scenarios

| Scenario | GDP Shift | Unemployment | Spread Multiplier |
|----------|-----------|--------------|-------------------|
| Baseline | 0% | 5% | 1.0x |
| Adverse | -3% | +3% | 2.0x |
| Severely Adverse | -6% | +8% | 4.0x |
| Stagflation | -2% | +3% | 2.5x |

---

## Usage

```python
from privatecredit.models import MacroVAE, MacroVAEConfig

config = MacroVAEConfig(
    n_macro_vars=9,
    seq_length=60,
    latent_dim=32
)

model = MacroVAE(config)

# Generate scenarios
scenarios = model.generate(
    scenario=0,  # baseline
    seq_length=60,
    n_samples=100
)
```

---

[Back to Architecture](index.html)
