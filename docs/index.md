---
layout: default
title: Home
---

# Private Credit

**Deep Generative Models for Private Credit SPV Analytics**

A hierarchical framework for modeling Special Purpose Vehicles (SPVs) that securitize loan portfolios across multiple asset classes.

---

## Key Features

| Component | Architecture | Purpose |
|-----------|--------------|---------|
| **Macro VAE** | Conditional LSTM-VAE | Generate correlated macro scenarios |
| **Transition Transformer** | Transformer Encoder | Cohort-level state dynamics |
| **Loan Trajectory Model** | AR Decoder + Diffusion | Individual loan path generation |
| **Portfolio Aggregator** | Differentiable Waterfall | Loss distribution & tranche returns |

---

## Quick Start

```bash
pip install privatecredit
```

```python
from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator
from privatecredit.models import MacroVAE, PortfolioAggregator

# Generate synthetic portfolio
generator = LoanTapeGenerator(n_loans=10000)
loans, panel = generator.generate()

# Generate macro scenarios
macro_gen = MacroScenarioGenerator()
baseline = macro_gen.generate_scenario('baseline')
adverse = macro_gen.generate_scenario('adverse')
```

---

## Architecture Overview

```
Level 1: MACRO SCENARIO GENERATOR (VAE)
    |-- GDP, Unemployment, Credit Spreads
    v
Level 2: TRANSITION TRANSFORMER
    |-- Cohort-level transition matrices
    v
Level 3: LOAN TRAJECTORY MODEL
    |-- Individual loan state & payment sequences
    v
Level 4: PORTFOLIO AGGREGATOR
    |-- Waterfall, VaR, CVaR, Tranche Returns
```

---

## Asset Classes Supported

- **Corporate Loans**: SME, mid-market, large corporate
- **Consumer Credit**: Personal loans, auto loans
- **Real Estate**: Commercial and residential mortgages
- **Trade Receivables**: Invoice financing, factoring

---

## Use Cases

| Application | Output |
|-------------|--------|
| **Pricing** | Fair value of SPV tranches |
| **Risk Management** | VaR, CVaR, capital allocation |
| **Regulatory** | IFRS 9 ECL, Basel IRB, Solvency II |
| **Investment** | Risk-adjusted returns by tranche |

---

## Documentation

- [Getting Started](getting-started.html) - 5-minute quickstart
- [Installation](installation.html) - Detailed installation guide
- [Architecture](architecture/) - Model documentation
- [Tutorials](tutorials/) - Step-by-step guides
- [API Reference](api/) - Complete API documentation
- [Research](research/) - Methodology and references

---

## Citation

```bibtex
@software{privatecredit2026,
  title = {Private Credit: Deep Generative Models for SPV Analytics},
  author = {Digital Finance Research},
  year = {2026},
  url = {https://github.com/Digital-AI-Finance/private-credit}
}
```

---

## License

MIT License - see [LICENSE](https://github.com/Digital-AI-Finance/private-credit/blob/main/LICENSE)

---

[View on GitHub](https://github.com/Digital-AI-Finance/private-credit){: .btn }
