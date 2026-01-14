# Private Credit

[![Tests](https://github.com/Digital-AI-Finance/private-credit/actions/workflows/tests.yml/badge.svg)](https://github.com/Digital-AI-Finance/private-credit/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://digital-ai-finance.github.io/private-credit/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Deep Generative Models for Private Credit SPV Analytics**

A hierarchical framework for loan-level trajectory simulation, portfolio loss estimation, and tranche-level cashflow projections.

## Features

- **Macro VAE**: Conditional variational autoencoder for macroeconomic scenario generation
- **Transition Transformer**: Cohort-level transition probability prediction
- **Loan Trajectory Model**: Autoregressive transformer with diffusion head for individual loan paths
- **Portfolio Aggregator**: Differentiable waterfall simulation for tranche-level analytics

## Installation

```bash
pip install privatecredit
```

Or install from source:

```bash
git clone https://github.com/Digital-AI-Finance/private-credit.git
cd private-credit
pip install -e .
```

## Quick Start

```python
from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator
from privatecredit.models import MacroVAE, PortfolioAggregator

# Generate synthetic loan portfolio
generator = LoanTapeGenerator(n_loans=10000, n_months=60)
loans_df, panel_df = generator.generate()

# Generate macro scenarios
macro_gen = MacroScenarioGenerator(n_months=60)
baseline = macro_gen.generate_scenario('baseline')
adverse = macro_gen.generate_scenario('adverse')

# Run portfolio simulation
aggregator = PortfolioAggregator(waterfall_config)
results = aggregator.monte_carlo_simulate(
    loans_df=loans_df,
    n_simulations=10000
)

print(f"Expected Loss: {results.expected_loss:.2%}")
print(f"VaR 99%: {results.var_99:.2%}")
```

## Architecture

```
Level 1: MACRO SCENARIO GENERATOR (Conditional VAE)
    |
    v
Level 2: TRANSITION TRANSFORMER (Cross-attention on macro)
    |
    v
Level 3: LOAN TRAJECTORY MODEL (AR Transformer + Diffusion)
    |
    v
Level 4: PORTFOLIO AGGREGATOR (Differentiable Waterfall)
```

## Documentation

Full documentation: [https://digital-ai-finance.github.io/private-credit/](https://digital-ai-finance.github.io/private-credit/)

- [Getting Started](https://digital-ai-finance.github.io/private-credit/getting-started.html)
- [Architecture Overview](https://digital-ai-finance.github.io/private-credit/architecture/)
- [Tutorials](https://digital-ai-finance.github.io/private-credit/tutorials/)
- [API Reference](https://digital-ai-finance.github.io/private-credit/api/)

## Asset Classes

The framework supports four asset classes:

| Asset Class | Examples |
|-------------|----------|
| Corporate | Term loans, revolvers, leveraged loans |
| Consumer | Auto loans, personal loans, credit cards |
| Real Estate | Commercial mortgages, residential loans |
| Receivables | Trade receivables, invoice financing |

## Citation

```bibtex
@software{privatecredit2026,
  title = {Private Credit: Deep Generative Models for SPV Analytics},
  author = {Digital Finance Research},
  year = {2026},
  url = {https://github.com/Digital-AI-Finance/private-credit}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
