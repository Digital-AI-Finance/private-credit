---
layout: default
title: API Reference
---

# API Reference

Complete API documentation for the privatecredit package.

---

## Package Structure

```
privatecredit/
    data/           # Data generation and processing
    models/         # Deep learning models
    training/       # Training utilities
    evaluation/     # Metrics and visualization
    utils/          # Configuration and helpers
```

---

## Quick Import

```python
# Core imports
from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator
from privatecredit.models import (
    MacroVAE,
    TransitionTransformer,
    LoanTrajectoryModel,
    PortfolioAggregator
)

# Training
from privatecredit.training import Trainer

# Evaluation
from privatecredit.evaluation import (
    calculate_metrics,
    plot_loss_distribution,
    StressTestReport
)
```

---

## Module Reference

| Module | Description | Link |
|--------|-------------|------|
| **data** | Synthetic data generation, loan tape schemas | [Data API](data.html) |
| **models** | Deep generative model components | [Models API](models.html) |
| **training** | Training loops and optimizers | [Training API](training.html) |
| **evaluation** | Metrics, visualization, reporting | [Evaluation API](evaluation.html) |

---

## Data Module

### LoanTapeGenerator

Generate synthetic loan portfolios.

```python
from privatecredit.data import LoanTapeGenerator

generator = LoanTapeGenerator(
    n_loans=10000,
    n_months=60,
    asset_class_weights={'corporate': 0.5, 'consumer': 0.5}
)

loans_df, panel_df = generator.generate()
```

[Full Data API documentation](data.html)

---

## Models Module

### MacroVAE

Conditional VAE for macro scenario generation.

```python
from privatecredit.models import MacroVAE, MacroVAEConfig

config = MacroVAEConfig(n_macro_vars=9, latent_dim=32)
model = MacroVAE(config)

scenarios = model.generate(scenario=0, n_samples=100)
```

[Full Models API documentation](models.html)

---

## Training Module

### Trainer

Unified training interface.

```python
from privatecredit.training import Trainer

trainer = Trainer(model, lr=1e-4)
history = trainer.fit(data, epochs=100)
```

[Full Training API documentation](training.html)

---

## Evaluation Module

### Metrics

```python
from privatecredit.evaluation import calculate_metrics

metrics = calculate_metrics(
    predicted=predicted_losses,
    actual=actual_losses
)

print(metrics['rmse'], metrics['mae'])
```

[Full Evaluation API documentation](evaluation.html)

---

## Type Hints

All functions include type hints:

```python
def generate(
    self,
    scenario: int = 0,
    seq_length: int = 60,
    n_samples: int = 100
) -> torch.Tensor:
    ...
```

---

## Configuration Classes

Each model has a corresponding config dataclass:

```python
from dataclasses import dataclass

@dataclass
class MacroVAEConfig:
    n_macro_vars: int = 9
    seq_length: int = 60
    latent_dim: int = 32
    hidden_dim: int = 128
    n_layers: int = 2
    beta: float = 1.0
```

---

[Back to Home](../index.html)
