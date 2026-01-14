---
layout: default
title: Models API
---

# Models API

Deep generative model components.

---

## MacroVAE

Conditional Variational Autoencoder for macroeconomic scenario generation.

### MacroVAEConfig

```python
@dataclass
class MacroVAEConfig:
    n_macro_vars: int = 9       # Number of macro variables
    seq_length: int = 60        # Sequence length in months
    latent_dim: int = 32        # Latent space dimension
    hidden_dim: int = 128       # Hidden layer size
    n_layers: int = 2           # Number of LSTM layers
    beta: float = 1.0           # KL divergence weight
    n_scenarios: int = 4        # Number of scenario types
```

### MacroVAE

```python
class MacroVAE(nn.Module):
    def __init__(self, config: MacroVAEConfig)
```

#### Methods

**forward()**
```python
def forward(
    self,
    x: torch.Tensor,           # (batch, seq_len, n_vars)
    scenario: torch.Tensor      # (batch,) scenario indices
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (reconstruction, mu, log_var)
    """
```

**generate()**
```python
def generate(
    self,
    scenario: int = 0,
    seq_length: int = 60,
    n_samples: int = 100
) -> torch.Tensor:
    """
    Generate macro paths from latent space.
    Returns: Tensor of shape (n_samples, seq_length, n_vars)
    """
```

**generate_conditional()**
```python
def generate_conditional(
    self,
    conditions: Dict[str, Dict],  # {var_name: {'month': int, 'value': float}}
    seq_length: int = 60,
    n_samples: int = 100
) -> torch.Tensor:
    """
    Generate paths conditioned on specific values.
    """
```

---

## TransitionTransformer

Transformer model for cohort-level transition probability prediction.

### TransitionConfig

```python
@dataclass
class TransitionConfig:
    n_cohort_features: int = 12
    n_macro_vars: int = 9
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    n_states: int = 7
    dropout: float = 0.1
```

### TransitionTransformer

```python
class TransitionTransformer(nn.Module):
    def __init__(self, config: TransitionConfig)
```

#### Methods

**forward()**
```python
def forward(
    self,
    cohort_features: torch.Tensor,  # (batch, n_features)
    macro_path: torch.Tensor         # (batch, seq_len, n_macro)
) -> torch.Tensor:
    """
    Returns: Transition matrices (batch, seq_len, n_states, n_states)
    """
```

**predict()**
```python
def predict(
    self,
    cohort_features: pd.DataFrame,
    macro_paths: torch.Tensor
) -> torch.Tensor:
    """
    Convenience method accepting DataFrames.
    """
```

---

## LoanTrajectoryModel

Autoregressive transformer with diffusion head for loan-level generation.

### TrajectoryConfig

```python
@dataclass
class TrajectoryConfig:
    n_loan_features: int = 24
    n_states: int = 7
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    diffusion_steps: int = 100
    idio_scale: float = 0.3      # Idiosyncratic variance scale
    dropout: float = 0.1
```

### LoanTrajectoryModel

```python
class LoanTrajectoryModel(nn.Module):
    def __init__(self, config: TrajectoryConfig)
```

#### Methods

**forward()**
```python
def forward(
    self,
    loan_features: torch.Tensor,      # (batch, n_features)
    cohort_transitions: torch.Tensor, # (batch, seq_len, n_states, n_states)
    target_states: torch.Tensor = None # (batch, seq_len) for training
) -> Dict[str, torch.Tensor]:
    """
    Returns: Dict with 'state_logits', 'value_params'
    """
```

**generate()**
```python
def generate(
    self,
    loan_features: torch.Tensor,
    cohort_transitions: torch.Tensor,
    macro_path: torch.Tensor,
    n_samples: int = 1
) -> Dict[str, torch.Tensor]:
    """
    Generate loan trajectories autoregressively.
    Returns: Dict with 'states', 'payments', 'balances', 'losses'
    """
```

---

## PortfolioAggregator

Differentiable waterfall simulation.

### WaterfallConfig

```python
@dataclass
class WaterfallConfig:
    tranches: List[Dict]         # Tranche specifications
    oc_trigger: float = 1.20     # OC test threshold
    ic_trigger: float = 1.05     # IC test threshold
    reinvestment_period: int = 24
    management_fee: float = 0.005
    soft_temperature: float = 0.1  # For differentiable gates
```

### PortfolioAggregator

```python
class PortfolioAggregator(nn.Module):
    def __init__(self, config: WaterfallConfig)
```

#### Methods

**aggregate_cashflows()**
```python
def aggregate_cashflows(
    self,
    trajectories: Dict[str, torch.Tensor],
    loan_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate loan-level to portfolio-level cashflows.
    """
```

**apply_waterfall()**
```python
def apply_waterfall(
    self,
    portfolio_cf: torch.Tensor  # (n_sims, n_months, n_cf_types)
) -> Dict[str, torch.Tensor]:
    """
    Apply waterfall rules to distribute cashflows.
    Returns: Dict with tranche cashflows and coverage tests
    """
```

**monte_carlo_simulate()**
```python
def monte_carlo_simulate(
    self,
    loans_df: pd.DataFrame,
    macro_vae: MacroVAE,
    transition_model: TransitionTransformer,
    trajectory_model: LoanTrajectoryModel,
    n_simulations: int = 10000,
    scenario_mix: Dict[str, float] = None
) -> SimulationResults:
    """
    Run full Monte Carlo simulation.
    """
```

---

## MarkovTransitionModel

Baseline homogeneous Markov chain model.

### MarkovTransitionModel

```python
class MarkovTransitionModel:
    def __init__(self, n_states: int = 7)
```

#### Methods

**fit()**
```python
def fit(self, panel_df: pd.DataFrame) -> None:
    """
    Estimate transition matrix from observed data.
    """
```

**simulate()**
```python
def simulate(
    self,
    initial_states: np.ndarray,
    n_steps: int
) -> np.ndarray:
    """
    Simulate state sequences.
    Returns: Array of shape (n_loans, n_steps)
    """
```

---

## SimulationResults

Container for Monte Carlo simulation outputs.

```python
@dataclass
class SimulationResults:
    portfolio_losses: np.ndarray     # (n_sims,)
    tranche_results: List[TrancheResult]
    cashflows: Dict[str, np.ndarray]

    # Computed properties
    @property
    def expected_loss(self) -> float: ...
    @property
    def var_99(self) -> float: ...
    @property
    def cvar_99(self) -> float: ...

    def to_excel(self, path: str) -> None: ...
    def summary(self) -> pd.DataFrame: ...
```

---

[Back to API Reference](index.html)
