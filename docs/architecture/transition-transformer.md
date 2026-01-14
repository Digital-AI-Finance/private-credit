---
layout: default
title: Transition Transformer
---

# Transition Transformer

Transformer-based model for cohort-level transition probability prediction.

---

## Architecture

```
Cohort Features (vintage, asset_class, ...)
    |
    v
Feature Embedding Layer
    |
    v
Macro Path (from Macro VAE)
    |
    v
Cross-Attention Transformer Encoder
    |
    v
Time-Varying Transition Matrices P(t)
```

---

## Input Features

| Feature Type | Examples |
|--------------|----------|
| Cohort Static | Vintage month, asset class, origination LTV bucket |
| Macro Series | GDP growth, unemployment, spreads |
| Temporal | Month index, seasonality encoding |

---

## Output: Transition Matrices

For each cohort and time step, outputs a 7x7 transition matrix:

```
States: [Performing, 30DPD, 60DPD, 90DPD, Default, Prepaid, Matured]

P(t) = [
    [p_00  p_01  p_02  ...  p_06]
    [p_10  p_11  p_12  ...  p_16]
    ...
    [p_60  p_61  p_62  ...  p_66]
]
```

---

## Cross-Attention Mechanism

The transformer uses cross-attention to condition on macro paths:

```
Q = Cohort embeddings
K = Macro sequence
V = Macro sequence

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

---

## Configuration

```python
from privatecredit.models import TransitionTransformer, TransitionConfig

config = TransitionConfig(
    n_cohort_features=12,
    n_macro_vars=9,
    d_model=128,
    n_heads=4,
    n_layers=3,
    n_states=7
)

model = TransitionTransformer(config)

# Predict transitions
P_matrices = model.forward(
    cohort_features=cohort_tensor,  # (batch, n_features)
    macro_path=macro_tensor,        # (batch, seq_len, n_macro)
    seq_length=60
)
# Output shape: (batch, seq_len, n_states, n_states)
```

---

## Training

```python
from privatecredit.training import TransitionTrainer

trainer = TransitionTrainer(
    model=model,
    lr=1e-4,
    weight_decay=1e-5
)

# Train on observed transitions
history = trainer.fit(
    cohort_data=cohort_df,
    transition_counts=counts_tensor,
    macro_data=macro_df,
    epochs=100
)
```

---

## Loss Function

The model is trained with cross-entropy loss on observed transitions:

```
L = -sum_t sum_i sum_j n_ij(t) * log(p_ij(t))
```

Where `n_ij(t)` is the count of transitions from state i to state j at time t.

---

## Macro Sensitivity

The transition matrices vary smoothly with macro conditions:

| Macro Shock | Effect on Transitions |
|-------------|----------------------|
| GDP -3% | P(default) increases ~40% |
| Unemployment +3% | P(30DPD) increases ~25% |
| Spreads 2x | P(cure) decreases ~20% |

---

[Back to Architecture](index.html)
