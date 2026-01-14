---
layout: default
title: "Tutorial 2: Training Models"
---

# Tutorial 2: Training Models

Train the deep generative model components.

---

## Step 1: Prepare Training Data

```python
from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator

# Generate training data
loan_gen = LoanTapeGenerator(n_loans=50000, n_months=60, random_seed=42)
macro_gen = MacroScenarioGenerator(n_months=60)

# Multiple scenarios for macro VAE
scenarios = {
    'baseline': macro_gen.generate_scenario('baseline'),
    'adverse': macro_gen.generate_scenario('adverse'),
    'severe': macro_gen.generate_scenario('severely_adverse'),
}

# Generate loan panel with baseline macro
loans_df, panel_df = loan_gen.generate(macro_df=scenarios['baseline'])
```

---

## Step 2: Train Macro VAE

```python
from privatecredit.models import MacroVAE, MacroVAEConfig
from privatecredit.training import MacroVAETrainer

# Configure model
config = MacroVAEConfig(
    n_macro_vars=9,
    seq_length=60,
    latent_dim=32,
    hidden_dim=128,
    n_layers=2
)

model = MacroVAE(config)

# Prepare macro data tensor
import torch
macro_tensor = torch.stack([
    torch.tensor(df.values, dtype=torch.float32)
    for df in scenarios.values()
])

# Train
trainer = MacroVAETrainer(model, lr=1e-3)
history = trainer.fit(
    macro_data=macro_tensor,
    scenario_labels=torch.tensor([0, 1, 2]),
    epochs=200,
    batch_size=32
)

print(f"Final loss: {history['loss'][-1]:.4f}")
```

---

## Step 3: Train Transition Transformer

```python
from privatecredit.models import TransitionTransformer, TransitionConfig
from privatecredit.training import TransitionTrainer

# Configure model
config = TransitionConfig(
    n_cohort_features=12,
    n_macro_vars=9,
    d_model=128,
    n_heads=4,
    n_layers=3,
    n_states=7
)

model = TransitionTransformer(config)

# Prepare cohort data
cohort_df = panel_df.groupby(['vintage_month', 'asset_class']).agg({
    'original_balance': 'sum',
    'loan_id': 'count'
}).reset_index()

# Extract observed transitions
from privatecredit.data import extract_transitions
transitions = extract_transitions(panel_df)

# Train
trainer = TransitionTrainer(model, lr=1e-4)
history = trainer.fit(
    cohort_data=cohort_df,
    transitions=transitions,
    macro_data=scenarios['baseline'],
    epochs=100
)
```

---

## Step 4: Train Loan Trajectory Model

```python
from privatecredit.models import LoanTrajectoryModel, TrajectoryConfig
from privatecredit.training import TrajectoryTrainer

# Configure model
config = TrajectoryConfig(
    n_loan_features=24,
    n_states=7,
    d_model=256,
    n_heads=8,
    n_layers=6,
    diffusion_steps=100
)

model = LoanTrajectoryModel(config)

# Prepare loan trajectories from panel
from privatecredit.data import prepare_trajectories
trajectories = prepare_trajectories(panel_df, loans_df)

# Train
trainer = TrajectoryTrainer(model, lr=1e-4)
history = trainer.fit(
    loan_features=loans_df,
    trajectories=trajectories,
    epochs=50,
    batch_size=256
)
```

---

## Step 5: End-to-End Fine-Tuning

```python
from privatecredit.training import EndToEndTrainer
from privatecredit.models import PortfolioAggregator, WaterfallConfig

# Configure waterfall
waterfall_config = WaterfallConfig(
    tranches=[
        {'name': 'Senior', 'size': 0.70, 'coupon': 0.05},
        {'name': 'Mezz', 'size': 0.15, 'coupon': 0.08},
        {'name': 'Junior', 'size': 0.10, 'coupon': 0.12},
        {'name': 'Equity', 'size': 0.05, 'coupon': None},
    ]
)
aggregator = PortfolioAggregator(waterfall_config)

# Joint fine-tuning
e2e_trainer = EndToEndTrainer(
    macro_vae=macro_vae,
    transition_model=transition_model,
    trajectory_model=trajectory_model,
    aggregator=aggregator,
    lr=1e-5
)

# Fine-tune with portfolio-level targets
history = e2e_trainer.fit(
    loans_df=loans_df,
    panel_df=panel_df,
    target_loss_rate=0.02,  # Historical average
    epochs=20
)
```

---

## Step 6: Save Models

```python
import torch

# Save all components
torch.save({
    'macro_vae': macro_vae.state_dict(),
    'transition_model': transition_model.state_dict(),
    'trajectory_model': trajectory_model.state_dict(),
    'configs': {
        'macro_vae': macro_config,
        'transition': transition_config,
        'trajectory': trajectory_config,
    }
}, 'models/trained_models.pt')

print("Models saved successfully!")
```

---

## Training Tips

| Component | Epochs | Batch Size | Learning Rate |
|-----------|--------|------------|---------------|
| Macro VAE | 200-500 | 32 | 1e-3 |
| Transition | 100-200 | 64 | 1e-4 |
| Trajectory | 50-100 | 256 | 1e-4 |
| End-to-End | 10-20 | 1024 | 1e-5 |

---

## Next Steps

- [Tutorial 3: Scenario Analysis](03-scenario-analysis.html)
- [Back to Tutorials](index.html)
