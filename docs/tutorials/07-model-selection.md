---
layout: default
title: Model Selection
parent: Tutorials
nav_order: 7
---

# Tutorial 7: Model Selection Guide

Learn when to use VAE, GAN, Flow, or Ensemble models for macro scenario generation.

## Overview

| Model | Best For | Avoid When |
|-------|----------|------------|
| **VAE** | Interpolation, fast inference | Need exact likelihood |
| **GAN** | Sharp samples, mode coverage | Training instability unacceptable |
| **Flow** | Exact likelihood, tail risk | Computational budget limited |
| **Ensemble** | Production, uncertainty | Interpretability required |

## 1. Model Characteristics

### Variational Autoencoder (VAE)

```python
from privatecredit.models import MacroVAE, MacroVAEConfig

config = MacroVAEConfig(
    n_macro_vars=9,
    seq_length=60,
    latent_dim=32,
    hidden_dim=128,
    n_scenarios=4
)

vae = MacroVAE(config)
print(f"VAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
```

**Strengths:**
- Smooth latent space enables interpolation
- Fast training and inference
- Stable optimization (ELBO objective)
- Good for scenario blending

**Weaknesses:**
- Can produce blurry/averaged samples
- Posterior collapse risk
- No exact likelihood

**Use When:**
- Need to interpolate between scenarios
- Fast prototyping
- Limited compute budget

### Wasserstein GAN (WGAN-GP)

```python
from privatecredit.models import MacroGAN, MacroGANConfig

config = MacroGANConfig(
    n_macro_vars=9,
    seq_length=60,
    latent_dim=64,
    hidden_dim=256,
    n_critic=5,
    lambda_gp=10.0
)

gan = MacroGAN(config)
print(f"GAN Parameters: {sum(p.numel() for p in gan.parameters()):,}")
```

**Strengths:**
- Sharp, realistic samples
- No mode averaging
- Good for capturing extreme scenarios
- Flexible architecture

**Weaknesses:**
- Training can be unstable
- Mode collapse possible
- No likelihood estimation
- Requires careful tuning

**Use When:**
- Need sharp scenario boundaries
- Sufficient training data
- Can invest in hyperparameter tuning

### Normalizing Flows (Real NVP)

```python
from privatecredit.models import MacroFlow, MacroFlowConfig

config = MacroFlowConfig(
    n_macro_vars=9,
    seq_length=60,
    n_coupling_layers=8,
    hidden_dim=128
)

flow = MacroFlow(config)
print(f"Flow Parameters: {sum(p.numel() for p in flow.parameters()):,}")
```

**Strengths:**
- Exact log-likelihood computation
- Invertible (can encode real data)
- No mode collapse
- Principled density estimation

**Weaknesses:**
- Higher computational cost
- Architectural constraints (invertibility)
- May struggle with complex multimodal distributions

**Use When:**
- Need exact likelihood for risk metrics
- Tail risk quantification critical
- Have sufficient compute resources

### Ensemble Model

```python
from privatecredit.models import MacroEnsemble, EnsembleConfig, EnsembleMethod

config = EnsembleConfig(
    n_macro_vars=9,
    seq_length=60,
    method=EnsembleMethod.WEIGHTED
)

ensemble = MacroEnsemble(
    config=config,
    vae_model=vae,
    gan_model=gan,
    flow_model=flow
)
```

**Strengths:**
- Robust to individual model failures
- Uncertainty quantification via disagreement
- Often best overall performance
- Production-ready

**Weaknesses:**
- Requires training all component models
- Higher memory/compute footprint
- Less interpretable

**Use When:**
- Production deployment
- Need uncertainty estimates
- Can afford computational overhead

## 2. Decision Framework

### Decision Tree

```
START
  |
  v
Need exact likelihood?
  |
  +--YES--> Flow
  |
  +--NO--> Need uncertainty quantification?
           |
           +--YES--> Ensemble
           |
           +--NO--> Training data > 10K samples?
                    |
                    +--YES--> Need sharp samples?
                    |         |
                    |         +--YES--> GAN
                    |         |
                    |         +--NO--> Need interpolation?
                    |                  |
                    |                  +--YES--> VAE
                    |                  |
                    |                  +--NO--> GAN or VAE
                    |
                    +--NO--> VAE (most stable with limited data)
```

### Quick Selection Guide

| Scenario | Recommended Model |
|----------|-------------------|
| Quick prototype | VAE |
| Production system | Ensemble |
| Tail risk analysis | Flow |
| Stress testing | GAN or Flow |
| Scenario interpolation | VAE |
| Limited data (<5K) | VAE |
| Abundant data (>50K) | GAN or Flow |
| Need confidence intervals | Ensemble |

## 3. Performance Comparison

### Training Speed

```python
import time
import torch

# Benchmark training time (single epoch)
def benchmark_training(model, data, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    start = time.time()

    for _ in range(epochs):
        for batch in data:
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

    return (time.time() - start) / epochs

# Results (relative to VAE = 1.0)
training_speed = {
    'VAE': 1.0,
    'GAN': 2.5,  # More iterations, discriminator
    'Flow': 1.8,  # Complex Jacobian computation
    'Ensemble': 5.5  # Train all three
}

print("Training Speed (relative):")
for model, speed in training_speed.items():
    print(f"  {model}: {speed:.1f}x VAE time")
```

### Inference Speed

```python
# Inference benchmarks (samples per second)
inference_speed = {
    'VAE': 10000,
    'GAN': 8000,
    'Flow': 3000,  # Sequential coupling layers
    'Ensemble': 2500  # Run all models
}

print("\nInference Speed (samples/sec):")
for model, speed in inference_speed.items():
    print(f"  {model}: {speed:,}")
```

### Memory Requirements

```python
# Memory footprint (MB for 1000 samples, batch_size=64)
memory_usage = {
    'VAE': 150,
    'GAN': 280,
    'Flow': 220,
    'Ensemble': 600
}

print("\nMemory Usage (MB):")
for model, mem in memory_usage.items():
    print(f"  {model}: {mem} MB")
```

## 4. Quality Metrics

### Evaluating Generated Samples

```python
from privatecredit.evaluation import ModelEvaluator

def evaluate_model(model, real_data, n_samples=1000):
    """Comprehensive model evaluation."""
    evaluator = ModelEvaluator()

    # Generate samples
    generated = model.generate(n_samples=n_samples)

    # Compute metrics
    metrics = {
        'mmd': evaluator.maximum_mean_discrepancy(real_data, generated),
        'wasserstein': evaluator.wasserstein_distance(real_data, generated),
        'correlation_error': evaluator.correlation_matrix_error(real_data, generated),
        'acf_error': evaluator.autocorrelation_error(real_data, generated),
        'coverage': evaluator.mode_coverage(real_data, generated),
    }

    return metrics

# Example evaluation results
evaluation_results = {
    'VAE': {'mmd': 0.15, 'wasserstein': 0.08, 'correlation_error': 0.05},
    'GAN': {'mmd': 0.12, 'wasserstein': 0.06, 'correlation_error': 0.07},
    'Flow': {'mmd': 0.10, 'wasserstein': 0.05, 'correlation_error': 0.04},
    'Ensemble': {'mmd': 0.08, 'wasserstein': 0.04, 'correlation_error': 0.03}
}
```

### Interpretation

| Metric | Good | Description |
|--------|------|-------------|
| MMD | < 0.1 | Distribution similarity |
| Wasserstein | < 0.05 | Earth mover's distance |
| Correlation Error | < 0.05 | Cross-variable dependencies |
| ACF Error | < 0.1 | Temporal dynamics |
| Coverage | > 0.9 | Mode coverage (avoid collapse) |

## 5. Ensemble Strategies

### Method Selection

```python
from privatecredit.models.ensemble import EnsembleMethod

# Simple averaging
ensemble_avg = MacroEnsemble(config, vae, gan, flow)
ensemble_avg.method = EnsembleMethod.AVERAGE

# Learned weights
ensemble_weighted = MacroEnsemble(config, vae, gan, flow)
ensemble_weighted.method = EnsembleMethod.WEIGHTED
ensemble_weighted.fit_weights(validation_data)

# Stacking (meta-learner)
ensemble_stacked = MacroEnsemble(config, vae, gan, flow)
ensemble_stacked.method = EnsembleMethod.STACKING
ensemble_stacked.fit_meta_learner(train_data, validation_data)

# Dynamic selection
ensemble_selection = MacroEnsemble(config, vae, gan, flow)
ensemble_selection.method = EnsembleMethod.SELECTION
```

### When to Use Each

| Method | Use When |
|--------|----------|
| AVERAGE | Quick baseline |
| WEIGHTED | Models have different strengths |
| STACKING | Have validation data |
| SELECTION | One model dominates per context |

## 6. Practical Recommendations

### For Different Use Cases

**Regulatory Stress Testing:**
```python
# Use Flow for exact likelihood required by regulators
# Or Ensemble for robustness
model = MacroFlow(config) if need_likelihood else MacroEnsemble(config)
```

**Portfolio Optimization:**
```python
# VAE for fast scenario generation during optimization
# Can generate millions of scenarios quickly
model = MacroVAE(config)
```

**Risk Reporting:**
```python
# Ensemble provides uncertainty bands for reports
model = MacroEnsemble(config)
samples, uncertainty = model.generate_with_uncertainty(n_samples=10000)
```

**Research/Backtesting:**
```python
# Flow for principled density estimation
# Allows log-likelihood comparisons
model = MacroFlow(config)
```

### Hyperparameter Guidelines

**VAE:**
```python
# Start conservative, increase if underfitting
vae_config = MacroVAEConfig(
    latent_dim=32,       # 16-64
    hidden_dim=128,      # 64-256
    n_layers=2,          # 2-4
    beta_start=0.0,      # KL annealing
    beta_end=1.0,
    beta_warmup=1000
)
```

**GAN:**
```python
# More discriminator updates for stability
gan_config = MacroGANConfig(
    latent_dim=64,       # 32-128
    hidden_dim=256,      # 128-512
    n_critic=5,          # 3-10
    lambda_gp=10.0,      # 1-20
    lr_g=1e-4,           # Generator LR
    lr_d=1e-4            # Discriminator LR
)
```

**Flow:**
```python
# More layers for expressivity
flow_config = MacroFlowConfig(
    n_coupling_layers=8,  # 4-16
    hidden_dim=128,       # 64-256
    use_batch_norm=True,
    use_actnorm=True
)
```

## 7. Migration Guide

### From VAE to Ensemble

```python
# Step 1: Keep VAE, train additional models
vae = MacroVAE(vae_config)
vae.load_state_dict(torch.load('vae_checkpoint.pt'))

gan = MacroGAN(gan_config)
flow = MacroFlow(flow_config)

# Step 2: Train new models
gan.fit(train_data)
flow.fit(train_data)

# Step 3: Create ensemble
ensemble = MacroEnsemble(ensemble_config, vae, gan, flow)

# Step 4: Validate improvement
metrics_before = evaluate_model(vae, test_data)
metrics_after = evaluate_model(ensemble, test_data)
```

## Summary

| Criterion | VAE | GAN | Flow | Ensemble |
|-----------|-----|-----|------|----------|
| Training Stability | +++ | + | ++ | ++ |
| Sample Quality | ++ | +++ | +++ | +++ |
| Inference Speed | +++ | ++ | + | + |
| Exact Likelihood | - | - | +++ | + |
| Uncertainty | + | + | ++ | +++ |
| Memory | + | ++ | ++ | +++ |

**Rule of Thumb:**
- Start with **VAE** for prototyping
- Move to **Ensemble** for production
- Use **Flow** when likelihood matters
- Use **GAN** when sample sharpness critical

**Next:** [Production Deployment](08-production-deployment.md)
