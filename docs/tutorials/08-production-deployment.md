---
layout: default
title: Production Deployment
parent: Tutorials
nav_order: 8
---

# Tutorial 8: Production Deployment Guide

Learn how to deploy Private Credit models to production environments.

## 1. Model Serialization

### Save Trained Models

```python
import torch
from pathlib import Path
from privatecredit.models import MacroVAE, MacroVAEConfig

# Train your model
config = MacroVAEConfig(n_macro_vars=9, seq_length=60, n_scenarios=4)
model = MacroVAE(config)
model.fit(train_data, epochs=100)

# Save model checkpoint
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

checkpoint = {
    'config': config.__dict__,
    'model_state_dict': model.state_dict(),
    'version': '1.0.0',
    'training_info': {
        'epochs': 100,
        'final_loss': model.last_loss,
        'timestamp': str(datetime.now())
    }
}

torch.save(checkpoint, checkpoint_dir / 'macro_vae_v1.pt')
print(f"Model saved to {checkpoint_dir / 'macro_vae_v1.pt'}")
```

### Load for Inference

```python
def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Recreate config
    config = MacroVAEConfig(**checkpoint['config'])

    # Initialize and load weights
    model = MacroVAE(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['training_info']

model, info = load_model('checkpoints/macro_vae_v1.pt')
print(f"Loaded model trained on {info['timestamp']}")
```

### Export to ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path, seq_length=60):
    """Export model to ONNX format for cross-platform deployment."""
    model.eval()

    # Create dummy input
    dummy_scenario = torch.tensor([0])  # Baseline scenario
    dummy_noise = torch.randn(1, model.config.latent_dim)

    # Export
    torch.onnx.export(
        model,
        (dummy_scenario, dummy_noise),
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['scenario', 'noise'],
        output_names=['generated_sequence'],
        dynamic_axes={
            'noise': {0: 'batch_size'},
            'generated_sequence': {0: 'batch_size'}
        }
    )
    print(f"ONNX model exported to {output_path}")

export_to_onnx(model, 'models/macro_vae.onnx')
```

## 2. FastAPI Deployment

### Basic API Server

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Optional

app = FastAPI(
    title="Private Credit API",
    description="Macro scenario generation and credit risk modeling",
    version="1.0.0"
)

# Load model at startup
model = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    model, _ = load_model('checkpoints/macro_vae_v1.pt')
    print("Model loaded successfully")

# Request/Response schemas
class ScenarioRequest(BaseModel):
    scenario_type: str = "baseline"  # baseline, adverse, severely_adverse, stagflation
    n_samples: int = 100
    seq_length: int = 60
    seed: Optional[int] = None

class ScenarioResponse(BaseModel):
    samples: List[List[List[float]]]  # (n_samples, seq_length, n_vars)
    metadata: dict

@app.post("/generate", response_model=ScenarioResponse)
async def generate_scenarios(request: ScenarioRequest):
    """Generate macro scenarios."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Map scenario names to indices
    scenario_map = {
        'baseline': 0,
        'adverse': 1,
        'severely_adverse': 2,
        'stagflation': 3
    }

    if request.scenario_type not in scenario_map:
        raise HTTPException(status_code=400, detail=f"Unknown scenario: {request.scenario_type}")

    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)

    # Generate
    scenario_idx = torch.tensor([scenario_map[request.scenario_type]])
    with torch.no_grad():
        samples = model.generate(
            scenario_idx,
            n_samples=request.n_samples,
            seq_length=request.seq_length
        )

    return ScenarioResponse(
        samples=samples.numpy().tolist(),
        metadata={
            'scenario': request.scenario_type,
            'n_samples': request.n_samples,
            'seq_length': request.seq_length,
            'model_version': '1.0.0'
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
```

### Run the Server

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Client Usage

```python
import requests

# Generate scenarios
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "scenario_type": "adverse",
        "n_samples": 100,
        "seq_length": 60
    }
)

data = response.json()
samples = np.array(data['samples'])
print(f"Generated {samples.shape[0]} scenarios")
```

## 3. Batch Inference Optimization

### Efficient Batch Processing

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

class BatchInferenceEngine:
    """Optimized batch inference for production."""

    def __init__(self, model, device='cuda', batch_size=256):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def generate_large_sample(self, scenario_idx, n_samples):
        """Generate large number of samples efficiently."""
        all_samples = []

        # Process in batches
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in range(n_batches):
                batch_size = min(self.batch_size, n_samples - i * self.batch_size)
                scenario = torch.tensor([scenario_idx]).to(self.device)

                samples = self.model.generate(scenario, n_samples=batch_size)
                all_samples.append(samples.cpu())

        return torch.cat(all_samples, dim=0)

    @torch.inference_mode()
    def generate_parallel(self, scenarios, n_samples_per_scenario):
        """Generate for multiple scenarios in parallel."""
        results = {}

        for name, idx in scenarios.items():
            scenario = torch.tensor([idx]).to(self.device)
            samples = self.model.generate(scenario, n_samples=n_samples_per_scenario)
            results[name] = samples.cpu()

        return results

# Usage
engine = BatchInferenceEngine(model, device='cuda', batch_size=512)

# Generate 1M samples efficiently
samples = engine.generate_large_sample(scenario_idx=0, n_samples=1_000_000)
print(f"Generated {samples.shape[0]} samples")
```

### GPU Memory Management

```python
def generate_with_memory_limit(model, n_samples, max_memory_gb=8):
    """Generate samples while respecting GPU memory limit."""

    # Estimate memory per sample (in GB)
    sample_size_bytes = (
        model.config.seq_length *
        model.config.n_macro_vars *
        4  # float32
    )
    memory_per_sample_gb = sample_size_bytes / 1e9

    # Calculate safe batch size
    safe_batch_size = int(max_memory_gb * 0.5 / memory_per_sample_gb)
    safe_batch_size = min(safe_batch_size, 10000)  # Cap

    print(f"Using batch size: {safe_batch_size}")

    # Generate
    all_samples = []
    for i in range(0, n_samples, safe_batch_size):
        batch_size = min(safe_batch_size, n_samples - i)
        with torch.no_grad():
            samples = model.generate(scenario, n_samples=batch_size)
            all_samples.append(samples.cpu())

        # Clear cache
        torch.cuda.empty_cache()

    return torch.cat(all_samples, dim=0)
```

## 4. Monitoring and Logging

### Model Performance Monitoring

```python
import logging
from datetime import datetime
import json

class ModelMonitor:
    """Monitor model performance in production."""

    def __init__(self, model_name, log_path='logs/model_monitor.jsonl'):
        self.model_name = model_name
        self.log_path = log_path
        self.logger = logging.getLogger(model_name)

        # Metrics storage
        self.inference_times = []
        self.request_counts = {}

    def log_inference(self, scenario, n_samples, inference_time, output_stats):
        """Log inference request."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'scenario': scenario,
            'n_samples': n_samples,
            'inference_time_ms': inference_time * 1000,
            'output_mean': float(output_stats['mean']),
            'output_std': float(output_stats['std'])
        }

        # Write to log file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Update metrics
        self.inference_times.append(inference_time)
        self.request_counts[scenario] = self.request_counts.get(scenario, 0) + 1

    def get_statistics(self):
        """Get monitoring statistics."""
        return {
            'total_requests': sum(self.request_counts.values()),
            'requests_by_scenario': self.request_counts,
            'avg_inference_time_ms': np.mean(self.inference_times) * 1000,
            'p95_inference_time_ms': np.percentile(self.inference_times, 95) * 1000
        }

# Usage with API
monitor = ModelMonitor('macro_vae_v1')

@app.post("/generate")
async def generate_scenarios(request: ScenarioRequest):
    start_time = time.time()

    # Generate samples
    samples = model.generate(...)

    # Log
    inference_time = time.time() - start_time
    monitor.log_inference(
        scenario=request.scenario_type,
        n_samples=request.n_samples,
        inference_time=inference_time,
        output_stats={'mean': samples.mean(), 'std': samples.std()}
    )

    return response
```

### Alerting on Drift

```python
class DriftDetector:
    """Detect model drift in production."""

    def __init__(self, reference_stats):
        self.reference = reference_stats
        self.alert_threshold = 0.1  # 10% deviation

    def check_drift(self, current_stats):
        """Check for significant drift from reference."""
        alerts = []

        for key in self.reference:
            if key in current_stats:
                ref_val = self.reference[key]
                cur_val = current_stats[key]
                pct_change = abs(cur_val - ref_val) / (ref_val + 1e-10)

                if pct_change > self.alert_threshold:
                    alerts.append({
                        'metric': key,
                        'reference': ref_val,
                        'current': cur_val,
                        'pct_change': pct_change
                    })

        return alerts

# Initialize with baseline stats
reference_stats = {
    'gdp_mean': 0.025,
    'gdp_std': 0.015,
    'unemployment_mean': 0.045
}
drift_detector = DriftDetector(reference_stats)
```

## 5. Retraining Pipeline

### Trigger Conditions

```python
class RetrainingTrigger:
    """Determine when to retrain model."""

    def __init__(self, config):
        self.config = config
        self.drift_threshold = config.get('drift_threshold', 0.15)
        self.min_days_between_retraining = config.get('min_days', 30)
        self.last_retrain_date = None

    def should_retrain(self, drift_alerts, performance_metrics):
        """Check if retraining is needed."""
        reasons = []

        # Check drift
        if len(drift_alerts) > 0:
            max_drift = max(a['pct_change'] for a in drift_alerts)
            if max_drift > self.drift_threshold:
                reasons.append(f"Drift detected: {max_drift:.1%}")

        # Check performance degradation
        if performance_metrics.get('brier_score', 0) > 0.1:
            reasons.append(f"Performance degradation: Brier={performance_metrics['brier_score']:.3f}")

        # Check time since last retrain
        if self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since > 90:  # Retrain at least quarterly
                reasons.append(f"Scheduled retrain (last: {days_since} days ago)")

        return len(reasons) > 0, reasons
```

### Automated Retraining

```python
def retrain_pipeline(model_path, new_data_path, output_path):
    """Automated retraining pipeline."""

    # Load current model
    current_model, info = load_model(model_path)

    # Load new data
    new_data = pd.read_parquet(new_data_path)

    # Combine with historical data
    train_data = prepare_training_data(new_data)

    # Initialize new model with same config
    config = MacroVAEConfig(**info['config'])
    new_model = MacroVAE(config)

    # Train
    new_model.fit(train_data, epochs=100)

    # Validate
    val_metrics = validate_model(new_model, validation_data)

    # Only deploy if better
    if val_metrics['brier_score'] < info['metrics']['brier_score']:
        # Save new model
        save_checkpoint(new_model, output_path)
        print(f"New model deployed: Brier improved from "
              f"{info['metrics']['brier_score']:.4f} to {val_metrics['brier_score']:.4f}")
        return True
    else:
        print("New model not better, keeping current")
        return False
```

## 6. Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY privatecredit/ ./privatecredit/
COPY api/ ./api/
COPY checkpoints/ ./checkpoints/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
    environment:
      - MODEL_PATH=/app/checkpoints/macro_vae_v1.pt
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

### Build and Run

```bash
# Build image
docker build -t privatecredit-api:v1 .

# Run container
docker run -d \
    --name privatecredit \
    -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    --gpus all \
    privatecredit-api:v1

# Check logs
docker logs -f privatecredit
```

## Summary

| Stage | Key Actions |
|-------|-------------|
| **Serialization** | Save checkpoints, export ONNX |
| **API** | FastAPI with health checks |
| **Optimization** | Batch inference, GPU management |
| **Monitoring** | Log requests, detect drift |
| **Retraining** | Automated pipelines |
| **Deployment** | Docker, Kubernetes |

**Production Checklist:**
- [ ] Model versioning in place
- [ ] Health check endpoints working
- [ ] Logging and monitoring configured
- [ ] Drift detection alerts set up
- [ ] Retraining pipeline tested
- [ ] Load testing completed
- [ ] Rollback procedure documented
